import os
import uuid
import tempfile
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import sys
import torch
import imageio

# 获取当前脚本所在的目录，然后向上回退到项目的根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils

# 配置环境变量（避免每次请求重复设置）
os.environ['SPCONV_ALGO'] = 'native'

app = FastAPI(title="3D Model Generator API")

# 模型初始化（启动时加载，避免每次请求加载）
try:
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/TRELLIS-image-large"
    )
    pipeline.cuda()
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Model initialization failed: {str(e)}") from e


@app.post("/generate-3d/")
async def generate_3d(file: UploadFile = File(...)):
    """
    上传图片 → 生成3D模型（GLB/Mesh/视频）
    返回包含所有生成文件的ZIP压缩包
    """
    # 1. 保存上传的图片到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        temp_image_path = tmp.name

    try:
        image = Image.open(temp_image_path)
        outputs = pipeline.run(image, seed=1)

        # 2. 创建唯一ID和临时目录用于存放所有生成的文件
        unique_id = uuid.uuid4().hex[:8]
        output_dir = tempfile.mkdtemp()

        generated_files = []

        # 3. 仅 gaussian 支持 .save_ply()
        if 'gaussian' in outputs:
            gauss_obj = outputs['gaussian'][0]
            if hasattr(gauss_obj, 'save_ply'):
                ply_path = os.path.join(output_dir, f"gaussian_{unique_id}.ply")
                gauss_obj.save_ply(ply_path)
                generated_files.append(ply_path)
            else:
                print("⚠️ Gaussian object does not have 'save_ply' method")

        # 4. 渲染视频：gaussian, radiance_field, mesh
        for asset_type in ['gaussian', 'radiance_field', 'mesh']:
            if asset_type in outputs:
                try:
                    video = render_utils.render_video(outputs[asset_type][0])['color']
                    mp4_path = os.path.join(output_dir, f"{asset_type}_{unique_id}.mp4")
                    imageio.mimsave(
                        mp4_path,
                        video,
                        fps=30,
                        format='ffmpeg',
                        codec='libx264',
                        pixelformat='yuv420p'
                    )
                    if 'gaussian' in outputs:
                        generated_files.append(mp4_path)
                except Exception as e:
                    print(f"⚠️ Failed to render video for {asset_type}: {e}")

        # 5. 导出 GLB（需要 gaussian + mesh）
        if 'gaussian' in outputs and 'mesh' in outputs:
            try:
                glb_path = os.path.join(output_dir, f"model_{unique_id}.glb")
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024
                )
                glb.export(glb_path)
                generated_files.append(glb_path)
            except Exception as e:
                print(f"⚠️ Failed to export GLB: {e}")

        # 6. 打包所有生成的文件（只包含实际存在的 .glb, .ply, .mp4）
        zip_file_path = os.path.join(output_dir, f"3d_model_assets_{unique_id}.zip")
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for file_path in generated_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    zipf.write(file_path, filename)

        return FileResponse(
            zip_file_path,
            media_type="application/octet-stream",
            filename=f"3d_model_assets_{unique_id}.zip"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    finally:
        # 清理临时图片
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.3.3.1", port=8031)