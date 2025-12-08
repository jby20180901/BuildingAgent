import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import sys
import torch

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
    返回生成的GLB文件
    """
    # 1. 保存上传的图片到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        temp_image_path = tmp.name

    # 2. 生成3D模型
    try:
        image = Image.open(temp_image_path)
        outputs = pipeline.run(
            image,
            seed=1,
            # 可扩展参数：如添加 cfg_strength 等
        )

        # 3. 生成临时文件名（避免冲突）
        unique_id = uuid.uuid4().hex[:8]
        output_dir = tempfile.mkdtemp()
        glb_path = os.path.join(output_dir, f"model_{unique_id}.glb")

        # 4. 保存GLB文件
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024
        )
        glb.export(glb_path)

        # 5. 返回GLB文件（自动清理临时文件）
        return FileResponse(
            glb_path,
            media_type="application/octet-stream",
            filename=f"3d_model_{unique_id}.glb"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        ) from e

    finally:
        # 清理临时图片
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.3.3.1", port=8031)