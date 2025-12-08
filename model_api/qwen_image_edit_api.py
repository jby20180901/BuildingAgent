import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List
import uvicorn
from io import BytesIO
import base64
import numpy as np
import json  # <-- 新增

# 初始化模型
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def load_model():
    global pipeline
    if pipeline is None:
        print("Loading Qwen-Image-Edit-Plus model...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen-Image-Edit-2509", 
            torch_dtype=torch_dtype
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=None)
        print("Model loaded successfully")

# FastAPI应用
app = FastAPI(title="Qwen Image Edit Plus API")

class ImageEditRequest(BaseModel):
    prompt: str
    negative_prompt: str = " "
    seed: int = 0
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    num_inference_steps: int = 40
    num_images_per_prompt: int = 1

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/edit-image")
async def edit_image(
    request: str = Form(...),  # <-- 修改：接收字符串
    images: List[UploadFile] = File(...)
):
    try:
        # 手动解析 request 字符串为 Pydantic 模型
        request_data = json.loads(request)
        request_obj = ImageEditRequest(**request_data)  # 验证并创建实例

        # 读取上传的图片
        input_images = []
        for img_file in images:
            img_bytes = await img_file.read()
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            input_images.append(img)
        
        if len(input_images) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images are required")

        # 准备输入参数（使用 request_obj）
        inputs = {
            "image": input_images,
            "prompt": request_obj.prompt,
            "generator": torch.manual_seed(request_obj.seed),
            "true_cfg_scale": request_obj.true_cfg_scale,
            "negative_prompt": request_obj.negative_prompt,
            "num_inference_steps": request_obj.num_inference_steps,
            "guidance_scale": request_obj.guidance_scale,
            "num_images_per_prompt": request_obj.num_images_per_prompt,
        }

        # 执行图片编辑
        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
            
            # 将结果转换为base64
            buffered = BytesIO()
            output_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "image_base64": img_str,
            "seed": request_obj.seed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8022)