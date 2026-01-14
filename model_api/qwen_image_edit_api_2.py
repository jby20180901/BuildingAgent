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
import json
import asyncio

# === 模型单例 ===
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

# === 并发控制：最多 4 个推理同时进行 ===
inference_semaphore = asyncio.Semaphore(4)

# === 同步推理函数（在线程中运行）===
def run_inference_sync(inputs):
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

# === FastAPI App ===
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
    request: str = Form(...),
    images: List[UploadFile] = File(...)
):
    try:
        # 解析参数
        request_data = json.loads(request)
        request_obj = ImageEditRequest(**request_data)

        # 读取图像（异步）
        input_images = []
        for img_file in images:
            img_bytes = await img_file.read()
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            input_images.append(img)

        print(request_obj.prompt)

        # 构造输入
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

        # 获取信号量（控制并发）
        await inference_semaphore.acquire()
        try:
            # 在线程池中运行同步推理（不阻塞事件循环）
            loop = asyncio.get_event_loop()
            img_str = await loop.run_in_executor(None, run_inference_sync, inputs)
        finally:
            inference_semaphore.release()  # 释放信号量

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
    uvicorn.run(app, host="0.0.0.0", port=8024)