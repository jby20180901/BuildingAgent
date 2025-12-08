import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8021/generate"
payload = {
    "prompt": "生成一个高保真的3-5层的巴洛克建筑，要求立体图视角，能够看到建筑的全貌，真实风格",
    "negative_prompt": "",
    "seed": 5678,
    "randomize_seed": False,
    "aspect_ratio": "16:9",
    "guidance_scale": 5.0, # 提示词引导强度
    # 数值越高 → 模型更严格遵循 prompt 描述，但可能导致画面失真或不自然
    # 数值较低 → 模型更自由生成，可能偏离 prompt 描述，但画面更自然。
    "num_inference_steps": 50 # 扩散模型的推理步数
    # 步数越多 → 图像质量越高、细节更丰富，但生成时间更长
    # 步数过少 → 生成速度快，但可能模糊或细节不足
}

res = requests.post(url, json=payload)
data = res.json()

if data["success"]:
    img_data = base64.b64decode(data["image_base64"])
    img = Image.open(BytesIO(img_data))
    img.save("result.png")
    print("Saved to result.png")
else:
    print("Generation failed:", data)
