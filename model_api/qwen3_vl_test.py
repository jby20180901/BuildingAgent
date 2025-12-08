import requests
import json
import base64
from PIL import Image
import io

def compress_image(image_path, max_size=(800, 600), quality=80):
    """
    压缩图片以减少token使用量
    """
    with Image.open(image_path) as img:
        # 转换为RGB模式（如果是RGBA或其他模式）
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # 调整图片大小，保持宽高比
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 将图片保存到内存中
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()

def encode_image_to_base64_compressed(image_path, max_size=(800, 600), quality=80):
    """
    压缩图片并转换为Base64编码字符串。
    """
    compressed_image_data = compress_image(image_path, max_size, quality)
    return base64.b64encode(compressed_image_data).decode('utf-8')

def send_multimodal_request(text_prompt, image_paths, model_name="/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen3-VL-32B-Thinking", base_url="http://localhost:8012"):
    """
    向本地运行的多模态模型服务器发送图文混合请求。
    使用标准的OpenAI API格式
    """
    # 构建 content：使用标准的多模态格式
    content = [{"type": "text", "text": text_prompt}]
    
    for i, path in enumerate(image_paths):
        base64_img = encode_image_to_base64_compressed(path, max_size=(800, 600), quality=80)
        # 使用标准的image_url格式，但包含base64数据
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 100000
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# 示例用法
if __name__ == "__main__":
    prompt = "Describe these images in one sentence."
    images = [
        "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/edit2509_1.jpg",
        "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/edit2509_2.jpg"
    ]

    try:
        result = send_multimodal_request(prompt, images)
        # 提取助手回复内容
        reply = result['choices'][0]['message']['content']
        print("Model response:", reply)
    except Exception as e:
        print("Error:", e)