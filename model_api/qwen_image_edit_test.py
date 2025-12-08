import requests
import base64
from PIL import Image
from io import BytesIO
import os
import time
import json  

# API配置
API_URL = "http://localhost:8022/edit-image"
IMAGE_URLS = [
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg"
]
OUTPUT_PATH = "output_image_edit_plus.png"

def download_image(url):
    response = requests.get(url)
    return BytesIO(response.content)

def test_api():
    # 1. 健康检查
    print("Checking API health...")
    health_url = API_URL.replace("/edit-image", "/health")
    try:
        health_res = requests.get(health_url, timeout=5)
        print(f"Health check: {health_res.status_code} - {health_res.json()}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        return

    # 2. 准备测试数据
    print("\nPreparing test data...")
    img_bytes_list = []
    for i, url in enumerate(IMAGE_URLS):
        try:
            response = requests.get(url)
            img_bytes = BytesIO(response.content)
            img = Image.open(img_bytes)
            print(f"Image {i+1}: {img.size} {img.mode}")
            img_bytes.seek(0)  # 重置指针
            img_bytes_list.append(img_bytes.getvalue())  # 保存 bytes
        except Exception as e:
            print(f"Failed to download image {url}: {e}")
            return

    # 3. 准备请求数据
    prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
    payload = {
        "prompt": prompt,
        "negative_prompt": " ",
        "seed": 0,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "num_inference_steps": 40,
        "num_images_per_prompt": 1,
    }

    # 4. 发送请求
    print("\nSending edit request...")
    start_time = time.time()
    try:
        files = [('images', (f"image_{i}.jpg", img_bytes, 'image/jpeg')) for i, img_bytes in enumerate(img_bytes_list)]
        
        # 直接将 payload 作为 form-data 的一部分
        response = requests.post(
            API_URL,
            data={"request": json.dumps(payload)},
            files=files,
            timeout=300
        )
        
        elapsed = time.time() - start_time
        print(f"Request completed in {elapsed:.2f}s")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                img_data = base64.b64decode(result["image_base64"])
                img = Image.open(BytesIO(img_data))
                img.save(OUTPUT_PATH)
                print(f"\n✅ Success! Image saved to {OUTPUT_PATH}")
                print(f"   Seed used: {result.get('seed', 'N/A')}")
                print(f"   Image size: {img.size}")
                try:
                    img.show()
                except:
                    print("Note: Could not display image, please check the output file")
            else:
                print("\n❌ Failed:", result.get("error", "Unknown error"))
        else:
            print("\n❌ Request failed:", response.text)

    except requests.exceptions.Timeout:
        print("\n❌ Request timed out")
    except Exception as e:
        print(f"\n❌ Error during request: {e}")
if __name__ == "__main__":
    test_api()