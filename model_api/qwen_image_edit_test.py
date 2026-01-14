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
    "/home/jiangbaoyang/GitHub/Skyfall-GS/outputs/JAX_idu/JAX_068/idu/e90.0_r320.0/render/00091.png",
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
    for i, path in enumerate(IMAGE_URLS):
        try:
            with open(path, "rb") as f:
                img_data = f.read()
            img_bytes = BytesIO(img_data)
            img = Image.open(img_bytes)
            print(f"Image {i+1}: {img.size} {img.mode}")
            img_bytes_list.append(img_data)  # 直接保存 bytes
        except Exception as e:
            print(f"Failed to load image from {path}: {e}")
            return

    # 3. 准备请求数据
    prompt = """在修复一张低质量航拍或三维场景图像时，请以生成高质量、接近真实摄影级的4K分辨率（3840×2160）图像为目标。注意以下指导原则：
保持原始光照和色彩风格：尽量保留原图中的光影细节、颜色和对比度，确保整体色调和谐一致。
重点减少锯齿和毛刺伪影：特别关注并消除图像中出现的锯齿边缘、毛刺、色带及任何其他形式的视觉伪影。使用高级算法和技术使这些区域变得尽可能平滑自然，避免产生新的异常现象。
维持结构稳定性：确保主要地物的位置、形状、比例关系基本不变形，保持原图的空间逻辑。
合理增强细节：对不清晰的部分进行适当补充，但避免过度锐化导致人工痕迹明显。重点关注如何在提升细节清晰度的同时保证过渡区间的柔和性。
呈现自然摄影风格：使最终结果看起来像是通过专业设备拍摄的照片，而非经过大量后期处理的作品。特别注重图像的整体和谐感和真实感。
移除非地理元素：清理水印、文字说明等不属于实景的内容。
输出高分辨率图像：利用先进的超分辨率技术提升图片清晰度，同时保证边缘柔和、过渡自然，特别注意图像各部分之间的无缝连接。
核心强调点
“请特别注意使用高级算法和技术来平滑锯齿边缘和毛刺伪影，使它们尽可能不可见，同时确保整个图像看起来自然而真实。”
"""
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