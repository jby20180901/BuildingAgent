import requests
import os

# 本地测试用：确保服务已运行在 http://localhost:8000
API_URL = "http://10.3.3.1:8031/generate-3d/"

def test_3d_generation():
    """测试API：上传图片并下载生成的GLB文件"""
    # 准备测试图片（替换为你的测试图路径）
    test_image_path = "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/result.png"  # 确保存在此文件
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at {test_image_path}")

    # 发送POST请求
    with open(test_image_path, "rb") as f:
        files = {"file": (os.path.basename(test_image_path), f, "image/png")}
        response = requests.post(API_URL, files=files)

    # 检查响应
    if response.status_code == 200:
        # 保存返回的GLB文件
        output_path = f"generated_model_{os.urandom(4).hex()}.zip"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Success! Model saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"❌ Failed with status code: {response.status_code}")
        print("Response:", response.text)

if __name__ == "__main__":
    test_3d_generation()