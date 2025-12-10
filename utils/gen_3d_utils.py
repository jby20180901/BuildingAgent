import requests
import os
import uuid
from typing import Optional


def call_gen_3d_api(
        image_path: str,
        output_path: Optional[str] = None,
        api_url: str = "http://10.3.3.1:8031/generate-3d/"
) -> Optional[str]:
    """
    上传图片生成3D模型（GLB/ZIP）文件。

    Args:
        image_path (str): 输入图片的本地路径。
        output_path (str, optional): 保存结果的文件路径。如果为None，将自动生成文件名。
        api_url (str): API接口地址。

    Returns:
        Optional[str]: 成功时返回保存的文件路径，失败时返回 None。
    """

    # 1. 检查输入文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误: 找不到输入图片: {image_path}")
        return None

    try:
        # 2. 准备请求数据
        filename = os.path.basename(image_path)
        # 根据后缀简单推断MIME类型，默认为 image/png
        mime_type = "image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"

        with open(image_path, "rb") as f:
            # files参数格式: {'字段名': (文件名, 文件对象, MIME类型)}
            files = {"file": (filename, f, mime_type)}

            print(f"正在上传图片 '{filename}' 到 {api_url} ...")
            response = requests.post(api_url, files=files, timeout=300)  # 设置超时防止无限等待

        # 3. 处理响应
        if response.status_code == 200:
            # 如果未指定输出路径，生成一个随机文件名 (generated_model_xxxx.zip)
            if not output_path:
                random_suffix = uuid.uuid4().hex[:8]
                # 假设返回的是zip，如果不确定可以稍后改为根据header判断
                output_path = f"generated_model_{random_suffix}.zip"

                # 自动创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存文件
            with open(output_path, "wb") as f:
                f.write(response.content)

            file_size = os.path.getsize(output_path)
            print(f"✅ 生成成功! 模型已保存至: {output_path}")
            print(f"文件大小: {file_size / 1024:.2f} KB")
            return output_path

        else:
            print(f"❌ 请求失败 (状态码: {response.status_code})")
            print("响应内容:", response.text)
            return None

    except requests.exceptions.ConnectionError:
        print(f"❌ 网络错误: 无法连接到服务 {api_url}，请检查服务是否启动或IP是否正确。")
        return None
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return None


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置你的测试图片路径
    TEST_IMG = "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/result.png"

    # 1. 基础调用（自动命名输出文件）
    result_path = call_gen_3d_api(TEST_IMG)

    # 2. 指定输出路径和自定义API地址（如果有变化）
    # result_custom = generate_3d_from_image(
    #     image_path=TEST_IMG,
    #     output_path="./models/my_building.zip",
    #     api_url="http://10.3.3.1:8031/generate-3d/"
    # )
