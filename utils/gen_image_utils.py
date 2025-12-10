import requests
import base64
from PIL import Image
from io import BytesIO
from typing import Optional, Union


def call_gen_image_api(
        prompt: str,
        output_path: str = "result.png",
        url: str = "http://localhost:8021/generate",
        negative_prompt: str = "",
        seed: int = 5678,
        randomize_seed: bool = False,
        aspect_ratio: str = "16:9",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        save_image: bool = True
) -> Optional[Image.Image]:
    """
    生成AI图像

    Args:
        prompt: 图像生成的提示词
        output_path: 输出图像的保存路径，默认 "result.png"
        url: API服务地址，默认 "http://localhost:8021/generate"
        negative_prompt: 负面提示词，默认为空
        seed: 随机种子，默认 5678
        randomize_seed: 是否随机化种子，默认 False
        aspect_ratio: 图像宽高比，默认 "16:9"
        guidance_scale: 提示词引导强度 (越高越严格遵循prompt)，默认 5.0
        num_inference_steps: 推理步数 (越多质量越高但速度越慢)，默认 50
        save_image: 是否保存图像到文件，默认 True

    Returns:
        PIL.Image对象，如果失败则返回None
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "aspect_ratio": aspect_ratio,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps
    }

    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()  # 检查HTTP错误
        data = res.json()

        if data.get("success"):
            img_data = base64.b64decode(data["image_base64"])
            img = Image.open(BytesIO(img_data))

            if save_image:
                img.save(output_path)
                print(f"图像已保存到: {output_path}")

            return img
        else:
            print("生成失败:", data)
            return None

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except Exception as e:
        print(f"处理错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 基础使用
    img = call_gen_image_api(
        prompt="生成一个高保真的3-5层的巴洛克建筑，要求立体图视角，能够看到建筑的全貌，真实风格"
    )

    # 自定义参数
    img = call_gen_image_api(
        prompt="一只可爱的猫咪",
        output_path="cat.png",
        aspect_ratio="1:1",
        guidance_scale=7.0,
        num_inference_steps=30
    )

    # 只获取图像对象，不保存
    img = call_gen_image_api(
        prompt="美丽的风景",
        save_image=False
    )
    if img:
        img.show()  # 直接显示图像
