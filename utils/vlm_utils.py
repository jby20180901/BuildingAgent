import base64
import io
import json
import os
import sys

import cv2  # OpenCV for video processing
import requests
from PIL import Image


# --- 1. 辅助函数：媒体处理与编码 ---

def _compress_image_data(image_data: bytes, max_size: tuple = (1024, 1024), quality: int = 75) -> bytes:
    """
    接收图片字节数据，进行压缩和调整大小。

    Args:
        image_data (bytes): 原始图片的字节数据。
        max_size (tuple, optional): 图片的最大尺寸 (宽, 高)。默认为 (1024, 1024)。
        quality (int, optional): JPEG压缩质量 (1-95)。默认为 75。

    Returns:
        bytes: 压缩后的JPEG图片字节数据。
    """
    with Image.open(io.BytesIO(image_data)) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()


def _encode_image_to_base64(image_path: str, max_size: tuple = (1024, 1024), quality: int = 75) -> str:
    """
    读取、压缩本地图片并编码为Base64字符串。
    """
    with open(image_path, "rb") as image_file:
        compressed_data = _compress_image_data(image_file.read(), max_size, quality)
        return base64.b64encode(compressed_data).decode('utf-8')


def _process_video_to_base64_frames(
        video_path: str,
        sample_rate_hz: int = 1,
        max_frames: int = 20,
        max_size: tuple = (1024, 1024),
        quality: int = 75
) -> list[str]:
    """
    从视频文件中按指定频率提取帧，压缩后编码为Base64字符串列表。

    Args:
        video_path (str): 视频文件路径。
        sample_rate_hz (int, optional): 每秒采样多少帧。默认为 1。
        max_frames (int, optional): 最多提取的帧数，防止视频过长。默认为 20。
        max_size (tuple, optional): 每帧图像的最大尺寸。默认为 (1024, 1024)。
        quality (int, optional): 每帧图像的压缩质量。默认为 75。

    Returns:
        list[str]: 包含多张Base64编码帧的列表。
    """
    base64_frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告: 无法打开视频文件 {video_path}，将跳过。")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / sample_rate_hz))

    frame_count = 0
    captured_frames = 0

    print(f"正在从 {os.path.basename(video_path)} 提取帧... (采样率: {sample_rate_hz}Hz, 最多: {max_frames}帧)")

    try:
        while cap.isOpened() and captured_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # OpenCV 读取的是 BGR, Pillow 需要 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 将 numpy array 转换为 Pillow Image
                img = Image.fromarray(frame_rgb)

                # Pillow Image -> bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')  # 先无损保存到内存

                # 压缩并编码
                compressed_data = _compress_image_data(buffer.getvalue(), max_size, quality)
                base64_frames.append(base64.b64encode(compressed_data).decode('utf-8'))

                captured_frames += 1
                # 实时打印进度
                print(f"\r -> 已提取 {captured_frames}/{max_frames} 帧...", end="")

            frame_count += 1
    finally:
        cap.release()
        print("\n视频帧提取完成。")

    return base64_frames


# --- 2. 主函数：统一的多模态API调用 ---

def call_vlm_api (
        text_prompt: str,
        media_paths: list[str],
        model_name: str,
        base_url: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        # 媒体处理选项
        image_max_size: tuple = (1024, 1024),
        image_quality: int = 75,
        video_sample_rate_hz: int = 1,
        video_max_frames: int = 20
) -> str | None:
    """
    向本地多模态LLM发送一个包含文本、图像和视频的请求。

    Args:
        text_prompt (str): 文本提示。
        media_paths (list[str]): 包含图像和/或视频文件路径的列表。
        model_name (str): 要调用的模型名称。
        base_url (str): 模型服务的根URL (例如 "http://localhost:8012")。
        max_tokens (int, optional): 最大生成token数。 Defaults to 8192。
        temperature (float, optional): 温度。 Defaults to 0.7。
        stream (bool, optional): 是否流式返回。 Defaults to False。
        image_max_size (tuple, optional): 图片压缩尺寸。 Defaults to (1024, 1024)。
        image_quality (int, optional): 图片压缩质量。 Defaults to 75。
        video_sample_rate_hz (int, optional): 视频采样率(帧/秒)。 Defaults to 1。
        video_max_frames (int, optional): 单个视频最大采样帧数。 Defaults to 20。

    Returns:
        str | None: 返回模型的文本响应。如果出错则返回None。
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}

    # 1. 构建 payload 的 content 部分
    content = [{"type": "text", "text": text_prompt}]

    for path in media_paths:
        if not os.path.exists(path):
            print(f"警告: 文件不存在 {path}，将跳过。")
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            print(f"正在处理图片: {os.path.basename(path)}")
            base64_media = _encode_image_to_base64(path, image_max_size, image_quality)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_media}"}
            })

        elif ext in VIDEO_EXTENSIONS:
            print(f"正在处理视频: {os.path.basename(path)}")
            base64_frames = _process_video_to_base64_frames(
                path, video_sample_rate_hz, video_max_frames, image_max_size, image_quality
            )
            for frame in base64_frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })
        else:
            print(f"警告: 不支持的文件类型 {ext}，文件 {path} 将被跳过。")

    # 2. 准备请求
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    headers = {"Content-Type": "application/json"}
    api_url = f"{base_url}/v1/chat/completions"

    # 3. 发送请求并处理响应
    try:
        print(f"\n正在向 {api_url} 发送请求...")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=stream)
        response.raise_for_status()  # 如果状态码不是 2xx，则抛出异常

        full_response_content = ""

        if stream:
            print("模型流式返回结果:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        # 移除 "data: " 前缀
                        json_str = decoded_line[6:]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(json_str)
                            delta_content = chunk['choices'][0]['delta'].get('content', '')
                            if delta_content:
                                print(delta_content, end="")
                                sys.stdout.flush()
                                full_response_content += delta_content
                        except json.JSONDecodeError:
                            print(f"\n警告: 无法解析流中的JSON行: {json_str}")
            print("\n--- 流式输出结束 ---")
            return full_response_content

        else:  # 非流式
            print("模型正在一次性生成结果...")
            result = response.json()
            full_response_content = result['choices'][0]['message']['content']
            return full_response_content

    except requests.exceptions.RequestException as e:
        print(f"\n请求失败: {e}")
        # 尝试打印更详细的错误信息
        if e.response is not None:
            print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"\n处理请求时发生未知错误: {e}")
        return None


# --- 3. 示例用法 ---
if __name__ == "__main__":

    # --- 配置 ---
    # 请确保vLLM服务正在此地址运行
    MODEL_BASE_URL = "http://localhost:8012/v1"
    MODEL_TO_USE = "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen3-VL-32B-Thinking"

    # --- 准备输入 ---
    # 你的文本提示
    prompt_text = "请详细描述这些媒体内容。首先总结图片，然后总结视频里的动态过程。"

    # 你的媒体文件列表 (包含图片和视频)
    # **请替换为真实有效的文件路径**
    media_file_paths = [
        "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/edit2509_1.jpg",
        "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/edit2509_2.jpg",
        # "/path/to/your/video.mp4" # <--- 在此处添加你的视频文件路径
    ]
    # 如果你没有视频文件用于测试，可以注释掉视频路径
    # 如果路径无效，程序会打印警告并跳过

    print("=" * 50)
    print("示例 1: 流式调用")
    print("=" * 50)

    try:
        model_reply = call_vlm_api(text_prompt=prompt_text, media_paths=media_file_paths, model_name=MODEL_TO_USE,
                                   base_url=MODEL_BASE_URL.rsplit('/v1', 1)[0], stream=True, video_max_frames=10)

        if model_reply:
            print("\n\n最终捕获到的完整回复内容:")
            print(model_reply)
        else:
            print("\n函数调用失败，未能获取模型回复。")

    except Exception as e:
        print(f"执行示例时发生错误: {e}")

    # print("\n\n" + "=" * 50)
    # print("示例 2: 非流式调用")
    # print("=" * 50)

    # try:
    #     model_reply_nostream = call_multimodal_llm(
    #         text_prompt="These are two images.",
    #         media_paths=[
    #             "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/edit2509_1.jpg",
    #             "/home/jiangbaoyang/GitHub/BuildingAgent/model_api/edit2509_2.jpg"
    #         ],
    #         model_name=MODEL_TO_USE,
    #         base_url=MODEL_BASE_URL.rsplit('/v1', 1)[0],
    #         stream=False
    #     )

    #     if model_reply_nostream:
    #         print("\n模型回复:")
    #         print(model_reply_nostream)
    #     else:
    #         print("\n函数调用失败。")

    # except Exception as e:
    #     print(f"执行示例时发生错误: {e}")

