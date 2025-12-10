# 1. 导入必要的库
import openai
import httpx
import sys
import datetime


def call_llm_api(
        prompt: str,
        model_name: str,
        base_url: str,
        api_key: str = "not-needed",
        temperature: float = 0.7,
        max_tokens: int = 100000,
        timeout: float = 300.0,
        stream: bool = True,
        output_filename: str | None = None
) -> str | None:
    """
    调用部署在 vLLM 上的 OpenAI 兼容 API。

    Args:
        prompt (str): 发送给模型的用户提示。
        model_name (str): 要使用的模型名称（必须与 vLLM 启动时指定的一致）。
        base_url (str): vLLM 服务的 OpenAI 兼容 API 地址。
        api_key (str, optional): API 密钥，对于本地 vLLM 通常不需要。默认为 "not-needed"。
        temperature (float, optional): 控制生成文本的随机性。默认为 0.7。
        max_tokens (int, optional): 生成响应的最大 token 数量。默认为 100000。
        timeout (float, optional): API 请求的超时时间（秒）。默认为 300.0。
        stream (bool, optional): 是否使用流式传输。默认为 True。
        output_filename (str | None, optional): 如果提供，则将模型输出保存到此文件。默认为 None。

    Returns:
        str | None: 如果成功，返回模型生成的完整字符串；如果发生错误，则返回 None。
    """
    try:
        # 初始化 OpenAI 客户端
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # 准备要发送的消息
        messages_data = [
            {"role": "user", "content": prompt}
        ]

        print(f"\n--- 正在向模型 {model_name} 发送请求 ---")
        if output_filename:
            print(f"模型输出将保存到文件: {output_filename}")

        # 根据是否流式传输执行不同逻辑
        if stream:
            # --- 流式处理 ---
            full_response_content = ""
            with open(output_filename, 'w', encoding='utf-8') if output_filename else open(sys.stdout.fileno(), 'w',
                                                                                           encoding='utf-8',
                                                                                           closefd=False) as f:
                stream_response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_data,
                    temperature=temperature,
                    timeout=httpx.Timeout(timeout),
                    max_tokens=max_tokens,
                    stream=True,
                )

                print("\n模型流式返回结果:")
                for chunk in stream_response:
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        # 实时打印到控制台
                        print(chunk_content, end="")
                        sys.stdout.flush()

                        # 如果指定了文件名，则写入文件
                        if output_filename:
                            f.write(chunk_content)

                        full_response_content += chunk_content

            print("\n--- 流式输出结束 ---")
            if output_filename:
                print(f"完整内容已成功保存到文件: {output_filename}")

            return full_response_content

        else:
            # --- 非流式处理 ---
            print("\n模型正在一次性生成结果，请稍候...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages_data,
                temperature=temperature,
                timeout=httpx.Timeout(timeout),
                max_tokens=max_tokens,
            )

            result_content = response.choices[0].message.content
            print("\n模型返回结果:")
            print(result_content)

            # 如果指定了文件名，则写入文件
            if output_filename:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(result_content)
                print(f"\n完整内容已成功保存到文件: {output_filename}")

            return result_content

    except httpx.ConnectTimeout:
        print(f"\n错误: 连接超时。请检查网络连接或 API 端点地址 '{base_url}' 是否正确。")
        return None
    except Exception as e:
        print(f"\n调用 API 或文件操作时发生错误: {e}")
        return None


# ==============================================================================
# --- 主程序：如何使用这个函数 ---
# ==============================================================================
if __name__ == "__main__":

    # --- 关键配置部分 ---
    VLLM_BASE_URL = "http://127.0.0.1:8011/v1"
    # 这个值必须与你 vLLM 启动脚本中的 --model 参数完全一致
    MODEL_NAME = "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen3-Next-80B-A3B-Thinking-FP8"

    PROMPT_TEXT = """
你是一位顶级的AI世界总设计师（Chief World Architect），你的核心任务是为一个程序化内容生成（PCG）系统创建一份无懈可击的设计蓝图。这份蓝图必须逻辑严密、细节丰富，并能直接驱动下游的资产生成和场景组装Agent。

输出必须是严格的、不包含任何额外注释的JSON格式。

### 用户宏观概念
{"theme": "西方城镇风格，写实高仿真", "scale": "3个街区", "time_of_day": "白天晴天"}

### 你的思考与生成流程（必须遵循）
1.  **主题解构**: 首先，深入分析用户概念中的核心主题。
2.  **制定宏观法则**: 基于主题，设定城市的宏观规则 (`layout_rules`)，如垂直分层和人口密度分布。
3.  **划分功能分区**: 设计城市的功能分区 (`districts`)，确保分区类型（如金融、商业、住宅、工业、市政等）能够反映一个真实城市的复杂性。
4.  **按需设计资产（核心步骤）**: 这是最重要的环节。**你必须遍历你刚刚设计的每一个`district`**，并思考：“要让这个区域看起来真实可信，我需要哪些专属的资产？” 为每个区域至少设计：
    *   3-4种**主体建筑** (`primary_building`)。
    *   若干种**环境道具**（静态`prop_static`或动态`prop_dynamic`），如街道、设施、植被、招牌。
5.  **编译总资产目录**: 将上一步为所有分区设计的资产汇总到 `asset_catalogue` 中。确保每个资产的描述足够详细，能够指导下游的2D/3D艺术家。
6.  **自我验证**: 在输出最终JSON前，检查并确认 `asset_catalogue` 中的资产是否能完全覆盖所有 `districts` 的核心功能和视觉需求。

### 输出JSON结构规范：

1.  `city_profile` (对象): 城市档案。
    *   `name` (字符串): 城市名称。
    *   `theme` (字符串): 核心主题。
    *   `description` (字符串): 富有文学性的基调描述。

2.  `layout_rules` (对象): 宏观布局规则。
    *   `verticality` (字符串): 城市垂直分层描述。
    *   `density_map` (字符串): 区域密度规则。

3.  `districts` (数组): 城市功能分区列表。
    *   `district_id` (字符串): 唯一ID, 如 "D01"。
    *   `name` (字符串): 区域名。
    *   `type` (枚举): `financial`, `commercial`, `residential`, `municipal` (市政区), `park`。
    *   `description` (字符串): 区域特点描述。
    *   `grid_allocation`: (数组) 分配给该矩形区域的网格坐标（单位米，和现实要对应）

4.  `asset_catalogue` (数组): 主资产目录。
    *   `asset_id` (字符串): 唯一ID, 如 "BUILDING_BANK_ARTDECO_01"。
    *   `type` (枚举): `building`, `vehicle`, `prop_dynamic`, `prop_static`, `vegetation` (植被)。
    *   `subtype` (字符串): 子类, 如 `bank`, `apartment`, `classic_sedan`, `street_lamp`, `oak_tree`。
    *   `style_tags` (数组): 风格标签, 如 `["Art Deco", "Realism", "Brick", "Worn"]`。
    *   `description` (字符串): **极其详细的视觉描述**，面向下游的AI艺术家。必须包含：**形态**（几何形状、轮廓）、**材质**（如砖石、钢材、玻璃）、**颜色**（主色调、辅助色）、**细节**（如窗户样式、门廊雕刻、空调外机、轻微的墙皮剥落或水渍）。
    *   `placement_rules` (对象): 放置规则。
        *   `allowed_districts` (数组): **必须准确**指定此资产所属的区域类型列表。
        *   `placement_type` (枚举): `primary_building`, `facade_prop` (如空调、雨棚), `street_level_prop` (如消防栓、长椅、柏油路), `rooftop_prop` (如水箱、天线)。
    *   `quantity_required` (整数): 需求数量。注意，只有完全一样的资产才能算作一种，计入需求数量，否则必须单独成为一种资产。

请严格按照上述流程和规范，立即开始生成这个规划方案。
"""

    # 示例1：使用流式传输并保存到带时间戳的文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/jiangbaoyang/model_api/model_response_{timestamp}.txt"

    model_response = call_llm_api(
        prompt=PROMPT_TEXT,
        model_name=MODEL_NAME,
        base_url=VLLM_BASE_URL,
        stream=True,  # 显式指定流式传输
        output_filename=output_file
    )

    if model_response:
        print("\n函数调用成功，并返回了完整内容。")
        # 你可以在这里对 model_response (这是一个完整的字符串) 进行后续处理，例如解析 JSON
    else:
        print("\n函数调用失败。")

    # # 示例2：不使用流式传输，也不保存到文件（结果直接打印到控制台）
    # print("\n\n" + "="*50)
    # print("现在演示非流式调用：")
    # model_response_no_stream = call_vllm_model(
    #     prompt="简要介绍一下什么是大型语言模型。",
    #     model_name=MODEL_NAME,
    #     base_url=VLLM_BASE_URL,
    #     stream=False, # 关闭流式传输
    #     output_filename=None # 不保存文件
    # )

    # if model_response_no_stream:
    #     print("\n非流式调用成功。")
