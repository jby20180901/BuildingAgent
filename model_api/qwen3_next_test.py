# 1. 导入必要的库
import openai
import httpx
import sys

# --- 关键配置部分 ---

# 2. 初始化 OpenAI 客户端
# vLLM 启动的 OpenAI 兼容 API 的地址
# 如果在同一台机器上，使用 localhost。如果在不同机器，请替换为服务器的IP地址。
# 注意：标准的 OpenAI 兼容接口路径是 /v1
base_url = "http://127.0.0.1:8011/v1" 

# 对于本地部署的 vLLM 服务，API 密钥不是必须的，但 openai 库要求提供一个值。
# 所以我们可以填写任何非空字符串，例如 "not-needed" 或 "dummy"。
api_key = "not-needed"

test_str = """
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

# 创建客户端实例，这对应你代码中的 self.client_dp
# 我们将其命名为 client
client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# 3. 准备调用参数
# 这个值必须与你 vLLM 启动脚本中的 --model 参数完全一致
model_name = "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/Qwen3-Next-80B-A3B-Thinking-FP8"

# 准备要发送的消息，格式为 OpenAI 的 message 列表
# 这对应你代码中的 data
messages_data = [
    {"role": "user", "content": test_str}
]

# --- 执行 API 调用 ---

print(f"正在向模型 {model_name} 发送请求...")

# try:
#     # 4. 使用你的代码逻辑进行调用
#     response = client.chat.completions.create(
#         model=model_name,  # 使用我们定义的 model_name
#         messages=messages_data,  # 使用我们准备的 messages_data
#         temperature=0.7,
#         timeout=httpx.Timeout(300.0), 
#         max_tokens=200000,
#     )
    
#     # 5. 提取并打印结果
#     # 这对应你代码中的 return response.choices[0].message.content
#     result_content = response.choices[0].message.content
#     print("\n模型返回结果:")
#     print(result_content)

# except Exception as e:
#     print(f"调用 API 时发生错误: {e}")

import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"/home/jiangbaoyang/model_api/model_response_{timestamp}.txt"
print(f"模型输出将被实时保存到文件: {output_filename}")


try:
    # 新增 2. 使用 'with' 语句打开文件
    # 'w' 表示写入模式，如果文件已存在则会覆盖
    # encoding='utf-8' 对于处理包含各种语言和符号的AI生成内容至关重要
    with open(output_filename, 'w', encoding='utf-8') as f:
        # 在 API 调用中添加 stream=True
        stream_response = client.chat.completions.create(
            model=model_name,
            messages=messages_data,
            temperature=0.7,
            timeout=httpx.Timeout(300.0), 
            max_tokens=100000, # 注意：这个值可能过大，建议根据模型支持调整
            stream=True,
        )
        
        print("\n模型流式返回结果:")
        
        full_response_content = ""
        # 遍历流式响应
        for chunk in stream_response:
            if chunk.choices[0].delta.content is not None:
                chunk_content = chunk.choices[0].delta.content
                
                # a. 实时打印到控制台 (原有逻辑)
                print(chunk_content, end="")
                sys.stdout.flush()
                
                # 新增 3. 将块内容实时写入文件
                f.write(chunk_content)
                # 可选：如果你希望在外部实时查看文件内容（如用 tail 命令），可以刷新文件缓冲区
                # f.flush() 

                # b. 拼接到完整字符串中 (原有逻辑)
                full_response_content += chunk_content
    
    # 'with' 语句块结束时，文件 f 会被自动关闭

    print() 
    print("\n--- 流式输出结束 ---")
    print(f"完整内容已成功保存到文件: {output_filename}")
    
    # 可以在这里继续使用 full_response_content 变量
    # print("\n最终收集到的完整内容:")
    # print(full_response_content)

except httpx.ConnectTimeout:
    print(f"错误: 连接超时。请检查网络连接或 API 端点地址。")
except Exception as e:
    print(f"调用 API 或文件操作时发生错误: {e}")

