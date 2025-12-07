# api_stubs.py
import json
import time
import random

# --- 模拟后端API ---

def qwen_api_mock(prompt: str) -> str:
    """模拟Qwen大语言模型的API。根据prompt的内容返回不同的结构化文本。"""
    print(f"\n🤖 Qwen接收到Prompt:\n---\n{prompt[:150]}...\n---")
    time.sleep(1) # 模拟网络延迟和处理时间

    # 1. 模拟城市规划
    if "生成一个详细的城市规划" in prompt:
        plan = {
            "city_name": "东方明珠-2077",
            "theme": "装饰艺术与赛博朋克的融合",
            "layout_grid": [
                {"block_id": "A1", "type": "金融核心区", "height_limit": 500},
                {"block_id": "A2", "type": "商业娱乐区", "height_limit": 300},
                {"block_id": "B1", "type": "高档住宅区", "height_limit": 200}
            ],
            "asset_requirements": [
                {"asset_id": "asset_001", "type": "摩天大楼", "style": "装饰艺术", "quantity": 3, "description": "作为城市地标，需要有华丽的金属和玻璃外墙。"},
                {"asset_id": "asset_002", "type": "全息广告牌", "style": "赛博朋克", "quantity": 10, "description": "动态、霓虹灯效果，内容随机。"},
                {"asset_id": "asset_003", "type": "悬浮车", "style": "未来主义", "quantity": 5, "description": "流线型设计，带有发光线条。"}
            ]
        }
        return json.dumps(plan, ensure_ascii=False, indent=2)

    # 2. 模拟为文生图生成Prompt
    if "请为以下资产生成一个高质量的文生图prompt" in prompt:
        if "摩天大楼" in prompt:
            return "一栋高达50层的装饰艺术风格摩天大楼的正面视图，正交投影，平坦中性的光照，石灰岩和青铜材质，复杂的几何雕刻，超现实，4K，细节丰富 --style raw"
        if "全息广告牌" in prompt:
            return "一个巨大的赛博朋克风格全息广告牌，显示着动态的日文和霓虹图案，悬浮在建筑侧面，夜晚，雨天的潮湿街道反射，数字艺术，电影感 --ar 16:9"

    # 3. 模拟生成场景组装逻辑
    if "生成一个Python伪代码用于场景组装" in prompt:
        return """
# 场景组装伪代码
import scene_engine as se

# 从资产库加载模型
skyscrapers = asset_library.get_by_type("摩天大楼")
ads = asset_library.get_by_type("全息广告牌")

# 根据规划布局
se.place_object(skyscrapers[0], position="A1", rotation=90)
se.place_object(skyscrapers[1], position="A2", rotation=0)
se.attach_object(ads[0], to=skyscrapers[0], at="facade_center")
print("场景已根据伪代码组装完成。")
"""
    # 4. 模拟分析评估报告并决策
    if "分析以下视觉评估报告" in prompt:
        if "摩天大楼灯光过亮" in prompt:
            decision = {
                "decision": "迭代",
                "reason": "场景氛围不符，需要调整资产。",
                "actions": [
                    {"action_type": "regenerate_asset", "asset_id": "asset_001", "feedback": "降低窗户和外部灯光的亮度，增加更多阴影。"}
                ]
            }
        else:
            decision = {
                "decision": "满意",
                "reason": "场景整体风格统一，布局合理，符合初始概念。",
                "actions": []
            }
        return json.dumps(decision, ensure_ascii=False, indent=2)
    
    return "未知类型的prompt，无法处理。"

def qwen_image_api_mock(prompt: str) -> str:
    """模拟Qwen-Image文生图模型的API。"""
    print(f"\n🎨 Qwen-Image接收到Prompt: '{prompt}'")
    time.sleep(2) # 模拟生成时间
    asset_name = "generated_image_" + str(random.randint(1000, 9999)) + ".png"
    print(f"✅ 成功生成图片: {asset_name}")
    return asset_name

def sam3d_api_mock(image_path: str) -> str:
    """模拟SAM3D图生3D模型的API。"""
    print(f"\n🧊 SAM3D正在处理图片: '{image_path}'")
    time.sleep(3) # 模拟3D转换时间
    model_name = image_path.replace('.png', '.gs')
    print(f"✅ 成功生成3D高斯泼溅模型: {model_name}")
    return model_name

def qwen_vl_api_mock(image_path: str, prompt: str) -> dict:
    """模拟Qwen-VL多模态模型的API。"""
    print(f"\n👀 Qwen-VL正在审查: '{image_path}'，问题: '{prompt}'")
    time.sleep(1)

    # 模拟资产质量校验
    if "这张图片的风格是装饰艺术吗" in prompt:
        # 80%的概率合格
        if random.random() < 0.8:
            response = {"evaluation": "合格", "reason": "风格符合装饰艺术，结构清晰，适合3D化。"}
            print("✅ Qwen-VL校验结果: 合格")
        else:
            response = {"evaluation": "不合格", "reason": "细节过于现代，缺少装饰艺术的经典几何元素。"}
            print("❌ Qwen-VL校验结果: 不合格")
        return response

    # 模拟场景评估
    if "描述这个场景的整体氛围" in prompt:
        # 50%的概率发现问题
        if random.random() < 0.5:
            response = {"evaluation_report": "场景整体布局不错，但A1区的摩天大楼灯光过亮，破坏了赛博朋克的阴暗氛围。商业区的广告牌密度可以再高一些。"}
            print("⚠️ Qwen-VL场景审查发现问题。")
        else:
            response = {"evaluation_report": "场景氛围渲染得很好，建筑风格统一，光影效果出色，符合'装饰艺术赛博朋克'的主题。"}
            print("👍 Qwen-VL场景审查通过！")
        return response
    
    return {"evaluation": "错误", "reason": "无法理解的问题。"}
