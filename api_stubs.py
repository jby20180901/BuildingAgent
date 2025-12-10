# api_stubs.py (V3 - 全面升级以支持所有高级Agent)

import json
import os
import time
import random
import zipfile
from typing import Dict, Any, Optional, Union

# 确保tmp目录存在
if not os.path.exists("tmp"):
    os.makedirs("tmp")

# --- 核心大模型模拟 ---

def call_llm_api(prompt: str, image_path: Optional[str] = None) -> str:
    """
    【已升级】模拟Qwen文本或多模态模型API。
    根据高度结构化的Prompt返回相应的JSON或文本。
    """
    if image_path:
        print(f"🧠 VLM (Qwen) received prompt: '{prompt[:100]}...' AND image '{image_path}'")
    else:
        print(f"🧠 LLM (Qwen) received prompt: '{prompt[:100]}...'")
    time.sleep(0.5)
    
    # 模拟 阶段一：城市规划
    if "顶级的AI世界总设计师" in prompt:
        return get_planner_response()
        
    # 模拟 阶段三：多模态场景布局
    elif "虚拟城市布局师" in prompt:
        x = round(random.uniform(-50, 50), 2)
        z = round(random.uniform(-50, 50), 2)
        y_rot = round(random.uniform(0, 360), 2)
        y = 0.0 if "building" in prompt.lower() else 0.5
        response = {
            "position": {"x": x, "y": y, "z": z},
            "rotation": {"x": 0.0, "y": y_rot, "z": 0.0}
        }
        return json.dumps(response, indent=2)

    # 模拟 阶段二：估算尺寸
    elif "估算其真实世界尺寸" in prompt:
        if "building" in prompt.lower() or "大楼" in prompt:
            return "Length: 30m, Width: 20m, Height: 60m"
        elif "vehicle" in prompt.lower() or "车" in prompt:
            return "Length: 4.5m, Width: 1.8m, Height: 1.5m"
        else:
            return "Length: 1m, Width: 1m, Height: 2m"

    # 模拟 阶段二：文生图Prompt优化 (内部步骤)
    elif "命令：生成资产概念图" in prompt:
        return prompt # 直接返回，模拟一个简单的优化或透传

    # 模拟 阶段四：场景评审决策
    elif "分析以下视觉和数据报告" in prompt:
        if "整体光照偏暗" in prompt: # 模拟评审发现问题
            decision = {
                "decision": "迭代",
                "reason": "场景的整体氛围与规划中的'白天晴天'不符，光线太暗。",
                "actions": [
                    {
                        "action_type": "regenerate_asset", 
                        "asset_id": "BUILDING_BANK_ARTDECO_01", 
                        "feedback": "外墙材质颜色需要更明亮，减少表面的风化和污渍效果。"
                    },
                    {
                        "action_type": "adjust_assembly",
                        "feedback": "重新运行时，请尝试将全局光照强度提高20%。"
                    }
                ]
            }
        else: # 模拟评审通过
            decision = {
                "decision": "满意",
                "reason": "场景布局合理，资产细节丰富，整体视觉效果符合规划要求。",
                "actions": []
            }
        return json.dumps(decision, ensure_ascii=False, indent=2)
    
    return json.dumps({"error": "未知的prompt类型"})


def call_vlm_api(media_path: Union[str, Dict], prompt: str) -> str:
    """
    【已升级】模拟Qwen-VL多模态模型。
    根据不同的QA任务返回结构化的JSON响应。
    """
    time.sleep(1)

    # 模拟 阶段三：场景放置差分对比评估
    if "差分对比" in prompt:
        any_image_path = list(media_path.values())[0] if isinstance(media_path, dict) else ""
        print(f"👀 Qwen-VL (Differential QA) on {len(media_path)} images for asset: '{any_image_path.split('_')[2]}'")
        if "retry_1" in any_image_path:
            response = {"pass": False, "reason": "对比局部图发现，资产明显悬浮于地面之上。"}
        else:
            response = {"pass": True, "reason": "资产已稳固放置，与周围环境融合良好。"}
    
    # 模拟 阶段二：3D模型视频QA
    elif ".mp4" in str(media_path):
        print(f"👀 Qwen-VL (3D Video QA) on '{media_path}'")
        if "attempt_1" in str(media_path):
            response = {"pass": False, "reason": "模型存在明显的悬浮碎片和破面。"}
        else:
            response = {"pass": True, "reason": "模型完整，几何准确，渲染质量达标。"}

    # 模拟 阶段二：2D图像QA
    else:
        print(f"👀 Qwen-VL (2D Image QA) on '{media_path}'")
        if "attempt_1" in str(media_path):
            response = {"pass": False, "failed_criteria": [4], "reason": "图像存在明显的投射阴影，不符合3D建模要求。"}
        else:
            response = {"pass": True, "failed_criteria": [], "reason": "所有标准均已满足。"}
            
    return json.dumps(response, indent=2, ensure_ascii=False)


# --- 核心生成模型模拟 ---

def call_gen_image_api(prompt: str, attempt: int) -> str:
    """模拟文生图API。文件名中包含尝试次数，以便QA mock进行响应。"""
    print(f"🎨 Qwen-Image (Attempt {attempt}) processing prompt...")
    time.sleep(2)
    asset_name = os.path.join("tmp", f"gen_img_attempt_{attempt}_{random.randint(1000, 9999)}.png")
    with open(asset_name, 'w') as f: f.write('fake png data')
    print(f"  -> Generated: {asset_name}")
    return asset_name

def call_gen_3d_api(image_path: str, attempt: int) -> str:
    """模拟图生3D模型API。返回一个包含多个文件的zip包。"""
    print(f"🧊 SAM3D (Attempt {attempt}) processing image: '{image_path}'")
    time.sleep(3)
    base_name = os.path.basename(image_path).replace('.png', f'_3d_model_attempt_{attempt}')
    zip_path = os.path.join("tmp", f"{base_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # 创建假的内部文件
        ply_path = "model.ply"
        with open(ply_path, 'w') as f: f.write('fake ply data')
        zipf.write(ply_path)
        os.remove(ply_path)

        video_path = "render.mp4"
        with open(video_path, 'w') as f: f.write('fake mp4 data')
        zipf.write(video_path)
        os.remove(video_path)
        
    print(f"  -> Generated 3D package: {zip_path}")
    return zip_path


# --- 场景与高斯泼溅模拟 ---

def gaussian_splatting_merge_mock(base_scene_ply: Optional[str], new_asset_ply: str, position: Dict, rotation: Dict, step: int) -> str:
    """模拟合并高斯模型。"""
    print(f"   - [API STUB] Merging asset into scene...")
    time.sleep(1)
    merged_path = os.path.join("tmp", f"scene_merged_step_{step}.ply")
    with open(merged_path, "w") as f: f.write(f"Fake merged PLY data, step {step}")
    return merged_path

def gaussian_splatting_snapshot_mock(scene_ply: Optional[str], camera_mode: str, info: str, target_pos: Optional[Dict] = None) -> str:
    """【已升级】模拟为高斯场景生成快照。"""
    scene_name = "empty_scene" if scene_ply is None else os.path.basename(scene_ply)
    print(f"   - [API STUB] Taking '{camera_mode}' snapshot of '{scene_name}' for '{info}'...")
    time.sleep(0.5)
    snapshot_path = os.path.join("tmp", f"snapshot_{info}.png")
    with open(snapshot_path, 'w') as f: f.write(f"Fake {camera_mode} snapshot data")
    return snapshot_path

def get_planner_response():
    """返回一个符合CityPlannerAgent V3规范的、丰富的城市规划模拟数据。"""
    plan = {
        "city_profile": {
            "name": "翡翠城 (Emerald City)",
            "theme": "装饰艺术(Art Deco)与现实主义风格融合的20世纪中期大都市",
            "description": "一座在战后经济繁荣时期崛起的城市，天际线被雄伟的砖石与黄铜建筑所占据。街道上，经典汽车与步履匆匆的市民交织，空气中弥漫着乐观与一丝不易察 amarelo的紧张。"
        },
        "layout_rules": {
            "verticality": "城市中心区域是高楼的森林，向外围逐渐过渡到中低层建筑。",
            "density_map": "金融区和商业区密度最高，住宅区次之，公园区域最低。"
        },
        "districts": [
            {
                "district_id": "D01",
                "name": "中央金融区",
                "type": "financial",
                "description": "城市的经济心脏，布满了银行总部、证券交易所和摩天办公楼。",
                "grid_allocation": [[-500, -500], [0, 0]]
            },
            {
                "district_id": "D02",
                "name": "第五大道商业街",
                "type": "commercial",
                "description": "高端商店、剧院和餐厅的聚集地，夜晚霓虹闪烁。",
                "grid_allocation": [[0, -500], [500, 0]]
            },
            {
                "district_id": "D03",
                "name": "西区公寓",
                "type": "residential",
                "description": "中产阶级的居住区，以砖砌公寓楼为主，街道较为安静。",
                "grid_allocation": [[-500, 0], [0, 500]]
            }
        ],
        "asset_catalogue": [
            {
                "asset_id": "BUILDING_BANK_ARTDECO_01",
                "type": "building",
                "subtype": "bank",
                "style_tags": ["Art Deco", "Realism", "Brick", "Limestone"],
                "description": "一栋雄伟的装饰艺术风格银行大楼。主体为浅色石灰岩，基座和窗框采用深色砖石。入口处有巨大的黄铜雕花大门，窗户为垂直的长条形，楼顶有阶梯状的退台和旗杆。表面有轻微的风化水渍。",
                "placement_rules": {
                    "allowed_districts": ["financial"],
                    "placement_type": "primary_building"
                },
                "quantity_required": 1
            },
            {
                "asset_id": "PROP_STREETLAMP_CLASSIC_01",
                "type": "prop_static",
                "subtype": "street_lamp",
                "style_tags": ["Vintage", "Iron"],
                "description": "一盏经典的铸铁单头路灯，灯柱上有涡卷花纹装饰，灯罩是乳白色的玻璃球形。灯柱为黑色，有轻微的锈迹。",
                "placement_rules": {
                    "allowed_districts": ["financial", "commercial", "residential"],
                    "placement_type": "street_level_prop"
                },
                "quantity_required": 10
            },
            {
                "asset_id": "VEHICLE_SEDAN_1950S_RED_01",
                "type": "vehicle",
                "subtype": "classic_sedan",
                "style_tags": ["1950s", "Realism", "Chrome"],
                "description": "一辆1950年代风格的红色四门轿车。车身曲线圆润，拥有大量的镀铬装饰条、巨大的圆形前灯和尾鳍设计。车漆光亮但有细微划痕。",
                "placement_rules": {
                    "allowed_districts": ["financial", "commercial", "residential"],
                    "placement_type": "street_level_prop"
                },
                "quantity_required": 3
            }
        ]
    }
    return json.dumps(plan, ensure_ascii=False, indent=2)

