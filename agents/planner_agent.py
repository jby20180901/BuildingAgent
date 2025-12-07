# agents/planner_agent.py
import json
from typing import Dict, List, Any, Tuple

from .base_agent import BaseAgent
from api_stubs import qwen_api_mock

# agents/planner_agent.py (升级版)
import json
from typing import Dict, List, Any, Tuple

from .base_agent import BaseAgent
from api_stubs import qwen_api_mock

class CityPlannerAgent(BaseAgent):
    """
    阶段一：规划与设计 Agent (专家版)
    职责：接收用户概念，生成一个包含世界观、布局规则、区域规划和详细资产目录的深度城市规划方案。
    """
    def run(self, user_concept: Dict[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        print("\n" + "="*50)
        print("====== 阶段一：规划与设计 (专家模式) ======")
        print("="*50)
        
        # prompt = f"""..."""  (替换为以下内容)
        prompt = f"""
你是一位顶级的AI世界总设计师（Chief World Architect），你的核心任务是为一个程序化内容生成（PCG）系统创建一份无懈可击的设计蓝图。这份蓝图必须逻辑严密、细节丰富，并能直接驱动下游的资产生成和场景组装Agent。

输出必须是严格的、不包含任何额外注释的JSON格式。

### 用户宏观概念
{json.dumps(user_concept, ensure_ascii=False)}

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
    *   `name` (字符串): 区域名, 如 "联邦大道金融区"。
    *   `type` (枚举): `financial`, `commercial`, `residential`, `municipal` (市政区), `park`。
    *   `description` (字符串): 区域特点描述。
    *   `grid_allocation`: (数组) 分配给该区域的简单网格坐标

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

        
        response_str = qwen_api_mock(prompt)
        plan_data = json.loads(response_str)
        
        # 根据新的、更丰富的结构来解析数据
        city_plan = {
            "profile": plan_data.get("city_profile", {}),
            "rules": plan_data.get("layout_rules", {}),
            "districts": plan_data.get("districts", [])
        }
        asset_queue = plan_data.get("asset_catalogue", [])
        
        print("\n✅ 深度城市规划生成完毕！")
        print(f"城市名称: {city_plan['profile'].get('name', 'N/A')}")
        print(f"待生成资产目录中的项目数量: {len(asset_queue)}")
        
        return city_plan, asset_queue

