# agents/review_agent.py
import json
from typing import Dict

from .base_agent import BaseAgent
from api_stubs import call_llm_api, call_vlm_api

class SceneReviewAgent(BaseAgent):
    """
    阶段四：评估与迭代 Agent
    职责：审查场景快照，进行多模态分析，并决策是否需要迭代。
    """
    def run(self, scene_snapshot: str, city_plan: Dict) -> Dict:
        print("\n" + "="*50)
        print("====== 阶段四：评估与迭代 ======")
        print("="*50)
        
        # 1. 场景审查 (Qwen-VL)
        review_prompt = f"描述这个场景的整体氛围，并根据'{city_plan['theme']}'的主题判断其一致性和潜在问题。"
        review_result = call_vlm_api(scene_snapshot, review_prompt)
        
        # 2. 整合决策 (Qwen)
        decision_prompt = f"""
        作为项目总监，请分析以下视觉评估报告，并决定下一步行动。
        如果报告是正面的，决定为'满意'。
        如果报告指出了问题，决定为'迭代'，并给出具体的行动指令（'actions'）。

        视觉评估报告:
        {review_result['evaluation_report']}
        """
        
        decision_str = call_llm_api(decision_prompt)
        decision = json.loads(decision_str)
        
        print(f"\n🎬 总监决策: {decision['decision']}")
        print(f"理由: {decision['reason']}")
        return decision
