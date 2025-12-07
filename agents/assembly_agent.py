# agents/assembly_agent.py
import json
import time
from typing import Dict

from .base_agent import BaseAgent
from api_stubs import qwen_api_mock

class SceneAssemblyAgent(BaseAgent):
    """
    阶段三：场景程序化组装 Agent
    职责：根据城市规划和资产库，生成组装逻辑并模拟场景的实例化。
    """
    def run(self, city_plan: Dict, asset_library: Dict) -> str:
        print("\n" + "="*50)
        print("====== 阶段三：场景程序化组装 ======")
        print("="*50)

        prompt = f"""
        你是一位场景组装工程师。请根据以下城市规划和已有的资产库，生成一个Python伪代码用于场景组装。

        城市规划:
        {json.dumps(city_plan, ensure_ascii=False)}

        可用资产列表:
        {json.dumps(list(asset_library.keys()), ensure_ascii=False)}
        """
        
        assembly_script = qwen_api_mock(prompt)
        print("\n📜 生成的组装脚本:")
        print(assembly_script)
        
        print("\n⚙️ 正在执行场景组装...")
        time.sleep(2) # 模拟渲染时间
        scene_snapshot = "scene_snapshot_v1.jpg"
        print(f"✅ 场景组装完成，已生成快照: {scene_snapshot}")
        
        return scene_snapshot
