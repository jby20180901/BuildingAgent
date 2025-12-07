# agents/asset_agent.py
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from api_stubs import qwen_api_mock, qwen_image_api_mock, sam3d_api_mock, qwen_vl_api_mock

class AssetGenerationAgent(BaseAgent):
    """
    阶段二：资产原子化生成 Agent
    职责：处理单个资产生成任务，包括Prompt生成、2D图像生成、质量校验和3D模型生成。
    """
    def run(self, asset_task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print("\n" + "-"*40)
        print(f"====== 阶段二：开始处理资产任务: {asset_task['type']} ({asset_task['style']}) ======")
        
        # 1. Prompt生成 (Qwen)
        prompt_for_image = f"""
        请为以下资产生成一个高质量的文生图prompt，目标是生成一张适合3D化的、风格明确的2D图像。
        资产类型: {asset_task['type']}
        风格: {asset_task['style']}
        描述: {asset_task['description']}
        """
        image_prompt = qwen_api_mock(prompt_for_image)
        
        # 2. 2D图像生成 (Qwen-Image)
        image_path = qwen_image_api_mock(image_prompt)
        
        # 3. 质量校验 (Qwen-VL)
        qa_prompt = f"这张图片的风格是{asset_task['style']}吗？它是否符合'{asset_task['type']}'的描述？"
        qa_result = qwen_vl_api_mock(image_path, qa_prompt)
        
        if qa_result["evaluation"] == "不合格":
            print(f"❌ 资产 '{asset_task['type']}' 质量校验失败: {qa_result['reason']}。任务终止。")
            return None
        
        # 4. 3D模型生成 (SAM3D)
        model_path = sam3d_api_mock(image_path)
        
        # 5. 打包资产
        generated_asset = {
            "asset_id": asset_task['asset_id'],
            "type": asset_task['type'],
            "style": asset_task['style'],
            "source_image": image_path,
            "model_3d_path": model_path
        }
        print(f"✅ 资产 '{asset_task['type']}' 生成成功！")
        return generated_asset
