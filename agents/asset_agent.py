# agents/asset_agent.py

import json
import os
import time
import zipfile
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from api_stubs import qwen_api_mock, qwen_image_api_mock, sam3d_api_mock, qwen_vl_api_mock

class AssetGenerationAgent(BaseAgent):
    """
    阶段二：资产原子化生成 Agent
    职责：通过一个清晰、分阶段且带有多重重试校验的流程，处理单个资产的完整生成任务。
    """

    # --- 主流程 Orchestrator ---

    def run(self, asset_task: Dict[str, Any], max_2d_retries: int = 3, max_3d_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        执行完整的资产生成流程，从2D概念到最终的3D资产包。
        每个质量校验环节都包含独立的重试机制。
        """
        print("\n" + "="*50)
        print(f"🚀 阶段二：启动资产生成流程 for '{asset_task['asset_id']}'")
        print("="*50)

        # 阶段 2.1: 生成并验证合格的2D图像 (带重试)
        verified_image_path = self._generate_and_verify_2d_image(asset_task, max_2d_retries)
        if not verified_image_path:
            print(f"🚨 流程终止：未能生成合格的2D图像。")
            return None

        # 阶段 2.2: 生成并验证3D模型 (带重试)
        model_files = self._generate_and_verify_3d_model(asset_task, verified_image_path, max_3d_retries)
        if not model_files:
            print(f"🚨 流程终止：生成的3D模型未能通过质量评估。")
            return None

        # 阶段 2.3: 估算物理尺寸 (通常无需重试，除非API可能失败)
        estimated_dimensions = self._estimate_dimensions(asset_task)
        
        # 阶段 2.4: 打包最终资产
        final_asset = self._package_final_asset(asset_task, verified_image_path, model_files, estimated_dimensions)
        
        print(f"\n🎉 资产 '{asset_task['asset_id']}' 已成功生成并打包！")
        return final_asset

    # --- 私有辅助方法 (Private Helper Methods) ---

    def _generate_and_verify_2d_image(self, asset_task: Dict[str, Any], max_retries: int) -> Optional[str]:
        """负责2D图像的生成和质量校验，包含重试逻辑。"""
        print("\n--- 📝 Phase 2.1: 生成并校验2D概念图 (带重试) ---")
        image_prompt_template = self._create_2d_image_prompt_template(asset_task)
        image_prompt = qwen_api_mock(image_prompt_template) # 生成最终prompt
        
        for attempt in range(1, max_retries + 1):
            print(f"\n   Attempt {attempt}/{max_retries} for 2D Image:")
            
            image_path = qwen_image_api_mock(image_prompt, attempt)
            
            qa_prompt = self._create_2d_qa_prompt(asset_task)
            qa_result_str = qwen_vl_api_mock(image_path, qa_prompt)
            
            try:
                qa_result = json.loads(qa_result_str)
                if qa_result.get("pass") is True:
                    print(f"   ✅ 2D图像质量校验通过！ -> {image_path}")
                    return image_path
                else:
                    print(f"   ❌ 2D图像质量校验失败: {qa_result.get('reason', '未知原因')}")
            except json.JSONDecodeError:
                print("   ❌ QA模型返回了无效的JSON格式。")

            if attempt < max_retries:
                print("      即将重试...")
                time.sleep(1)

        print(f"\n   🚨 在 {max_retries} 次尝试后，仍无法通过2D质量校验。")
        return None

    def _generate_and_verify_3d_model(self, asset_task: Dict[str, Any], image_path: str, max_retries: int) -> Optional[Dict[str, str]]:
        """负责3D模型的生成、解包和基于视频的质量评估，包含完整的重试逻辑。"""
        print("\n--- 📦 Phase 2.2: 生成并校验3D模型 (带重试) ---")
        
        for attempt in range(1, max_retries + 1):
            print(f"\n   Attempt {attempt}/{max_retries} for 3D Model:")
            
            model_zip_path = sam3d_api_mock(image_path, attempt) 
            
            print("   📁 正在解包3D资产...")
            unpacked_files = self._unpack_zip_mock(model_zip_path)
            render_video_path = unpacked_files["render_video"]
            
            qa_prompt = self._create_3d_qa_prompt(asset_task)
            qa_result_str = qwen_vl_api_mock(render_video_path, qa_prompt)
            
            try:
                qa_result = json.loads(qa_result_str)
                if qa_result.get("pass"):
                    print("   ✅ 3D模型视频评估通过！")
                    return {"model_zip_path": model_zip_path, **unpacked_files}
                else:
                    print(f"   ❌ 3D模型质量校验失败: {qa_result.get('reason', '未知原因')}")
            except json.JSONDecodeError:
                print("   ❌ 3D QA模型返回了无效的JSON格式。")

            if attempt < max_retries:
                print("      即将重试...")
                time.sleep(1)

        print(f"\n   🚨 在 {max_retries} 次尝试后，仍无法通过3D质量校验。")
        return None

    def _estimate_dimensions(self, asset_task: Dict[str, Any]) -> str:
        """调用LLM估算资产在真实世界中的物理尺寸。"""
        print("\n--- 📏 Phase 2.3: 估算物理尺寸 ---")
        dimension_prompt = f"""
你是一个经验丰富的场景设计师。根据以下描述估算其真实世界尺寸。
描述: "{asset_task['description']}"
类型: "{asset_task['type']}"
请以 "Length: Xm, Width: Ym, Height: Zm" 的格式给出合理估算。
"""
        estimation = qwen_api_mock(dimension_prompt)
        print(f"   -> 估算结果: {estimation}")
        return estimation

    def _package_final_asset(self, asset_task: Dict[str, Any], image_path: str, model_files: Dict, dimensions: str) -> Dict:
        """将所有生成的信息和路径整合到一个最终的资产字典中。"""
        print("\n--- 🎁 Phase 2.4: 打包最终资产 ---")
        final_package = {
            "asset_id": asset_task['asset_id'],
            "type": asset_task['type'],
            "style": asset_task['style'],
            "source_image_path": image_path,
            "model_3d_zip_path": model_files["model_zip_path"],
            "gaussian_splatting_path": model_files["model_file"],
            "render_video_path": model_files["render_video"],
            "estimated_dimensions": dimensions,
            "status": "Success"
        }
        print(f"   -> 打包完成: {json.dumps(final_package, indent=2, ensure_ascii=False)}")
        return final_package

    # --- Prompt模板和工具函数 ---
    
    def _create_2d_image_prompt_template(self, asset_task: Dict[str, Any]) -> str:
        """【已中文化】创建用于生成2D概念图的Prompt模板。"""
        style_str = ", ".join(asset_task['style'])
        return f"""
# 命令：生成资产概念图
## 核心主体: {asset_task['description']}
## 艺术风格: {style_str}，用于3D建模的、产品级质量的概念艺术图 (production-quality concept art for 3D modeling)。
## 构图与视角: 等轴测视角 (Isometric view) 或 3/4视角 (three-quarter view)，置于纯白色或浅灰色背景上 (plain white background)。
## 光照: 无阴影的全局光照 (shadowless global illumination)，柔和的影棚灯光 (soft studio lighting)，无任何投射阴影 (no cast shadows)。
## 质量: 杰作 (masterpiece), 最佳画质 (best quality), 4K, 超高细节 (ultra detailed), 线条清晰 (clean lineart)。
## 负面提示: --no blurry, shadows, complex background, atmospheric perspective, lens flare
"""

    def _create_2d_qa_prompt(self, asset_task: Dict[str, Any]) -> str:
        """【已中文化】创建用于2D图像质量校验的Prompt。"""
        style_str = ", ".join(asset_task['style'])
        return f"""
你是一个为程序化内容生成（PCG）流水线服务的自动化QA机器人。请根据以下标准评估图像。
你的回答必须是格式化的JSON对象，不含任何其它说明性文本。

### 评估标准清单
1. **风格一致性**: 图像的艺术风格是否符合以下关键词：`{style_str}`？
2. **内容准确性**: 图像是否准确描绘了核心主体：`{asset_task['description']}`？
3. **3D适用性 - 视角**: 图像是否为清晰的等轴测或3/4视角，能清楚展示物体结构，无严重遮挡？
4. **3D适用性 - 光照**: 光照是否均匀、全局，且没有明显的投射阴影？

### 输出格式
请严格按照以下格式输出JSON：
{{
  "pass": (布尔值),
  "failed_criteria": (一个包含未通过标准编号的数组，例如 [3, 4]),
  "reason": (字符串, 简要说明失败原因)
}}
"""

    def _create_3d_qa_prompt(self, asset_task: Dict[str, Any]) -> str:
        """【已中文化】创建用于3D模型视频质量校验的Prompt。"""
        return f"""
你是一位资深的3D艺术质量总监。请仔细观看这段360度模型渲染视频，并评估其质量。
你的回答必须是格式化的JSON对象，不含任何其它说明性文本。

### 评估标准清单
1. **模型完整性**: 模型是否存在非常明显的破洞、缺失的面或悬浮的零碎几何体？
2. **几何准确性**: 模型的整体形状和结构是否与源图像的主体（`{asset_task['description']}`）保持高度一致？
3. **渲染质量**: 视频中是否存在非常严重的渲染瑕疵、闪烁或伪影？

### 输出格式
请严格按照以下格式输出JSON：
{{
  "pass": (布尔值),
  "reason": (字符串, 简要说明评估结论)
}}
"""

    def _unpack_zip_mock(self, zip_path: str) -> Dict[str, str]:
        """辅助函数，解压ZIP文件并返回已知内部文件的路径。"""
        extract_dir = os.path.join("tmp", "unpacked_" + os.path.basename(zip_path).replace('.zip', ''))
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        return {
            "model_file": os.path.join(extract_dir, "model.ply"),
            "render_video": os.path.join(extract_dir, "render.mp4")
        }
