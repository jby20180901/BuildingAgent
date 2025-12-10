# agents/assembly_agent.py
import json
import time
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
# 注意：qwen_api_mock 现在也被用于多模态输入
from api_stubs import call_llm_api, gaussian_splatting_merge_mock, gaussian_splatting_snapshot_mock, call_vlm_api

class SceneAssemblyAgent(BaseAgent):
    """
    阶段三：场景程序化组装 Agent (V2 - 视觉增强版)
    职责：通过迭代、视觉验证和多模态决策的循环，智能地将资产逐一放置到场景中。
    """

    def run(self, city_plan: Dict, asset_library: Dict[str, Dict], max_placement_retries: int = 5) -> Optional[Dict[str, Any]]:
        """
        执行详细的、基于视觉反馈的场景组装流程。
        """
        print("\n" + "="*50)
        print("💡 阶段三：启动视觉增强型场景组装流程")
        print("="*50)

        scene_state = {
            "merged_ply_path": None,
            "placed_assets": []
        }

        asset_ids_sorted = sorted(asset_library.keys(), key=lambda x: "BUILDING" not in x)

        for i, asset_id in enumerate(asset_ids_sorted):
            asset_info = asset_library[asset_id]
            print(f"\n--- 正在处理资产 {i+1}/{len(asset_ids_sorted)}: '{asset_id}' ---")

            placement_success, updated_scene_state = self._place_and_verify_asset_multimodal(
                asset_id, asset_info, scene_state, city_plan, max_placement_retries
            )

            if placement_success:
                scene_state = updated_scene_state
                print(f"   ✅ 资产 '{asset_id}' 已成功放置并合并到场景中。")
            else:
                print(f"   🚨 警告：资产 '{asset_id}' 在 {max_placement_retries} 次尝试后仍无法成功放置，已跳过。")
        
        print("\n--- 🚀 所有资产处理完毕，生成最终场景快照 ---")
        if scene_state["merged_ply_path"]:
            final_snapshot = gaussian_splatting_snapshot_mock(scene_state["merged_ply_path"], "panoramic", "final_beauty_shot")
            print(f"🎉 场景组装完成！最终快照: {final_snapshot}")
            return {
                "final_scene_ply": scene_state["merged_ply_path"],
                "final_snapshot_path": final_snapshot,
                "placed_assets_info": scene_state["placed_assets"]
            }
        else:
            print("❌ 场景中没有任何资产被成功放置，组装失败。")
            return None

    def _place_and_verify_asset_multimodal(self, asset_id: str, asset_info: Dict, current_scene_state: Dict, city_plan: Dict, max_retries: int) -> (bool, Dict):
        """
        单个资产的放置、合并、验证循环（多模态增强版）。
        """
        # 1. 拍摄放置前的全景图，为布局决策提供视觉上下文
        print("   - 📸 正在拍摄当前场景全景图 (用于布局决策)...")
        panoramic_before_path = gaussian_splatting_snapshot_mock(
            current_scene_state["merged_ply_path"], "panoramic", f"before_{asset_id}"
        )

        for attempt in range(1, max_retries + 1):
            print(f"\n   [尝试 {attempt}/{max_retries}] for '{asset_id}':")
            
            # 2. 调用多模态模型决定放置位置（VLM nyní přijímá obraz)
            print("   - 🧠 请求VLM规划放置坐标 (附带场景视觉)...")
            placement_prompt = self._create_multimodal_placement_prompt(asset_id, asset_info, current_scene_state, city_plan)
            # 假设qwen_api_mock可以处理多模态输入
            placement_str = call_llm_api(placement_prompt, image_path=panoramic_before_path)
            
            try:
                placement_data = json.loads(placement_str)
                target_pos = placement_data['position']
            except (json.JSONDecodeError, KeyError):
                print("     ❌ 布局模型返回了无效的JSON或数据格式不正确。正在重试...")
                continue

            # 3. 拍摄放置前的“局部”快照
            print(f"   - 📸 正在拍摄目标区域 {target_pos} 的局部快照 (放置前)...")
            local_before_path = gaussian_splatting_snapshot_mock(
                current_scene_state["merged_ply_path"], "local", f"before_{asset_id}_local_retry_{attempt}", target_pos
            )

            # 4. 调用模拟API合并高斯模型
            print(f"   - 🔗 正在合并模型到场景中... (at {target_pos})")
            newly_merged_ply = gaussian_splatting_merge_mock(
                base_scene_ply=current_scene_state["merged_ply_path"],
                new_asset_ply=asset_info["gaussian_splatting_path"],
                position=target_pos,
                rotation=placement_data["rotation"],
                step=len(current_scene_state["placed_assets"]) + 1
            )

            # 5. 拍摄放置后的“局部”和“全景”快照
            print(f"   - 📸 正在拍摄目标区域 {target_pos} 的局部快照 (放置后)...")
            local_after_path = gaussian_splatting_snapshot_mock(
                newly_merged_ply, "local", f"after_{asset_id}_local_retry_{attempt}", target_pos
            )
            print("   - 📸 正在拍摄新场景的全景快照 (放置后)...")
            panoramic_after_path = gaussian_splatting_snapshot_mock(
                newly_merged_ply, "panoramic", f"after_{asset_id}_pano_retry_{attempt}"
            )
            
            # 将所有视觉证据打包
            visual_evidence = {
                "panoramic_before": panoramic_before_path,
                "local_before": local_before_path,
                "panoramic_after": panoramic_after_path,
                "local_after": local_after_path,
            }

            # 6. 调用VLM评估放置质量（使用四张对比图）
            print("   - 🧐 请求VLM进行差分对比，评估放置质量...")
            qa_prompt = self._create_differential_qa_prompt(asset_id, asset_info, placement_data)
            qa_result_str = call_vlm_api(visual_evidence, qa_prompt)

            try:
                qa_result = json.loads(qa_result_str)
                if qa_result.get("pass") is True:
                    updated_state = current_scene_state.copy()
                    updated_state["merged_ply_path"] = newly_merged_ply
                    updated_state["placed_assets"].append({"asset_id": asset_id, **placement_data})
                    return True, updated_state
                else:
                    print(f"     ❌ 放置质量校验失败: {qa_result.get('reason', '未知原因')}")
            except json.JSONDecodeError:
                print("     ❌ VLM评估返回了无效的JSON。")

            if attempt < max_retries:
                print("      即将重试放置...")
                time.sleep(1)

        return False, current_scene_state

    def _create_multimodal_placement_prompt(self, asset_id: str, asset_info: Dict, scene_state: Dict, city_plan: Dict) -> str:
        """【已升级】为多模态模型创建用于决定资产位置的Prompt。"""
        return f"""
你是一名专业的虚拟城市布局师。请仔细观察提供的**场景全景图**，并结合以下信息，为新资产决定一个最佳放置位置。

**场景规划:**
{json.dumps(city_plan, indent=2, ensure_ascii=False)}

**已放置的资产列表 (用于逻辑参考):**
{json.dumps(scene_state['placed_assets'], indent=2, ensure_ascii=False)}

**当前待放置的资产:**
- ID: {asset_id}
- 类型: {asset_info['type']}
- 描述: {asset_info.get('description', 'N/A')}
- 估算尺寸: {asset_info['estimated_dimensions']}

**你的任务:**
1.  **观察图像**: 分析图像中的空闲区域、道路位置和现有建筑布局。
2.  **结合规划**: 根据场景规划，将资产放置在合适的区域（如，车辆在道路上，建筑在住宅区）。
3.  **避免碰撞**: 在图像中寻找一个足够大的空地，确保新资产不会与已有物体发生视觉上的重叠。

**输出格式:**
请严格按照以下JSON格式返回，不要包含任何额外说明：
{{
  "position": {{ "x": float, "y": float, "z": float }},
  "rotation": {{ "x": 0.0, "y": float, "z": 0.0 }}
}}
"""

    def _create_differential_qa_prompt(self, asset_id: str, asset_info: Dict, placement_data: Dict) -> str:
        """【已升级】为VLM创建基于四张对比图进行质量评估的Prompt。"""
        return f"""
你是一个精密的场景搭建质量保证（QA）机器人。你收到了四张截图，分别是资产放置前后的全景图和局部图。请通过对比这些图像，评估本次操作的质量。

**操作信息:**
- 放置的资产ID: `{asset_id}`
- 类型: `{asset_info['type']}`
- 目标坐标: `{placement_data['position']}`

**评估任务 (对比分析):**
1.  **对比 `local_before` 和 `local_after`**:
    - 物理合理性: 新资产是否悬浮在空中？是否不自然地嵌入了地面或其他物体？
    - 碰撞与穿模: 是否有明显的模型交叉或穿透现象？
2.  **对比 `panoramic_before` 和 `panoramic_after`**:
    - 逻辑合理性: 从宏观上看，这个新资产的摆放位置是否符合城市规划的逻辑？（例如，汽车在路上，建筑在规划的街区内）
    - 整体和谐度: 新加入的资产是否破坏了场景的整体美感或布局？

**输出格式:**
请严格按照以下JSON格式返回，不要包含任何额外说明：
{{
  "pass": (布尔值),
  "reason": (字符串, 基于你的对比分析，简要说明评估结论，特别是失败原因)
}}
"""
