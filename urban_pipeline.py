# main.py (V3 - 全流程Orchestrator)
import json
import pprint
import os
import shutil
from random import random
from typing import Dict
from agents.planner_agent import CityPlannerAgent
from agents.asset_agent import AssetGenerationAgent
from agents.assembly_agent import SceneAssemblyAgent
from agents.base_agent import BaseAgent # 导入基类
from api_stubs import call_llm_api, call_vlm_api, gaussian_splatting_snapshot_mock

# 定义缺失的 SceneReviewAgent
class SceneReviewAgent(BaseAgent):
    """
    阶段四：场景评审 Agent
    职责：对组装完成的场景进行多维度评估，并决定是否需要迭代。
    """
    def run(self, final_scene: Dict, city_plan: Dict) -> Dict:
        print("\n" + "="*50)
        print("🤔 阶段四：启动场景评审流程")
        print("="*50)

        # 1. 生成最终报告用的多角度快照
        print("   - 📸 正在为最终报告生成多角度美学快照...")
        beauty_shot = final_scene['final_snapshot_path']
        top_down_shot = gaussian_splatting_snapshot_mock(final_scene['final_scene_ply'], "top_down", "report_top_down")
        
        # 2. VLM 分析视觉效果
        print("   - 🧐 VLM正在分析视觉效果...")
        vl_prompt = "请描述这个场景的整体氛围、光照和布局是否符合一个'白天晴天'的'20世纪中期大都市'主题。"
        visual_report_str = call_vlm_api(beauty_shot, vl_prompt)  # 注意：这里使用旧的vl_api_mock, 需要适配
        # 简单的适配
        if "evaluation_report" in visual_report_str:
             visual_report = json.loads(visual_report_str)["evaluation_report"]
        else: # 适配新的vl_api_mock的输出
             # 模拟一个基于新mock的报告
             if random() > 0.5:
                 visual_report = "场景整体光照偏暗，不符合白天晴天的设定。"
             else:
                 visual_report = "场景视觉效果优秀，符合规划。"


        # 3. LLM 综合决策
        print("   - 🧠 LLM正在综合所有信息进行最终决策...")
        decision_prompt = f"""
        你是一位项目总监。请分析以下视觉和数据报告，决定当前场景是否“满意”或需要“迭代”。

        **城市规划核心概念:**
        {json.dumps(city_plan['profile'], ensure_ascii=False, indent=2)}

        **视觉AI的观察报告:**
        {visual_report}

        **你的任务:**
        如果视觉报告指出了与规划核心概念的明显冲突（如氛围、天气），则决策为“迭代”，并在actions中提出具体的、可执行的修改建议。否则，决策为“满意”。
        严格返回JSON。
        """
        decision_str = call_llm_api(decision_prompt)
        decision = json.loads(decision_str)

        print(f"   - 最终决策: {decision['decision']}")
        if decision['decision'] == '迭代':
            print(f"   - 原因: {decision['reason']}")

        return decision


def main():
    """
    主流程 Orchestrator (V3)
    负责初始化Agents并按顺序驱动一个完整的、带迭代的PCG流程。
    """
    # 初始用户输入
    user_concept = {"theme": "西方城镇风格，写实高仿真", "scale": "3个街区", "time_of_day": "白天晴天"}
    
    # 初始化所有Agents
    planner = CityPlannerAgent()
    asset_generator = AssetGenerationAgent()
    assembler = SceneAssemblyAgent()
    reviewer = SceneReviewAgent()

    # 流程状态变量
    city_plan = None
    asset_queue = []
    asset_library = {} # 存储所有已生成的、唯一的资产实例

    max_iterations = 2
    for i in range(max_iterations):
        print("\n" + "#"*60)
        print(f"###### 开始第 {i+1}/{max_iterations} 轮迭代 ######")
        print("#"*60)

        # 阶段一：规划 (仅在第一次迭代时运行)
        if i == 0:
            city_plan, asset_queue = planner.run(user_concept)
            if not city_plan or not asset_queue:
                print("🚨 规划阶段失败，流程终止。")
                return

        # 阶段二：资产生成
        if not asset_queue:
            print("\n- 资产生成队列为空，跳过阶段二。")
        else:
            print("\n" + "="*50)
            print(f"====== 阶段二：生成 {len(asset_queue)} 类资产 ======")
            print("="*50)
            
            # 使用列表推导式安全地迭代和修改队列
            remaining_tasks = []
            for task_template in asset_queue:
                print(f"\n--- 正在处理资产类型: '{task_template['asset_id']}' (需求: {task_template['quantity_required']}) ---")
                
                # 为Agent准备一个更扁平化的任务字典
                run_task = {
                    "asset_id": task_template['asset_id'],
                    "description": task_template['description'],
                    "style": task_template['style_tags'], # 关键映射
                    "type": task_template['type']
                }

                # 根据需求数量生成资产实例
                for k in range(task_template['quantity_required']):
                    asset_instance = asset_generator.run(run_task)
                    if asset_instance:
                        # 使用唯一的实例ID存储到库中
                        instance_id = f"{task_template['asset_id']}_inst_{k+1}"
                        asset_library[instance_id] = {**asset_instance, **task_template} # 合并生成信息和规划信息
                    else:
                        print(f"   🚨 生成资产 '{task_template['asset_id']}' 的实例 {k+1} 失败，已跳过。")
            
            asset_queue.clear() # 清空本轮队列
            
            print("\n✅ 资产生成阶段完成！")
            print(f"资产库中共有 {len(asset_library)} 个资产实例。")

        # 阶段三：场景组装
        if not asset_library:
            print("🚨 资产库为空，无法进行场景组装，流程终止。")
            break
        final_scene = assembler.run(city_plan, asset_library)
        if not final_scene:
            print("🚨 场景组装失败，流程终止。")
            break
        
        # 阶段四：评审与决策
        final_decision = reviewer.run(final_scene, city_plan)
        
        if final_decision.get("decision") == "满意":
            print("\n" + "*"*60)
            print("🎉🎉🎉 最终场景通过审查！项目完成！ 🎉🎉🎉")
            print("*"*60)
            pprint.pprint(final_scene)
            break
        else:
            print("\n" + "!"*60)
            print("     场景未达标，根据反馈准备下一轮迭代...")
            print("!"*60)
            new_tasks = final_decision.get("actions", [])
            for action in new_tasks:
                if action['action_type'] == 'regenerate_asset':
                    # 从原始规划的资产目录中找到模板
                    original_task_template = next((t for t in city_plan['asset_catalogue'] if t['asset_id'] == action['asset_id']), None)
                    if original_task_template:
                        new_task = original_task_template.copy()
                        # 应用反馈并添加到队列
                        new_task['description'] += f" [迭代反馈: {action['feedback']}]"
                        new_task['quantity_required'] = 1 # 迭代通常只重新生成一个
                        asset_queue.append(new_task)
                        print(f"  -> 已将新任务 '{new_task['asset_id']}' 添加回生成队列。")
            
            if not asset_queue:
                print("\n没有需要迭代的任务，流程意外结束。")
                break
            
    else: # for循环正常结束（未被break）
        print("\n" + "X"*60)
        print("XXXXX 已达到最大迭代次数，项目终止。 XXXXX")
        print("X"*60)


if __name__ == "__main__":
    # 确保运行前清理旧的模拟文件
    if os.path.exists("tmp"):
        for f in os.listdir("tmp"):
            path_to_delete = os.path.join("tmp", f)
            # 2. 判断是文件还是目录，然后用对应的方法删除
            if os.path.isfile(path_to_delete):
                os.remove(path_to_delete)
            elif os.path.isdir(path_to_delete):
                shutil.rmtree(path_to_delete)  # 使用 shutil.rmtree() 删除目录

    main()
