# main.py
from agents import CityPlannerAgent, AssetGenerationAgent, SceneAssemblyAgent, SceneReviewAgent

def main():
    """
    主流程 Orchestrator
    负责初始化Agents并按顺序驱动整个Pipeline。
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
    asset_library = {}

    max_iterations = 2
    for i in range(max_iterations):
        print("\n" + "#"*60)
        print(f"###### 开始第 {i+1} 轮迭代 ######")
        print("#"*60)

        # 阶段一：只在第一次迭代时运行
        if i == 0:
            city_plan, asset_queue = planner.run(user_concept)

        # 阶段二：处理当前队列中的所有任务
        if asset_queue:
            print("\n" + "="*50)
            print("====== 阶段二：并行生成所有资产 ======")
            print("="*50)
            completed_tasks = []
            for task in asset_queue:
                # 模拟并行处理
                for _ in range(task['quantity']):
                    asset = asset_generator.run(task)
                    if asset:
                        # 使用唯一的id存储资产
                        asset_unique_id = f"{asset['asset_id']}_{len(asset_library) + 1}"
                        asset_library[asset_unique_id] = asset
                completed_tasks.append(task)
            
            # 从队列中移除已完成的任务
            asset_queue = [t for t in asset_queue if t not in completed_tasks]
            
            print("\nInventory: 资产生成阶段完成！")
            print(f"资产库中共有 {len(asset_library)} 个资产。")

        # 阶段三
        scene_snapshot = assembler.run(city_plan, asset_library)
        
        # 阶段四
        final_decision = reviewer.run(scene_snapshot, city_plan)
        
        if final_decision["decision"] == "满意":
            print("\n" + "*"*60)
            print("🎉🎉🎉 最终场景通过审查！项目完成！ 🎉🎉🎉")
            print("*"*60)
            break
        else:
            print("\n" + "!"*60)
            print("Iteration Required: 场景未达标，根据反馈准备下一轮迭代...")
            new_tasks = final_decision.get("actions", [])
            for action in new_tasks:
                if action['action_type'] == 'regenerate_asset':
                    original_task_template = next((t for t in city_plan['asset_requirements'] if t['asset_id'] == action['asset_id']), None)
                    if original_task_template:
                        new_task = original_task_template.copy()
                        new_task['description'] += f" [迭代反馈: {action['feedback']}]"
                        new_task['quantity'] = 1
                        asset_queue.append(new_task)
                        print(f"已将新任务 '{new_task['type']}' 添加回生成队列。")
            
            if not asset_queue:
                print("\n没有需要迭代的任务，流程结束。")
                break
            
            if i == max_iterations - 1:
                print("\n达到最大迭代次数，项目终止。")

if __name__ == "__main__":
    main()
