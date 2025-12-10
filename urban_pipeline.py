# main.py (V3 - 全流程Orchestrator)
import os
import shutil
from agents.planner_agent import CityPlannerAgent
from agents.asset_agent import AssetGenerationAgent
from agents.assembly_agent import SceneAssemblyAgent

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

    # 流程状态变量
    city_plan = None
    asset_queue = []
    asset_library = {} # 存储所有已生成的、唯一的资产实例

    max_iterations = 2

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
    final_scene = assembler.run(city_plan, asset_library)
    if not final_scene:
        print("🚨 场景组装失败，流程终止。")

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
