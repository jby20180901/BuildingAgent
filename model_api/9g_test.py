import asyncio
import time
import json

from json_repair import json_repair
from openai import AsyncOpenAI

test_str = """

你是美国海军参谋规划人员，你的任务是根据已有的**场景**、**任务**和**决策点**信息，创造性的思考，并产出格式化的作战预案
首先给出本次作战的**场景**、**任务**和**决策点**
- **战场环境:** 
### 想定场景 ###
#### 作战环境 ####
- **作战区域:** 菲律宾海，小笠原群岛及硫黄列岛周边
- **天气状况:** cloudy
- **海况等级:** 2

**关键地理区域:**
- 硫黄岛监视圈 (Iwo Jima Surveillance Circle): 类型为“Deep Sea”的区域，位于坐标 141.31667,24.78333，区域范围222000海里。
- 冲之鸟礁东部航道 (Okinotori East Transit Corridor): 类型为“Shipping Lane”的区域，位于坐标 140.00000,21.50000，区域范围111000海里。
- 反舰火力伏击区“东风” (AShM Ambush Zone "Dongfeng"): 类型为“Open Ocean”的区域，位于坐标 139.50000,26.00000，区域范围92600海里。

**参考点:**
- 火山列岛海底光缆节点 (Kazan Retto Cable Node): 经度 141.0, 纬度 25.16667
- 小笠原海沟观测点 (Ogasawara Trench Observation Point): 经度 143.5, 纬度 27.0
- ASW Baseline "Sea Serpent" (Start): 经度 140.5, 纬度 22.0
- ASW Baseline "Sea Serpent" (End): 经度 140.5, 纬度 24.0
- 紧急脱离点 (Contingency Disengagement Point "Trident"): 经度 143.0, 纬度 23.0
- 潜艇决断攻击点“海狼” (Submarine Attack Point "Sea Wolf"): 经度 140.16667, 纬度 21.75

#### 兵力部署 (Order of Battle) ####
### 【编组】 (友方 (US))
- 编组名称: Surface Combat Unit
- 编组ID: ee51e91d-3a69-419d-9ae9-c6d6b62f6c99
- 任务: 防御(defense)
- 中心位置: 经度 143.24000, 纬度 23.01000
- 下属单位:
  - 1个Ticonderoga-class cruiser:
  - 类型ID: 4e4e26d7-7c41-4ca0-bf6c-b72312c100cb:
  - 可执行行为: 机动(move) 攻击(attack) 干扰(jamming) 释放诱饵(decoy) 
  - 【实体】
    - 实体名称: USS Shiloh (CG-67)
    - 实体ID: a6a465ff-22d7-4b29-a466-b652b799d550:
    - 位置: 经度 143.26500, 纬度 23.03500
    - 主要武器系统:
    - Mk 45 5-inch/54 cal gun (射程: 24km), 数量: 2
    - Phalanx CIWS (射程: 3.6km), 数量: 2
    - MK 141 Harpoon Launcher (射程: 130km), 数量: 2
    - Mk 32 triple-tube torpedo launcher (射程: 11km), 数量: 2
    - Mk 41 Vertical Launch System (射程: 1700km), 数量: 2
    - 搭载装备:
    - SH-60 Seahawk Helicopter, 数量: 2
  - 1个Arleigh Burke-class destroyer:
  - 类型ID: 75666517-a5b7-45b7-8a2f-d96aa8f8a99b:
  - 可执行行为: 机动(move) 攻击(attack) 干扰(jamming) 释放诱饵(decoy) 
  - 【实体】
    - 实体名称: USS Milius (DDG-69)
    - 实体ID: e771d1e1-9d3b-4967-92f6-e4658d461706:
    - 位置: 经度 143.21500, 纬度 22.98500
    - 主要武器系统:
    - Mk 41 Vertical Launching System Cell (射程: 460km), 数量: 96
    - RGM-84 Harpoon (射程: 130km), 数量: 8
    - Mk 45 Mod 4 5-inch/62-caliber naval gun (射程: 24km), 数量: 1
    - Mk 15 Phalanx CIWS (射程: 2km), 数量: 2
    - Mk 32 Surface Vessel Torpedo Tubes (Mk 46/50 torpedo) (射程: 11km), 数量: 6
    - Mk 38 Mod 2 25mm Machine Gun System (射程: 3km), 数量: 2
    - Browning M2HB .50cal Machine Gun (射程: 2km), 数量: 4
    - 搭载装备:
    - SH-60 Seahawk Helicopter, 数量: 2

### 【编组】 (友方 (US))
- 编组名称: Logistics & Escort Unit
- 编组ID: a45ec5e8-6b74-41b9-91b3-fa423a2d4487
- 任务: 护航(escort)
- 中心位置: 经度 143.94000, 纬度 22.30000
- 下属单位:
  - 1个John Lewis-class replenishment oiler:
  - 类型ID: 54d7a16b-0f08-45e5-a3df-a398e1916702:
  - 可执行行为: 机动(move) 
  - 【实体】
    - 实体名称: USNS Harvey Milk (T-AO-206)
    - 实体ID: 555b8148-4732-4d31-a635-4c6953de36b7:
    - 位置: 经度 143.94000, 纬度 22.30000
    - 主要武器系统:
    - .50 caliber machine gun (射程: 2km), 数量: 4
    - Phalanx CIWS (射程: 3.6km), 数量: 0
    - RIM-116 Rolling Airframe Missile (射程: 10km), 数量: 0

### 【编组】 (友方 (US))
- 编组名称: Subsurface Combat Unit
- 编组ID: f9329b5d-5d02-48e4-bb0f-d10611c12a5d
- 任务: 反潜(anti_submarine)
- 中心位置: 经度 140.16700, 纬度 21.75000
- 下属单位:
  - 1个Los Angeles-class submarine:
  - 类型ID: 2304d42c-e03c-417c-904b-f18239fbc720:
  - 可执行行为: 机动(move) 攻击(attack) 
  - 【实体】
    - 实体名称: USS Topeka (SSN-754)
    - 实体ID: ea37c3bb-2703-4022-8dce-fca2c8527098:
    - 位置: 经度 140.16700, 纬度 21.75000
    - 主要武器系统:
    - BGM-109 Tomahawk cruise missile (射程: 2500km), 数量: 12
    - Mk 48 torpedo (射程: 50km), 数量: 37
    - UGM-84 Harpoon (射程: 280km), 数量: 0
    - Mine (射程: 0km), 数量: 0
    - 搭载装备:
    - SEAL Delivery Vehicle (SDV), 数量: 1

### 【编组】 (友方 (US))
- 编组名称: Air Superiority & Strike Unit
- 编组ID: 8d082d7d-9cc1-4ee3-ba7d-b2e070f9d1e0
- 任务: 巡逻(patrol)
- 中心位置: 经度 143.24000, 纬度 23.20000
- 下属单位:
  - 4个F/A-18 Hornet:
  - 类型ID: adc0b328-a4e5-4612-8e04-2beeb00fe74d:
  - 可执行行为: 机动(move) 侦察发射(recon_launch) 拦截(intercept) 攻击(attack) 
  - 【实体】
    - 实体名称: F/A-18 Hornet 1
    - 实体ID: e999ad51-95d5-4f60-a552-dc7221c727a2:
    - 位置: 经度 143.24000, 纬度 23.30000
    - 主要武器系统:
    - M61 Vulcan 20mm Cannon (射程: 1.5km), 数量: 1
    - AIM-120 AMRAAM (射程: 160km), 数量: 8
    - AIM-9 Sidewinder (射程: 35km), 数量: 4
    - AGM-88 HARM (射程: 150km), 数量: 2
    - AGM-84 Harpoon (射程: 220km), 数量: 4
    - AGM-84H/K SLAM-ER (射程: 270km), 数量: 2
    - AGM-158C LRASM (射程: 560km), 数量: 2
    - AGM-65 Maverick (射程: 22km), 数量: 6
    - AGM-154 JSOW (射程: 130km), 数量: 4
    - Paveway Laser-Guided Bomb (射程: 15km), 数量: 6
    - Joint Direct Attack Munition (JDAM) (射程: 28km), 数量: 6
  - 【实体】
    - 实体名称: F/A-18 Hornet 2
    - 实体ID: c482ca4f-ea51-41ad-9601-2a14d673ac8a:
    - 位置: 经度 143.34000, 纬度 23.20000
    - 主要武器系统:
    - M61 Vulcan 20mm Cannon (射程: 1.5km), 数量: 1
    - AIM-120 AMRAAM (射程: 160km), 数量: 8
    - AIM-9 Sidewinder (射程: 35km), 数量: 4
    - AGM-88 HARM (射程: 150km), 数量: 2
    - AGM-84 Harpoon (射程: 220km), 数量: 4
    - AGM-84H/K SLAM-ER (射程: 270km), 数量: 2
    - AGM-158C LRASM (射程: 560km), 数量: 2
    - AGM-65 Maverick (射程: 22km), 数量: 6
    - AGM-154 JSOW (射程: 130km), 数量: 4
    - Paveway Laser-Guided Bomb (射程: 15km), 数量: 6
    - Joint Direct Attack Munition (JDAM) (射程: 28km), 数量: 6
  - 【实体】
    - 实体名称: F/A-18 Hornet 3
    - 实体ID: 8251aa9c-5f5c-43de-8916-2a4ed2a88c60:
    - 位置: 经度 143.14000, 纬度 23.20000
    - 主要武器系统:
    - M61 Vulcan 20mm Cannon (射程: 1.5km), 数量: 1
    - AIM-120 AMRAAM (射程: 160km), 数量: 8
    - AIM-9 Sidewinder (射程: 35km), 数量: 4
    - AGM-88 HARM (射程: 150km), 数量: 2
    - AGM-84 Harpoon (射程: 220km), 数量: 4
    - AGM-84H/K SLAM-ER (射程: 270km), 数量: 2
    - AGM-158C LRASM (射程: 560km), 数量: 2
    - AGM-65 Maverick (射程: 22km), 数量: 6
    - AGM-154 JSOW (射程: 130km), 数量: 4
    - Paveway Laser-Guided Bomb (射程: 15km), 数量: 6
    - Joint Direct Attack Munition (JDAM) (射程: 28km), 数量: 6
  - 【实体】
    - 实体名称: F/A-18 Hornet 4
    - 实体ID: 8291c1df-f886-4a72-bec6-83963604443a:
    - 位置: 经度 143.24000, 纬度 23.10000
    - 主要武器系统:
    - M61 Vulcan 20mm Cannon (射程: 1.5km), 数量: 1
    - AIM-120 AMRAAM (射程: 160km), 数量: 8
    - AIM-9 Sidewinder (射程: 35km), 数量: 4
    - AGM-88 HARM (射程: 150km), 数量: 2
    - AGM-84 Harpoon (射程: 220km), 数量: 4
    - AGM-84H/K SLAM-ER (射程: 270km), 数量: 2
    - AGM-158C LRASM (射程: 560km), 数量: 2
    - AGM-65 Maverick (射程: 22km), 数量: 6
    - AGM-154 JSOW (射程: 130km), 数量: 4
    - Paveway Laser-Guided Bomb (射程: 15km), 数量: 6
    - Joint Direct Attack Munition (JDAM) (射程: 28km), 数量: 6

### 【编组】 (友方 (US))
- 编组名称: AEW&C Unit
- 编组ID: c5c6f62a-663d-4a35-bb0d-62f4192d3ec9
- 任务: 侦察(recon)
- 中心位置: 经度 144.00000, 纬度 22.50000
- 下属单位:
  - 1个E-2D Hawkeye Airborne Early Warning Aircraft:
  - 类型ID: 2125d92f-7c5a-4d06-9661-61f1b72f6e7b:
  - 可执行行为: 机动(move) 侦察(recon) 
  - 【实体】
    - 实体名称: E-2D Hawkeye 1
    - 实体ID: 27e64317-f6a4-4c6a-9d8d-c2053f88d7bd:
    - 位置: 经度 144.00000, 纬度 22.50000

### 【编组】 (友方 (US))
- 编组名称: Shiloh Helo Detachment
- 编组ID: b2b343e2-3832-486b-8a04-3bc162c4bd24
- 任务: 反潜(anti_submarine)
- 中心位置: 经度 143.26000, 纬度 23.03000
- 下属单位:
  - 2个SH-60 Seahawk Helicopter:
  - 类型ID: ff56acd1-0435-4286-99da-dbe77b5de84b:
  - 可执行行为: 机动(move) 侦察(recon) 攻击(attack) 
  - 【实体】
    - 实体名称: SH-60 Seahawk 1
    - 实体ID: fe6f3425-7c3b-4707-a324-01e914fd811c:
    - 位置: 经度 143.27000, 纬度 23.03000
    - 主要武器系统:
    - M240 Machine Gun (射程: 1.8km), 数量: 2
    - Mark 46/54 Torpedo, 数量: 2
    - AGM-114 Hellfire Missile, 数量: 8
    - AGM-119 Penguin Anti-ship Missile, 数量: 2
  - 【实体】
    - 实体名称: SH-60 Seahawk 2
    - 实体ID: f29d02e5-afbb-44bc-8e2a-df46994a6b92:
    - 位置: 经度 143.25000, 纬度 23.03000
    - 主要武器系统:
    - M240 Machine Gun (射程: 1.8km), 数量: 2
    - Mark 46/54 Torpedo, 数量: 2
    - AGM-114 Hellfire Missile, 数量: 8
    - AGM-119 Penguin Anti-ship Missile, 数量: 2

### 【编组】 (友方 (US))
- 编组名称: Milius Helo Detachment
- 编组ID: 517742c4-b7b2-45b7-8d5f-8423c9dad286
- 任务: 反潜(anti_submarine)
- 中心位置: 经度 143.22000, 纬度 22.99000
- 下属单位:
  - 2个SH-60 Seahawk Helicopter:
  - 类型ID: ff56acd1-0435-4286-99da-dbe77b5de84b:
  - 可执行行为: 机动(move) 侦察(recon) 攻击(attack) 
  - 【实体】
    - 实体名称: SH-60 Seahawk 3
    - 实体ID: bfdaeebe-69b2-4f33-98bc-06791c9cea6b:
    - 位置: 经度 143.23000, 纬度 22.99000
    - 主要武器系统:
    - M240 Machine Gun (射程: 1.8km), 数量: 2
    - Mark 46/54 Torpedo, 数量: 2
    - AGM-114 Hellfire Missile, 数量: 8
    - AGM-119 Penguin Anti-ship Missile, 数量: 2
  - 【实体】
    - 实体名称: SH-60 Seahawk 4
    - 实体ID: b3c4c99a-d08a-4829-80a3-9084b12e6f0e:
    - 位置: 经度 143.21000, 纬度 22.99000
    - 主要武器系统:
    - M240 Machine Gun (射程: 1.8km), 数量: 2
    - Mark 46/54 Torpedo, 数量: 2
    - AGM-114 Hellfire Missile, 数量: 8
    - AGM-119 Penguin Anti-ship Missile, 数量: 2

### 【编组】 (敌方 (CN))
- 编组名称: 101战斗编队
- 编组ID: 2d6f2dfa-bfb0-4cfe-ad94-4288b41a1a6f
- 任务: 巡逻(patrol)
- 中心位置: 经度 143.00000, 纬度 27.00000
- 下属单位:
  - 1个055型导弹驱逐舰:
  - 类型ID: 8a267c2e-78d8-46d2-9599-735fa98e7a1a:
  - 可执行行为: 机动(move) 攻击(attack) 干扰(jamming) 释放诱饵(decoy) 
  - 【实体】
    - 实体名称: 南昌舰(101)
    - 实体ID: a9f1ad2f-9b44-4dc1-9b25-8b3675ef1dca:
    - 位置: 经度 143.02500, 纬度 27.02500
    - 主要武器系统:
    - H/PJ-45型单管130毫米舰炮 (射程: 30km), 数量: 1
    - H/PJ-11型11管30毫米舰炮 (射程: 5km), 数量: 1
    - 红旗-10A近程防空导弹 (射程: 12km), 数量: 24
    - 红旗-9B中远程防空导弹 (射程: 200km), 数量: 56
    - 鹰击-18反舰导弹 (射程: 540km), 数量: 16
    - 鹰击-21高超音速反舰弹道导弹 (射程: 1500km), 数量: 8
    - 长剑-10陆攻巡航导弹 (射程: 1500km), 数量: 16
    - 鱼8火箭助飞鱼雷 (射程: 50km), 数量: 16
    - 搭载装备:
    - 直-20, 数量: 2
  - 1个054A型导弹护卫舰:
  - 类型ID: 235520ee-f26c-4792-9ffe-9b2f6680ce6d:
  - 可执行行为: 机动(move) 攻击(attack) 干扰(jamming) 释放诱饵(decoy) 
  - 【实体】
    - 实体名称: 舟山舰(529)
    - 实体ID: 55673d44-e48e-488c-a90a-3e1cdbbeb1b0:
    - 位置: 经度 142.97500, 纬度 26.97500
    - 主要武器系统:
    - 红旗-16中程防空导弹 (射程: 70km), 数量: 24
    - 鱼-8火箭助飞鱼雷 (射程: 50km), 数量: 8
    - 鹰击83反舰导弹 (射程: 250km), 数量: 8
    - H/PJ-26型单管76毫米舰炮 (射程: 17km), 数量: 1
    - H/PJ-11型11管30毫米舰炮 (射程: 4km), 数量: 2
    - 3200A型6管反潜火箭深弹发射装置 (射程: 5km), 数量: 2
    - 7424型324mm鱼雷发射器 (射程: 15km), 数量: 2
    - H/RJZ-726-4A型24联装干扰弹发射装置, 数量: 2
    - 搭载装备:
    - 直-9C舰载直升机, 数量: 1
    - 直-20, 数量: 1

### 【编组】 (敌方 (CN))
- 编组名称: 综合保障群
- 编组ID: a327989e-f2f5-47ec-bbe1-1ba5b1cc22cd
- 任务: 调动部署(maneuver)
- 中心位置: 经度 141.80000, 纬度 26.00000
- 下属单位:
  - 1个903型综合补给舰:
  - 类型ID: 9088e066-1ca8-44a2-9395-e61b97d5b56c:
  - 可执行行为: 机动(move) 
  - 【实体】
    - 实体名称: 太湖舰(889)
    - 实体ID: 1ad88b76-7fc3-4c0b-b6e8-07618dc1cb01:
    - 位置: 经度 141.80000, 纬度 26.00000
    - 主要武器系统:
    - H/PJ76F双管37毫米舰炮 (射程: 8.5km), 数量: 4
    - 搭载装备:
    - 直-9C舰载直升机, 数量: 1

### 【编组】 (敌方 (CN))
- 编组名称: 水下破袭分队
- 编组ID: e4e01e20-2c82-4c0e-a770-599ce9fd3b5e
- 任务: 反舰(anti_ship)
- 中心位置: 经度 140.50000, 纬度 21.50000
- 下属单位:
  - 1个039A型潜艇:
  - 类型ID: 22fba84b-85fe-45a6-ba73-7f82d5ece449:
  - 可执行行为: 机动(move) 攻击(attack) 
  - 【实体】
    - 实体名称: 330号艇(330)
    - 实体ID: 9c08a2bb-0aee-4f8f-8b5d-548cf5870846:
    - 位置: 经度 140.50000, 纬度 21.50000
    - 主要武器系统:
    - 533毫米鱼雷发射管 (射程: 540km), 数量: 6

### 【编组】 (敌方 (CN))
- 编组名称: 岸基航空兵特遣队
- 编组ID: a8b4654f-23d1-48b2-88d7-647c628ae27f
- 任务: 攻击(attack)
- 中心位置: 经度 139.00000, 纬度 26.50000
- 下属单位:
  - 4个歼-16战斗机:
  - 类型ID: 51dc9316-e9c6-436b-8d63-f9ec3ecf160a:
  - 可执行行为: 机动(move) 侦察发射(recon_launch) 拦截(intercept) 攻击(attack) 
  - 【实体】
    - 实体名称: 歼-16战斗机 1
    - 实体ID: 7c3495e0-12b2-4d07-b806-b2183d7751ce:
    - 位置: 经度 139.10000, 纬度 26.60000
    - 主要武器系统:
    - GSh-30-1型30毫米机炮 (射程: 2km), 数量: 1
    - 霹雳-8/9短程空对空导弹, 数量: 4
    - 霹雳-12/15中远程空对空导弹, 数量: 8
    - 霹雳-17远程空对空导弹, 数量: 4
    - 鹰击-83/91反舰/反辐射导弹, 数量: 4
    - 鹰击-62反舰巡航导弹, 数量: 2
    - 雷电-10反辐射导弹, 数量: 4
    - 雷霆/雷石制导/滑翔炸弹, 数量: 8
    - 常规航空炸弹, 数量: 8
  - 【实体】
    - 实体名称: 歼-16战斗机 2
    - 实体ID: bfe05d45-c3cb-4931-afc1-465d2e453d19:
    - 位置: 经度 138.90000, 纬度 26.60000
    - 主要武器系统:
    - GSh-30-1型30毫米机炮 (射程: 2km), 数量: 1
    - 霹雳-8/9短程空对空导弹, 数量: 4
    - 霹雳-12/15中远程空对空导弹, 数量: 8
    - 霹雳-17远程空对空导弹, 数量: 4
    - 鹰击-83/91反舰/反辐射导弹, 数量: 4
    - 鹰击-62反舰巡航导弹, 数量: 2
    - 雷电-10反辐射导弹, 数量: 4
    - 雷霆/雷石制导/滑翔炸弹, 数量: 8
    - 常规航空炸弹, 数量: 8
  - 【实体】
    - 实体名称: 歼-16战斗机 3
    - 实体ID: 3cb2edaa-97cd-4cf4-8e5a-139535a2a75b:
    - 位置: 经度 139.00000, 纬度 26.40000
    - 主要武器系统:
    - GSh-30-1型30毫米机炮 (射程: 2km), 数量: 1
    - 霹雳-8/9短程空对空导弹, 数量: 4
    - 霹雳-12/15中远程空对空导弹, 数量: 8
    - 霹雳-17远程空对空导弹, 数量: 4
    - 鹰击-83/91反舰/反辐射导弹, 数量: 4
    - 鹰击-62反舰巡航导弹, 数量: 2
    - 雷电-10反辐射导弹, 数量: 4
    - 雷霆/雷石制导/滑翔炸弹, 数量: 8
    - 常规航空炸弹, 数量: 8
  - 【实体】
    - 实体名称: 歼-16战斗机 4
    - 实体ID: 407802b0-1593-43f8-827b-3fbf8e0c6956:
    - 位置: 经度 139.00000, 纬度 26.50000
    - 主要武器系统:
    - GSh-30-1型30毫米机炮 (射程: 2km), 数量: 1
    - 霹雳-8/9短程空对空导弹, 数量: 4
    - 霹雳-12/15中远程空对空导弹, 数量: 8
    - 霹雳-17远程空对空导弹, 数量: 4
    - 鹰击-83/91反舰/反辐射导弹, 数量: 4
    - 鹰击-62反舰巡航导弹, 数量: 2
    - 雷电-10反辐射导弹, 数量: 4
    - 雷霆/雷石制导/滑翔炸弹, 数量: 8
    - 常规航空炸弹, 数量: 8

### 【编组】 (敌方 (CN))
- 编组名称: 南昌舰舰载机分队
- 编组ID: 791610af-70d7-41cc-a72f-16e294575582
- 任务: 反潜(anti_submarine)
- 中心位置: 经度 143.02000, 纬度 27.02000
- 下属单位:
  - 2个直-20:
  - 类型ID: 30582ca4-e648-4348-9a6f-a6f6efb549a2:
  - 可执行行为: 机动(move) 侦察(recon) 攻击(attack) 
  - 【实体】
    - 实体名称: 直-20 1
    - 实体ID: f985e63a-4e5e-45ca-a90e-9bf5a8e046a8:
    - 位置: 经度 143.03000, 纬度 27.02000
    - 主要武器系统:
    - AKD-10反坦克导弹 (射程: 15km), 数量: 8
    - PL-10E空空导弹 (射程: 20km), 数量: 4
    - YJ-9反舰导弹 (射程: 20km), 数量: 4
    - Yu-11轻型反潜鱼雷 (射程: 9km), 数量: 2
    - 90毫米火箭巢 (射程: 5km), 数量: 14
    - 12.7毫米舱门机枪 (射程: 1.8km), 数量: 2
  - 【实体】
    - 实体名称: 直-20 2
    - 实体ID: bfdfa8d3-60ee-4783-bb68-448d21f7cfd4:
    - 位置: 经度 143.01000, 纬度 27.02000
    - 主要武器系统:
    - AKD-10反坦克导弹 (射程: 15km), 数量: 8
    - PL-10E空空导弹 (射程: 20km), 数量: 4
    - YJ-9反舰导弹 (射程: 20km), 数量: 4
    - Yu-11轻型反潜鱼雷 (射程: 9km), 数量: 2
    - 90毫米火箭巢 (射程: 5km), 数量: 14
    - 12.7毫米舱门机枪 (射程: 1.8km), 数量: 2

### 【编组】 (敌方 (CN))
- 编组名称: 舟山舰舰载机分队
- 编组ID: cb3f9993-2413-46d0-8bbe-71d59bdacfdc
- 任务: 侦察(recon)
- 中心位置: 经度 142.98000, 纬度 26.98000
- 下属单位:
  - 1个直-9C舰载直升机:
  - 类型ID: 6d7e6436-f363-4b5d-98f8-b5b1e30260e1:
  - 可执行行为: 机动(move) 侦察(recon) 
  - 【实体】
    - 实体名称: 直-9C舰载直升机 1
    - 实体ID: 991b82a5-55f3-42d3-830c-ba2ba1c4c852:
    - 位置: 经度 142.98000, 纬度 26.98000
    - 主要武器系统:
    - 鱼-7型鱼雷 (射程: 15km), 数量: 2
    - C-701型反舰导弹 (射程: 20km), 数量: 2



- **初始任务:** 
### 当前任务 ###
**总任务描述:** “警惕海沟”行动是一次旨在维护国际法规定航行自由权、慑止对手单方面改变现状的军事行动。特混任务群70.1将通过展示强大的海空控制、精确的反潜作战和坚固的多层防御能力，确保关键后勤补给线畅通，同时向中方清晰传达美方捍卫其盟友及自身在西太平洋地区利益的坚定决心，最终以最小化摩擦的方式达成战略目标。

**核心目标:**
- 为确保航行自由，安全护送USNS Harvey Milk号补给舰通过“冲之鸟礁东部航道”。
- 在“硫黄岛监视圈”内建立局部海空优势，慑止中方的军事冒险行动，并展示美方在该关键水域的决心。

**关键任务:**
- **[Recon]** 命令USS Topeka号潜艇与舰载SH-60海鹰直升机协同，沿“海蛇”反潜基线执行搜索任务，务必在想定前半段定位并持续跟踪中方039A型潜艇，为后续模拟攻击和威慑行动提供情报支持。
- **[Defense]** 以USS Shiloh号巡洋舰为核心，利用E-2D预警机提供超视距预警，构建“守护者”导弹交战区（MEZ）。随时准备启动协同交战能力（CEC），优先拦截来自歼-16平台的高超音速反舰导弹，确保主力舰艇安全。
- **[Support]** 派遣F/A-18战斗机为水面作战单元提供持续的战斗空中巡逻（CAP），特别是在USNS Harvey Milk号进入航道期间，需前出拦截并驱离任何有威胁意图的中方战机，确保护航对象的绝对安全。



- **预设关键决策点:** 
### 决策点: 决策点 1：水下威胁初现 (ID: DP-US-001)
- 描述: 我方洛杉矶级核潜艇USS Topeka在“冲之鸟礁东部航道”北口预设阵位侦测到疑似中方039A型潜艇的声学信号。这是对我方高价值单位USNS Harvey Milk的直接潜在威胁，也是达成“持续跟踪敌潜艇”胜利条件的关键机会。指挥官需决定如何调配SH-60反潜直升机进行确认、建立跟踪，并评估此举是否会过早暴露我方反潜意图和兵力部署。
  触发条件 (满足以下所有条件):
  - 发现敌方单位“039A型潜艇 (330号艇(330))”，距离50。


### 数量要求 ###
要求最终得到三个具有多样性的预案

**你的任务：**
## 任务一 ##
首先你要作为一个专业的职业军官，遵循海军规划流程，简洁而深刻地进行分析，并将重心放在制定创新的作战预案上。
按以下框架进行深入思考

**1. 任务分析摘要**
你要按照如下的MDMP流程进行深入思考：
*   **作战与规划分析**
  *   **指挥官意图与最终状态:** 提炼任务的核心目标及期望达成的战役终局。
  *   **关键任务:** 列出为实现指挥官意图所必须完成的“规定任务”和“隐含任务”，并突出“核心任务“。
  *   **我方关键约束与风险:** 指出我方行动的主要限制、后勤和通信上的核心风险。
*   **情报分析**
  *   **作战环境关键因素:** 简述环境（水文、气象、电磁）对双方行动最关键的影响。
  *   **敌方能力与预案概要:**
    *   **关键能力:** 总结敌方最值得关注的侦察、火力、潜艇及网络战能力。
    *   **敌方最可能行动:** 几句话言简意赅、突出重点地描述。
    *   **敌方最危险行动:** 几句话言简意赅、突出重点地描述。
*   **后勤可行性分析**
  *   **评估后勤需求
  *   **评估海上补给线并识别后勤风险。
*   **通信与网络分析**
  *   **评估通信环境
  *   **我方指挥通信与网络侦察能力评估并识别通信风险。

**2. 作战预案分析**
依据以上分析，围绕“分布式海上作战”理念，制定一个或多个具有本质区别、相互平行、互不统属、互不干扰的作战预案，作战预案的数量严格满足###数量要求###。每个预案需满足“可行、可接受、适当、有区别、完整”的原则。
**为每个预案提供以下详细思考：**
*   **A. 核心作战构想:** 逻辑清晰、环环相扣的详细分析该预案的灵魂和主要战法。
*   **B. 战术推演:** 逻辑清晰、环环相扣、因果相连地分析该预案如何生成的，要求承接**A. 核心作战构想**，引起下文的**C. 本次预案的时空需求**、**D. 编组与部署**、**E. 兵力资源需求**、**F. 关键信息**、**G. 后续规划**，并理顺叙述CDEFG各项之间的因果关系，并且体现合理的反复斟酌过程，最终形成一个逻辑合理的分析段落（不少于200字）。
*   **C. 本次预案的时空需求:** 
  *   本次预案涉及的关键区域。
  *   本次预案的起止时间。
*   **D. 编组与部署:**
  *   识别想定场景中不同的作战编组。
  *   确定本次预案需要出动的作战编组。
  *   为每个出动的编组赋予清晰的且与其他预案不同的**主要指令** (从 "attack", "defense", "maneuver", "anti_submarine", "air_defense", "anti_ship", "anti_missile", "jamming", "intercept", "evade", "patrol", "escort", "search", "retreat", "recon_launch", "recon" 中选择)。
  *   根据任务和决策点的背景要求，为每个出动的编组简要赋予一个小的作战目的。
*   **E. 兵力资源需求:**
  *   识别想定场景中不同的作战编组中各下属单位的种类和数量。
  *   明确每个编组需要出动哪些种类的单位及需要的数量。
*   **F. 关键信息:**
  *   确定指挥官的一些关键信息需求。
  *   简述控制措施，描述不能进行的行动和不能进入的区域等。
*   **G. 后续规划:**
  *   预测敌方对本预案最可能的反应。
  *   简述我方应对该反应的下一步行动。

**3. 预案差异性验证**
*   **验证各个预案具有核心差异**
*   **每个预案均满足"可行、可接受、适当、有区别、完整"原则**

**4. 预案格式验证**
*   **检查点1**：area字段必须为关键地理区域的名称，一字不差
*   **检查点2**：time字段格式必须为"YYYY-MM-DD HH:mm:ss"，且时间在2028年之后，时间跨度合理
*   **检查点3**：allocation中group_id必须为UUID且与想定一致
*   **检查点4**：command必须从编组级指令中选择，且每个编组仅分配一个command。
*   **检查点5**：allocation中group_id和group_name不得重复
*   **检查点6**：units数组中的group_id必须与allocation一致
*   **检查点7**：description中编组装备选取必须与想定一致
*   **检查点8**：purpose需简洁描述本次行动目的，并符合战术逻辑。
*   **检查点9**：CCIR、control_measures描述必须具体，并符合战术逻辑
*   **检查点10**：reaction与counteraction需逻辑连贯，并符合双方战术逻辑
*   **检查点11**：remarks需逻辑连贯，并体现本预案的战术思想
*   **最终验证**：
  *   预案的数量符合**用户要求**
  *   各个预案的JSON结构完全符合输出格式要求
  *   每个预案的area、time、action、units、CCIR、control_measures、reaction、counteraction、remarks字段均存在
  *   无多余字段，无缺失字段
  *   各个预案在核心构想、战术细节、资源分配上具有显著差异性


## 任务二 ##
然后你需要根据上述成熟的思考过程，将生成的一个或多个口头作战预案进行格式化输出：
其中### 【编组级指令】及对应的含义 ###
{
  "attack",
  "defense", 
  "maneuver",
  "anti_submarine",
  "air_defense", 
  "anti_ship",
  "anti_missile",
  "jamming",
  "intercept",
  "evade",
  "patrol",
  "escort",
  "search",
  "retreat",
  "recon_launch",
  "recon"
}

### 输出格式 ###
请严格遵循以下规则，输出一个JSON对象：
1.  整个输出内容必须是一个单一的、合法的JSON对象。
2.  该JSON对象有且仅有一个键，其名称为 "results"。
3.  "results" 键对应的值必须是一个数组。
4.  这个数组必须不多不少，包含的元素数量正好和###数量要求###保持一致。
5.  数组中的每一个元素本身也必须是一个对象，代表一个独立的行动预案。
6.  多样性要求：每个独立的行动预案，必须从不同角度切入战场，互相之间有显著性的区别
【格式示例】
{
  "results": [
    {
      //行动预案
    },
    ...//行动预案的数量与###数量要求###保持一致
  ]
}
2. "results"数组必须包含一个或多个完整的行动预案对象,行动元对象的个数必须与###数量要求###保持一致,每个行动预案必须包含以下字段且格式如下：
{
  "area": [
    <关键区域的名称>
    ...//一个或者多个关键区域，来自于**关键地理区域**的名称
  ],
  "time": 
  {
    "start": "YYYY-MM-DD HH:mm:ss"<开始时间>,
    "end": "YYYY-MM-DD HH:mm:ss"<结束时间>
  },
  "action":
  {
    "description":  "一、作战推演：给出逻辑严密、层层相扣的作战推演分析，要求必须体现本预案如何形成“二、编组部署”的逻辑链条。（不少于200字）二、编组部署：1、【此处填充**想定场景**中编组的名称】选取【此处填充选取数量】个【此处填充**想定场景**中该编组任意下属单位中的装备类型的名称】、【此处填充选取数量】个【此处填充**想定场景**中该编组任意下属单位中的装备类型的名称】、...，组成编队以执行【本编组的指令】行动，以便于实现【填充作战目的】目的;...", //至少三条预案指令构成的段落，编组可以选取多种装备，但是每个编组只能在一条预案指令中出现出现，每个编组内部选取装备种类不能重复。
    "purpose": <本次行动目的>,
    "allocation": [//注意选取的【编组】的任意下属单位中的装备必须在description出现。已有的【编组】可以没有分配任务，每个【编组】最多只能出现一次！
      {
      "group_id": <我方编组ID>, //必须从**想定场景**选取相应的【编组】，写入对应的 编组ID，注意必须是UUID格式
      "group_name": <我方编组名称>,// 填写选取的【编组】内的 编组名称，必须与group_id（编组ID）匹配
      "command": <编组行动指令>,//只能从 【编组级指令】找到对应的英文指令填入，要求符合作战逻辑，一个【编组】只能分配唯一一个【编组级指令】
      "orders_list":[],//必须为空列表
      },
      ...//group_id和group_name 严禁重复出现
    ]
  },
  "units": [//allocation中出现的编组ID集合
    <友方编组id的字符串>,
    ...//一个或者多个友方编组id
  ],
  "CCIR": <指挥官关键信息需求字符串>,
  "control_measures": <控制措施，描述不能进行的行动和不能进入的区域>,
  "reaction": <敌人有可能的反应行动>,
  "counteraction": <针对敌人有可能的反应行动，我方可以实施的反制行动>,
  "remarks": <额外备注>,
}
按要求深入思考后，严格按照`输出格式`给出多样性方案的json结构输出。
    """


async def main():
    client1 = AsyncOpenAI(
        base_url="http://127.0.0.1:8058/v1",
        api_key="fm9g"
    )
    client2 = AsyncOpenAI(
        base_url="http://127.0.0.1:8059/v1",
        api_key="fm9g"
    )
    response = await client1.chat.completions.create(
        model="9g",
        messages=[
            {"role": "user", "content": "请你进行100字自我介绍"}
        ],
        stream=False,  # 设置为 False，直接获取完整响应
        temperature=0.6,
        max_tokens=32768,
        extra_body=dict(add_special_tokens=True)
    )
    print(response.choices[0].message.content)
    response = await client2.chat.completions.create(
        model="9g",
        messages=[
            {"role": "user", "content": "请你进行100字自我介绍"}
        ],
        stream=False,  # 设置为 False，直接获取完整响应
        temperature=0.6,
        max_tokens=32768,
    )
    print(response.choices[0].message.content)
    time.sleep(1)
    time1 = time.time()
    response = await client2.chat.completions.create(
        model="9g",
        messages=[
            {"role": "user", "content": test_str}
        ],
        stream=False,  # 设置为 False，直接获取完整响应
        temperature=0.6,
        max_tokens=32768,
    )
    time2 = time.time()
    # 直接访问响应内容
    print(response.choices[0].message.content)
    
    response = await client1.chat.completions.create(
        model="9g",
        messages=[
            {"role": "user", "content": "请你进行100字自我介绍"}
        ],
        stream=False,  # 设置为 False，直接获取完整响应
        temperature=0.6,
        max_tokens=32768,
        extra_body=dict(add_special_tokens=True)
    )
    print(response.choices[0].message.content)
    response = await client2.chat.completions.create(
        model="9g",
        messages=[
            {"role": "user", "content": "请你进行100字自我介绍"}
        ],
        stream=False,  # 设置为 False，直接获取完整响应
        temperature=0.6,
        max_tokens=32768,
    )
    print(response.choices[0].message.content)
    time.sleep(1)
    time3 = time.time()
    response = await client1.chat.completions.create(
        model="9g",
        messages=[
            {"role": "user", "content": test_str}
        ],
        stream=False,  # 设置为 False，直接获取完整响应
        temperature=0.6,
        max_tokens=32768,
        extra_body=dict(add_special_tokens=True)
    )
    time4 = time.time()
    # 直接访问响应内容
    print(response.choices[0].message.content)
    print(f"普通 : {time2 - time1}")
    print(f"投机 : {time4 - time3}")

asyncio.run(main())

