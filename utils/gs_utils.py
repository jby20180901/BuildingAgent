import os
import time
import numpy as np
from typing import Optional, Dict
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
from plyfile import PlyData
from gsplat.rendering import rasterization
from torchvision.utils import save_image
import math
import os
import time
from typing import Optional, Dict, Union

def gaussian_splatting_merge(
        base_scene_ply: Optional[str],
        new_asset_ply: str,
        position: Dict[str, float],
        rotation: Dict[str, float],
        scale: Optional[Dict[str, float]] = None,
        step: int = 0,
        output_dir: str = "tmp"
) -> str:
    """
    将新的高斯模型资产合并到基础场景中。

    Args:
        base_scene_ply: 基础场景的PLY文件路径。如果为None，则只对新资产进行变换。
        new_asset_ply: 要添加的新资产PLY文件路径。
        position: 位置字典，格式: {'x': 0.0, 'y': 0.0, 'z': 0.0}
        rotation: 旋转字典（欧拉角，单位：度），格式: {'x': 0.0, 'y': 0.0, 'z': 0.0}
        scale: 缩放字典，格式: {'x': 1.0, 'y': 1.0, 'z': 1.0}。默认为None（无缩放）
        step: 步骤编号，用于生成输出文件名。
        output_dir: 输出目录路径，默认为 "tmp"

    Returns:
        str: 合并后的PLY文件路径。

    Example:
        >>> merged = gaussian_splatting_merge(
        ...     base_scene_ply="scene.ply",
        ...     new_asset_ply="building.ply",
        ...     position={'x': 0, 'y': 0.5, 'z': 0.3},
        ...     rotation={'x': 0, 'y': 0, 'z': 90},
        ...     scale={'x': 1.2, 'y': 1.2, 'z': 1.0},
        ...     step=1
        ... )
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 转换字典参数为numpy数组
    pos_array = np.array([position.get('x', 0),
                          position.get('y', 0),
                          position.get('z', 0)], dtype=np.float32)

    rot_array = [rotation.get('x', 0),
                 rotation.get('y', 0),
                 rotation.get('z', 0)]

    # 处理缩放参数（如果未提供则默认为1）
    if scale is None:
        scale_array = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        scale_array = np.array([scale.get('x', 1.0),
                                scale.get('y', 1.0),
                                scale.get('z', 1.0)], dtype=np.float32)

    print(f"\n[Gaussian Splatting Merge] Step {step}")
    print(f"  📦 New asset: {new_asset_ply}")
    print(f"  📍 Position: {pos_array}")
    print(f"  🔄 Rotation: {rot_array}°")
    print(f"  📏 Scale: {scale_array}")

    all_vertices_data = []
    vertex_dtype = None

    try:
        # 1. 如果存在基础场景，先加载它
        if base_scene_ply and os.path.exists(base_scene_ply):
            print(f"  ⏳ Loading base scene: {base_scene_ply}")
            base_ply = PlyData.read(base_scene_ply)
            base_vertices = base_ply['vertex']
            all_vertices_data.append(base_vertices.data)
            vertex_dtype = base_vertices.data.dtype
            print(f"     ✓ Base scene loaded: {len(base_vertices.data)} vertices")

        # 2. 加载新资产
        if not os.path.exists(new_asset_ply):
            raise FileNotFoundError(f"Asset file not found: {new_asset_ply}")

        print(f"  ⏳ Loading new asset: {new_asset_ply}")
        asset_ply = PlyData.read(new_asset_ply)
        asset_vertices = asset_ply['vertex']

        # 如果这是第一个加载的文件，记录数据结构
        if vertex_dtype is None:
            vertex_dtype = asset_vertices.data.dtype

        # 3. 提取顶点坐标
        points = np.vstack([
            asset_vertices['x'],
            asset_vertices['y'],
            asset_vertices['z']
        ]).T

        print(f"     ✓ Asset loaded: {len(points)} vertices")

        # 4. 应用变换（缩放 -> 旋转 -> 平移）
        print(f"  🔧 Applying transformations...")

        # a. 缩放
        if not np.allclose(scale_array, [1.0, 1.0, 1.0]):
            points = points * scale_array
            print(f"     ✓ Scaled by {scale_array}")

        # b. 旋转
        if any(r != 0 for r in rot_array):
            rotation_matrix = R.from_euler('xyz', rot_array, degrees=True).as_matrix()
            points = points @ rotation_matrix.T
            print(f"     ✓ Rotated by {rot_array}°")

        # c. 平移
        if not np.allclose(pos_array, [0, 0, 0]):
            points = points + pos_array
            print(f"     ✓ Translated by {pos_array}")

        # 5. 更新变换后的顶点数据
        transformed_data = np.copy(asset_vertices.data)
        transformed_data['x'] = points[:, 0]
        transformed_data['y'] = points[:, 1]
        transformed_data['z'] = points[:, 2]

        all_vertices_data.append(transformed_data)

        # 6. 合并所有顶点
        print(f"  🔗 Merging vertices...")
        final_vertices = np.concatenate(all_vertices_data)
        print(f"     ✓ Total vertices: {len(final_vertices)}")

        # 7. 创建并保存PLY文件
        output_filename = f"scene_merged_step_{step}.ply"
        output_path = os.path.join(output_dir, output_filename)

        final_element = PlyElement.describe(final_vertices, 'vertex')
        final_ply = PlyData([final_element])

        print(f"  💾 Saving to: {output_path}")
        final_ply.write(output_path)
        print(f"  ✅ Merge completed successfully!\n")

        return output_path

    except Exception as e:
        print(f"  ❌ Error during merge: {e}")
        raise

# =================================================================================
#  load_ply 函数 (无需修改)
# =================================================================================
def load_ply(path, device="cuda"):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    opacities = torch.sigmoid(torch.tensor(vertices['opacity'], dtype=torch.float32, device=device))
    scales = torch.exp(torch.tensor(np.vstack([
        vertices['scale_0'], vertices['scale_1'], vertices['scale_2']
    ]), dtype=torch.float32, device=device).T)
    rotations = torch.tensor(np.vstack([
        vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']
    ]), dtype=torch.float32, device=device).T
    C0 = 0.28209479177387814
    rgbs = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T
    rgbs = C0 * rgbs + 0.5
    rgbs = np.clip(rgbs, 0.0, 1.0)
    return (
        torch.tensor(points, dtype=torch.float32, device=device),
        scales,
        torch.nn.functional.normalize(rotations),
        torch.tensor(rgbs, dtype=torch.float32, device=device),
        opacities.unsqueeze(-1),
    )


# =================================================================================
#  校正模型方向的函数 (保留备用)
# =================================================================================
def correct_model_orientation(means, scales, quats):
    """保留此函数以备后用"""
    device = means.device
    angle = -math.pi / 2
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    correction_matrix_3x3 = torch.tensor([
        [1, 0, 0],
        [0, cos_a, sin_a],
        [0, -sin_a, cos_a]
    ], dtype=torch.float32, device=device)
    angle_half = angle / 2.0
    correction_quat = torch.tensor(
        [math.cos(angle_half), math.sin(angle_half), 0, 0],
        dtype=torch.float32, device=device
    )
    means_corrected = (means @ correction_matrix_3x3.T).float()
    qw_c, qx_c, qy_c, qz_c = correction_quat.unbind()
    qw, qx, qy, qz = quats.unbind(dim=-1)
    quats_corrected = torch.stack([
        qw_c * qw - qx_c * qx - qy_c * qy - qz_c * qz,
        qw_c * qx + qx_c * qw + qy_c * qz - qz_c * qy,
        qw_c * qy - qx_c * qz + qy_c * qw + qz_c * qx,
        qw_c * qz + qx_c * qy - qy_c * qx + qz_c * qw,
    ], dim=-1).float()
    print("模型坐标已校正：绕X轴旋转-90度。")
    return means_corrected, scales, quats_corrected


# =================================================================================
#  render_view 函数
# =================================================================================
def render_view(
        means, scales, quats, rgbs, opacities,
        width, height,
        elevation_deg, azimuth_deg,
        output_path
):
    device = means.device
    scene_center = means.mean(dim=0)
    scene_size = torch.max(torch.sqrt(torch.sum((means - scene_center) ** 2, dim=1))).item()
    camera_distance = scene_size * 2.5
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    x_offset = camera_distance * math.cos(elevation) * math.sin(azimuth)
    y_offset = camera_distance * math.sin(elevation)
    z_offset = camera_distance * math.cos(elevation) * math.cos(azimuth)
    camera_pos = scene_center + torch.tensor([x_offset, y_offset, z_offset], device=device, dtype=torch.float32)
    look_at = scene_center

    if abs(elevation_deg - 90.0) < 1e-3:
        up_vector = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)
    elif abs(elevation_deg + 90.0) < 1e-3:
        up_vector = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    else:
        up_vector = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    forward_dir = torch.nn.functional.normalize(look_at - camera_pos, dim=-1)
    right_dir = torch.nn.functional.normalize(torch.cross(forward_dir, up_vector), dim=-1)
    up_dir = torch.nn.functional.normalize(torch.cross(right_dir, forward_dir), dim=-1)
    c2w = torch.eye(4, device=device, dtype=torch.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right_dir, up_dir, forward_dir, camera_pos
    world_to_view = torch.linalg.inv(c2w).unsqueeze(0)
    fovy = math.radians(49.1)
    fy = height / (2 * math.tan(fovy / 2))
    fx = fy
    K = torch.tensor([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], device=device,
                     dtype=torch.float32).unsqueeze(0)
    backgrounds = torch.ones(3, device=device, dtype=torch.float32)

    print(f"   - 正在渲染视角 (仰角={elevation_deg}°, 方位角={azimuth_deg}°)...")

    outputs, _, _ = rasterization(
        means=means.float(),
        quats=quats.float(),
        scales=scales.float(),
        opacities=opacities.squeeze(-1).float(),
        colors=rgbs.float(),
        viewmats=world_to_view.float(),
        Ks=K.float(),
        height=height,
        width=width,
        render_mode='RGB',
        backgrounds=backgrounds.float()
    )

    save_image(outputs[0].permute(2, 0, 1), output_path)
    print(f"   - 图像已保存到 '{output_path}'")
    return output_path


# =================================================================================
#  封装的高斯渲染快照函数
# =================================================================================
def gaussian_splatting_snapshot(
        scene_ply: Optional[str],
        camera_mode: str,
        info: str,
        target_pos: Optional[Dict] = None,
        width: int = 1024,
        height: int = 1024,
        apply_correction: bool = False,
        output_dir: str = "tmp"
) -> Dict[str, str]:
    """
    【已升级】为高斯场景生成快照。

    参数:
        scene_ply: .ply文件路径，如果为None则返回空结果
        camera_mode: 相机模式，可选 "all"(全部视角), "front", "top", "left", "perspective"
        info: 用于文件命名的标识信息
        target_pos: 可选的目标位置字典（预留参数，暂未使用）
        width: 渲染宽度
        height: 渲染高度
        apply_correction: 是否应用坐标校正
        output_dir: 输出目录

    返回:
        Dict[str, str]: 视角名称到图片路径的映射
            例如: {"front": "tmp/snapshot_info_front.png", "top": "tmp/snapshot_info_top.png", ...}
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 如果没有提供场景文件，返回空结果
    if scene_ply is None:
        scene_name = "empty_scene"
        print(f"   - [WARNING] 未提供场景文件，无法生成快照")
        return {}

    scene_name = os.path.basename(scene_ply)
    print(f"   - [Gaussian Splatting] 正在为场景 '{scene_name}' 生成 '{camera_mode}' 快照 (info: '{info}')...")

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - 使用设备: {device}")

    # 加载PLY文件
    try:
        (means, scales, quats, rgbs, opacities) = load_ply(scene_ply, device=device)
        print(f"   - 成功加载 {means.shape[0]} 个高斯球")
    except Exception as e:
        print(f"   - [ERROR] 加载 .ply 文件时出错: {e}")
        return {}

    # 应用坐标校正（如果需要）
    if apply_correction:
        means, scales, quats = correct_model_orientation(means, scales, quats)

    # 打印物体尺寸信息
    min_coords, _ = torch.min(means, dim=0)
    max_coords, _ = torch.max(means, dim=0)
    dimensions = max_coords - min_coords
    print(
        f"   - 物体尺寸 (X/Y/Z): {dimensions[0].item():.3f} / {dimensions[1].item():.3f} / {dimensions[2].item():.3f}")

    # 定义视角配置
    all_views = {
        "front": {"elevation": 0, "azimuth": 180},
        "top": {"elevation": -90, "azimuth": 0},
        "left": {"elevation": 0, "azimuth": 270},
        "perspective": {"elevation": -150, "azimuth": 45},
    }

    # 根据camera_mode选择要渲染的视角
    if camera_mode.lower() == "all":
        views_to_render = all_views
    elif camera_mode.lower() in all_views:
        views_to_render = {camera_mode.lower(): all_views[camera_mode.lower()]}
    else:
        print(f"   - [WARNING] 未知的相机模式 '{camera_mode}'，将渲染所有视角")
        views_to_render = all_views

    # 渲染各个视角
    snapshot_paths = {}
    for view_name, angles in views_to_render.items():
        # 生成输出文件名
        output_filename = f"snapshot_{info}_{view_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        # 渲染视角
        try:
            rendered_path = render_view(
                means, scales, quats, rgbs, opacities,
                width, height,
                angles["elevation"], angles["azimuth"],
                output_path
            )
            snapshot_paths[view_name] = rendered_path
        except Exception as e:
            print(f"   - [ERROR] 渲染 '{view_name}' 视角时出错: {e}")

    print(f"   - [完成] 成功生成 {len(snapshot_paths)} 个快照")
    return snapshot_paths


# =================================================================================
#  向后兼容的生成函数
# =================================================================================
def generate_views(
        ply_path: str = "test.ply",
        base_output_path: str = "output.png",
        width: int = 1024,
        height: int = 768,
        apply_correction: bool = False
):
    """向后兼容的生成函数"""
    path_without_ext, ext = os.path.splitext(base_output_path)
    output_dir = os.path.dirname(base_output_path) or "."
    info = os.path.basename(path_without_ext)

    snapshot_paths = gaussian_splatting_snapshot(
        scene_ply=ply_path,
        camera_mode="all",
        info=info,
        width=width,
        height=height,
        apply_correction=apply_correction,
        output_dir=output_dir
    )

    return snapshot_paths


if __name__ == "__main__":
    # 示例1: 使用新的封装函数（推荐）
    snapshot_paths = gaussian_splatting_snapshot(
        scene_ply=r"D:\github\BuildingAgent\model_api\generated_model_2fac3c83\gaussian_f1ef57b4.ply",
        camera_mode="all",  # 可选: "all", "front", "top", "left", "perspective"
        info="test_render",
        width=1024,
        height=1024,
        apply_correction=False,
        output_dir="tmp"
    )

    print("\n生成的快照路径:")
    for view_name, path in snapshot_paths.items():
        print(f"  {view_name}: {path}")

    # 示例1: 使用真实函数
    print("=" * 60)
    print("示例 1: 真实PLY合并")
    print("=" * 60)

    try:
        merged_scene = gaussian_splatting_merge(
            base_scene_ply=None,  # 第一个物体，无基础场景
            new_asset_ply="test3.ply",
            position={'x': 0, 'y': 0.4865, 'z': 0.2986},
            rotation={'x': 0, 'y': 0, 'z': 0},
            scale={'x': 1.0, 'y': 1.0, 'z': 1.0},
            step=1
        )

        # 添加第二个物体
        merged_scene = gaussian_splatting_merge(
            base_scene_ply=merged_scene,  # 使用上一步的结果
            new_asset_ply="test4.ply",
            position={'x': 0, 'y': -0.4327, 'z': 0.21115},
            rotation={'x': 0, 'y': 0, 'z': 90},
            scale={'x': 1.2, 'y': 1.2, 'z': 1.0},  # 缩放1.2倍
            step=2
        )

        # 添加第三个物体
        merged_scene = gaussian_splatting_merge(
            base_scene_ply=merged_scene,
            new_asset_ply="test5.ply",
            position={'x': 0, 'y': 0, 'z': 0},
            rotation={'x': 0, 'y': 0, 'z': 0},
            scale={'x': 0.8, 'y': 0.8, 'z': 1.5},  # 自定义缩放
            step=3
        )

        print(f"\n🎉 最终场景: {merged_scene}")

    except Exception as e:
        print(f"\n⚠️  真实文件不存在，跳过真实测试: {e}")

