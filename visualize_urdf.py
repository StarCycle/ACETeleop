import sapien.core as sapien
import numpy as np
from sapien.utils import Viewer
from sapien.core import SapienRenderer  # 新增渲染器导入
import pinocchio as pin

def visualize_urdf(urdf_path):
    # 初始化引擎和渲染器
    engine = sapien.Engine()
    renderer = SapienRenderer()  # 创建Vulkan渲染器
    engine.set_renderer(renderer)  # 绑定渲染器到引擎
    
    # 创建场景
    scene = engine.create_scene()
    scene.set_timestep(1 / 240)
    scene.add_ground(altitude=0)
    
    # 加载URDF模型和设置初始位置
    loader = scene.create_urdf_loader()
    robot = loader.load(urdf_path)
    if not robot:
        raise ValueError(f"Failed to load URDF from {urdf_path}")
    robot.set_pose(sapien.Pose([0, 0, 0]))

    # 打印Sapien中的关节顺序（活动关节）
    print("Joint Names and Indices in Sapien:")
    active_joints = robot.get_active_joints()
    for i, joint in enumerate(active_joints):
        print(f"Index {i}: Joint {joint.get_name()}")
    
    # 创建查看器
    viewer = Viewer(renderer)  # 传入渲染器而非Scene
    viewer.set_scene(scene)
    
    # 设置相机参数和光照
    scene.set_ambient_light([0.5, 0.5, 0.5])  # 环境光强度
    scene.add_directional_light(direction=[0, 1, -1], color=[1, 1, 1])  # 方向光源
    viewer.set_camera_xyz(x=0.3, y=0, z=0.2)
    viewer.set_camera_rpy(r=0, p=0, y=-np.pi)
    
    # 主循环
    while not viewer.closed:
        # scene.step()
        scene.update_render()
        viewer.render()

if __name__ == '__main__':
    urdf_path = "/home/starcycle/ACETeleop/ace_teleop/assets/ur10e/ur10e_with_right_dexrobot_hand.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    print("Joint Names and Indices in Pinocchio:")
    for i, joint in enumerate(model.names[1:]):  # 跳过universe
        print(f"Index {i}: Joint {joint}")
    visualize_urdf(urdf_path)