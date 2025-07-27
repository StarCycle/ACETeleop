import sapien.core as sapien
from sapien.utils import Viewer
import torch
import math
import numpy as np
from pathlib import Path
import argparse
import time
import yaml
from copy import deepcopy
from typing import Dict, Any

from avp_stream.utils.se3_utils import *
from avp_stream.utils.trn_constants import *

import ace_teleop
from ace_teleop.control.teleop import ACETeleop

def load_config(config_file_name: str) -> Dict[str, Any]:
    robot_config_path = (
        f"{ace_teleop.__path__[0]}/configs/robot" / Path(config_file_name)
    )
    with Path(robot_config_path).open("r") as f:
        cfg = yaml.safe_load(f)["robot_cfg"]

    return cfg


def np2tensor(
    data: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in data.items()
    }


def gym_quat_to_sapien(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


class Sim:
    def __init__(
        self,
        cfg: Dict[str, Any],
        collision: bool = False,
        print_freq: bool = False,
        debug: bool = False,
    ) -> None:
        self.print_freq = print_freq
        self.debug = debug

        # initialize sapien engine, render, scene and viewer
        self.engine = sapien.Engine()
        self.render = sapien.SapienRenderer()
        self.engine.set_renderer(self.render)

        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 60.0)

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self.viewer = Viewer(self.render)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(1, 1, 2)
        self.viewer.set_camera_rpy(0, 0, 0.7*math.pi)

        # add ground plane
        self.scene.add_ground(0.0)

        np.random.seed(17)

        # add robot asset
        robot_asset_root = Path(ace_teleop.__path__[0]) / "assets"
        robot_asset_file = cfg["urdf_path"]

        self.loader = self.scene.create_urdf_loader()
        if collision:
            self.loader.set_material(static_friction=20.0, dynamic_friction=20.0, restitution=0.0)

        self.robot = self.loader.load(str(robot_asset_root / robot_asset_file))
        self.dof = self.robot.dof
        self.set_dof_properties()

        # set robot pose
        self.robot.set_root_pose(sapien.Pose([0, 0, 1.1], [1, 0, 0, 0]))

        self.robot.set_qpos(np.zeros(self.dof))
        self.robot.set_drive_target(np.zeros(self.dof))

        if self.debug:
            axis_root = Path(ace_teleop.__path__[0]) / "assets" / "axis"
            self.axis_builder = self.create_axis_builder("normal", axis_root)
            self.small_axis_builder = self.create_axis_builder("small", axis_root)
            self.huge_axis_builder = self.create_axis_builder("huge", axis_root)

            self.sphere_builder = self.scene.create_actor_builder()
            self.sphere_builder.add_sphere_visual(radius=0.008, color=[1, 1, 1, 1])

            self.add_actors()

        self.right_finger_spheres = []
        self.left_finger_spheres = []
        self.right_small_axes = []
        self.left_small_axes = []

    def create_axis_builder(self, size: str, root: Path):
        builder = self.scene.create_actor_builder()
        asset_path = root / f"{size}.usd"
        builder.add_visual_from_file(str(asset_path))
        return builder

    def set_dof_properties(self) -> None:
        for joint in self.robot.get_active_joints():
            joint.set_drive_property(stiffness=10000.0, damping=10000.0)

    def add_actors(self) -> None:
        self.head_axis = self.axis_builder.build(kinematic=True, name="head")
        self.right_wrist_axis = self.axis_builder.build(kinematic=True, name="right_wrist")
        self.left_wrist_axis = self.axis_builder.build(kinematic=True, name="left_wrist")

        self.add_spheres()
        self.add_small_axes()
        self.env_axis = self.huge_axis_builder.build(kinematic=True, name="env_axis")

    def add_spheres(self) -> None:
        for i in range(25):
            actor = self.sphere_builder.build(kinematic=True, name=f"right_finger_{i}")
            color = [1, 1, 0, 1] if i in [0, 4, 9, 14, 19, 24] else [1, 1, 1, 1]
            actor.visual_bodies[0].rgba = color
            self.right_finger_spheres.append(actor)

        for i in range(25):
            actor = self.sphere_builder.build(kinematic=True, name=f"left_finger_{i}")
            color = [1, 1, 0, 1] if i in [0, 4, 9, 14, 19, 24] else [1, 1, 1, 1]
            actor.visual_bodies[0].rgba = color
            self.left_finger_spheres.append(actor)

    def add_small_axes(self) -> None:
        for i in range(25):
            actor = self.small_axis_builder.build(kinematic=True, name=f"right_finger_axis_{i}")
            self.right_small_axes.append(actor)

        for i in range(25):
            actor = self.small_axis_builder.build(kinematic=True, name=f"left_finger_axis_{i}")
            self.left_small_axes.append(actor)



    def step(
        self, cmd: np.ndarray, transformation: Dict[str, torch.Tensor] = None
    ) -> None:
        if self.print_freq:
            start = time.time()

        # Set robot DOF targets
        # self.robot.set_drive_target(cmd)
        self.robot.set_qpos(cmd) # similar to issac gym code

        # Step the physics
        self.scene.step()

        if self.debug and transformation is not None:
            self.update_debug_poses(transformation)

        self.viewer.render()

        if self.print_freq:
            end = time.time()
            print("Frequency:", 1 / (end - start))

    def update_debug_poses(
        self, transformations: Dict[str, torch.Tensor]
    ) -> None:
        visionos_head = transformations["head"].clone()
        visionos_head[0, 2, 3] += 0.25
        head_pq = mat2posquat(visionos_head)[0].cpu().numpy()
        head_p = head_pq[:3]
        head_q_xyzw = head_pq[3:]
        head_q = gym_quat_to_sapien(head_q_xyzw)
        self.head_axis.set_pose(sapien.Pose(head_p, head_q))

        sim_right_wrist = transformations["right_wrist"].clone()
        sim_right_wrist[0, 2, 3] += 1.1
        rw_pq = mat2posquat(sim_right_wrist)[0].cpu().numpy()
        rw_p = rw_pq[:3]
        rw_q_xyzw = rw_pq[3:]
        rw_q = gym_quat_to_sapien(rw_q_xyzw)
        self.right_wrist_axis.set_pose(sapien.Pose(rw_p, rw_q))

        sim_left_wrist = transformations["left_wrist"].clone()
        sim_left_wrist[0, 2, 3] += 1.1
        lw_pq = mat2posquat(sim_left_wrist)[0].cpu().numpy()
        lw_p = lw_pq[:3]
        lw_q_xyzw = lw_pq[3:]
        lw_q = gym_quat_to_sapien(lw_q_xyzw)
        self.left_wrist_axis.set_pose(sapien.Pose(lw_p, lw_q))

        sim_right_fingers = torch.cat(
            [
                sim_right_wrist @ finger
                for finger in transformations["right_fingers"]
            ],
            dim=0,
        )
        rf_pqs = mat2posquat(sim_right_fingers).cpu().numpy()

        sim_left_fingers = torch.cat(
            [
                sim_left_wrist @ finger
                for finger in transformations["left_fingers"]
            ],
            dim=0,
        )
        lf_pqs = mat2posquat(sim_left_fingers).cpu().numpy()

        for i in range(25):
            rf_pq = rf_pqs[i]
            rf_p = rf_pq[:3]
            rf_q_xyzw = rf_pq[3:]
            rf_q = gym_quat_to_sapien(rf_q_xyzw)
            self.right_finger_spheres[i].set_pose(sapien.Pose(rf_p, rf_q))
            self.right_small_axes[i].set_pose(sapien.Pose(rf_p, rf_q))

            lf_pq = lf_pqs[i]
            lf_p = lf_pq[:3]
            lf_q_xyzw = lf_pq[3:]
            lf_q = gym_quat_to_sapien(lf_q_xyzw)
            self.left_finger_spheres[i].set_pose(sapien.Pose(lf_p, lf_q))
            self.left_small_axes[i].set_pose(sapien.Pose(lf_p, lf_q))

    def end(self) -> None:
        self.viewer.close()


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        choices=["h1_inspire", "xarm_ability", "xarm_ability_right", "gr1", "franka", "ur10e_right", "ur10e_hand_right"],
        default="h1_inspire",
    )
    parser.add_argument("--ip", default="localhost")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_file_name = f"{args.config}.yml"
    cfg = load_config(config_file_name)

    teleoperator = ACETeleop(cfg, args.ip, debug=args.debug)
    simulator = Sim(cfg, print_freq=False, debug=args.debug)

    try:
        while not simulator.viewer.closed:
            if args.debug:
                cmd, latest = teleoperator.step()
                simulator.step(cmd, np2tensor(latest, simulator.device))
            else:
                cmd = teleoperator.step()
                simulator.step(cmd)
    except KeyboardInterrupt:
        simulator.end()
        exit(0)


if __name__ == "__main__":
    main()