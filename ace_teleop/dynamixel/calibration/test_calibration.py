import argparse
from typing import List, Dict, Tuple

import matplotlib
import numpy as np
import sapien.core as sapien
import transforms3d
from pathlib import Path
from sapien.utils import Viewer

from ace_teleop.server.dynamixel_agent import DynamixelAgent

COLOR_MAP = matplotlib.colormaps["Reds"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rt",
        "--use_rt",
        action="store_true",
        default=False,
        help="Whether to use ray tracing for rendering.",
    )
    parser.add_argument(
        "-s",
        "--simulate",
        action="store_true",
        default=True,
        help="Whether to physically simulate the urdf.",
    )
    parser.add_argument(
        "-f",
        "--fix-root",
        action="store_true",
        default=True,
        help="Whether to fix the root link of the urdf.",
    )
    parser.add_argument(
        "--disable-self-collision",
        action="store_true",
        default=False,
        help="Whether to disable the self collision of the urdf.",
    )
    parser.add_argument(
        "--port", type=str, required=True, help="Port to connect the Dynamixel agent."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["right", "right_m", "left", "left_m"],
        required=True,
        help="Type of the URDF to use for the agent.",
    )
    return parser.parse_args()


def create_cone_mesh(slices=16):
    vertices = np.zeros((slices + 1, 3), dtype=np.float32)
    vertices[0] = [1, 0, 0]  # tip
    for i in range(slices):
        theta = 2 * np.pi * i / slices
        vertices[i + 1] = [0, np.cos(theta), np.sin(theta)]
    triangles = np.zeros((slices, 3), dtype=np.uint32)
    for i in range(slices):
        triangles[i] = [0, i + 1, (i + 1) % slices + 1]
    return vertices, triangles


class ContactViewer(Viewer):
    def __init__(
        self,
        renderer,
        shader_dir="",
    ):
        super().__init__(renderer)

        # Contact arrow
        self.contact_actors = []
        self.highlighted_actors = []
        self.cone_vertices, self.cone_triangles = create_cone_mesh(16)

        # Material to highlight contact geom
        self.contact_collision_mat = renderer.create_material()
        self.contact_collision_mat.base_color = np.array([1, 0, 0, 1])

        self.scene = None
        self.renderer = renderer

    def set_scene(self, scene):
        super().set_scene(scene)
        self.scene = scene

    def draw_contact(self):
        # Clear contact arrows and highlights from the previous step
        for actor in self.contact_actors:
            self.scene.remove_actor(actor)
        self.contact_actors.clear()

        for actor in self.highlighted_actors:
            self.scene.remove_actor(actor)
        self.highlighted_actors.clear()

        # Fetch the contact information for current step
        contact_list, actor_geom_map = self.fetch_contact()

        # Draw collision visual shape
        for actor, geom_list in actor_geom_map.items():
            builder = self.scene.create_actor_builder()
            for collision_shape in geom_list:
                shape_type = collision_shape.type
                shape_pose = collision_shape.pose
                if shape_type == 'sphere':
                    data = collision_shape.get_geometric_shape_data()
                    builder.add_sphere_visual(
                        radius=data['radius'],
                        material=self.contact_collision_mat,
                        pose=shape_pose,
                    )
                elif shape_type == 'box':
                    data = collision_shape.get_geometric_shape_data()
                    builder.add_box_visual(
                        half_size=data['half_sizes'],
                        material=self.contact_collision_mat,
                        pose=shape_pose,
                    )
                elif shape_type == 'capsule':
                    data = collision_shape.get_geometric_shape_data()
                    builder.add_capsule_visual(
                        radius=data['radius'],
                        half_length=data['half_length'],
                        material=self.contact_collision_mat,
                        pose=shape_pose,
                    )
                elif shape_type == 'convex_mesh' or shape_type == 'nonconvex_mesh':
                    vertices = collision_shape.get_vertices()
                    triangles = collision_shape.get_triangles()
                    scale = collision_shape.get_scale()
                    builder.add_visual_mesh(
                        vertices,
                        triangles,
                        material=self.contact_collision_mat,
                        pose=shape_pose,
                        scale=scale,
                    )
                elif shape_type == 'plane':
                    builder.add_plane_visual(
                        scale=[1e4, 1e4],
                        material=self.contact_collision_mat,
                        pose=shape_pose,
                    )
                elif shape_type == 'cylinder':
                    data = collision_shape.get_geometric_shape_data()
                    builder.add_cylinder_visual(
                        radius=data['radius'],
                        half_length=data['half_length'],
                        material=self.contact_collision_mat,
                        pose=shape_pose,
                    )
                else:
                    raise Exception("invalid collision shape, this code should be unreachable.")
            highlight_actor = builder.build_kinematic(name="contact_highlight")
            highlight_actor.set_pose(actor.get_pose())
            for rb in highlight_actor.get_render_bodies():
                rb.shade_flat = True
            self.highlighted_actors.append(highlight_actor)

        # Draw contact arrow
        for pos, normal, color in contact_list:
            self.draw_contact_arrow(pos, normal, color)

    def fetch_contact(
        self,
    ) -> Tuple[
        List,
        Dict[
            sapien.Actor, List[sapien.CollisionShape]
        ],
    ]:
        min_impulse = 0.1
        max_impulse = 10
        contact_list = []

        actor_geom_map = {}
        for contact in self.scene.get_contacts():
            impulse = np.array([p.impulse for p in contact.points])
            total_impulse = np.linalg.norm(np.sum(impulse, axis=0))
            impulse_value = np.linalg.norm(impulse, axis=1)
            if total_impulse > min_impulse:
                weight = impulse_value / np.sum(impulse_value)
                position = np.sum(
                    np.array([p.position for p in contact.points]) * weight[:, None],
                    axis=0,
                )
                norm_impulse = np.clip(
                    (1 / total_impulse - 1 / min_impulse)
                    / (1 / max_impulse - 1 / min_impulse),
                    0,
                    1,
                )
                color = np.array(COLOR_MAP(norm_impulse))
                contact_list.append((position, np.sum(impulse, axis=0), color))
                actor0, actor1 = contact.actor1, contact.actor2
                print(
                    f"Find self collision: {actor0.name, actor1.name},"
                    f" impulse: {total_impulse}, position: {position}"
                )

                for act, shape in zip(
                    [actor0, actor1], [contact.collision_shape1, contact.collision_shape2]
                ):
                    if act in actor_geom_map:
                        actor_geom_map[act].append(shape)
                    else:
                        actor_geom_map[act] = [shape]

        return contact_list, actor_geom_map

    @staticmethod
    def compute_rotation_from_normal(normal: np.ndarray):
        normal = normal / np.linalg.norm(normal)
        x, y, z = normal
        if np.isclose(z, 1.0):
            return np.array([0.707, 0, -0.707, 0])
        elif np.isclose(z, -1.0):
            return np.array([0.707, 0, 0.707, 0])
        xy_sqrt = np.sqrt(x * x + y * y)
        y_axis = [y / xy_sqrt, -x / xy_sqrt, 0]
        z_axis = [x * z / xy_sqrt, y * z / xy_sqrt, -xy_sqrt]

        rotation_matrix = np.stack([normal, y_axis, z_axis], axis=1)
        quat = transforms3d.quaternions.mat2quat(rotation_matrix)
        return quat

    def draw_contact_arrow(
        self, pos: np.ndarray, normal: np.ndarray, color: np.ndarray
    ):
        builder = self.scene.create_actor_builder()
        material = self.renderer.create_material()
        material.emission = [1, 0, 0, 0]
        material.base_color = color
        material.specular = 0
        material.roughness = 0.8
        material.metallic = 0

        # Shaft (capsule)
        shaft_radius = 0.1 * 0.05
        shaft_half_length = 0.5 * 0.05
        shaft_pose = sapien.Pose([0.5 * 0.05, 0, 0])
        builder.add_capsule_visual(
            radius=shaft_radius,
            half_length=shaft_half_length,
            material=material,
            pose=shaft_pose,
        )

        # Head (cone)
        cone_pose = sapien.Pose([1 * 0.05, 0, 0])
        cone_scale = [0.5 * 0.05, 0.2 * 0.05, 0.2 * 0.05]
        builder.add_visual_mesh(
            self.cone_vertices,
            self.cone_triangles,
            material=material,
            pose=cone_pose,
            scale=cone_scale,
        )

        contact_actor = builder.build_kinematic(name="contact_arrow")
        contact_actor.set_pose(sapien.Pose(pos, self.compute_rotation_from_normal(normal)))
        for rb in contact_actor.get_render_bodies():
            rb.shade_flat = True
            rb.cast_shadow = False
        self.contact_actors.append(contact_actor)


def generate_joint_limit_trajectory(
    robot: sapien.Articulation, loop_steps: int
):
    joint_limits = robot.get_qlimits()
    for index, joint in enumerate(robot.get_active_joints()):
        if joint.type == "continuous":
            joint_limits[:, index] = [0, np.pi * 2]

    trajectory_via_points = np.stack(
        [joint_limits[:, 0], joint_limits[:, 1], joint_limits[:, 0]], axis=1
    )
    times = np.linspace(0.0, 1.0, int(loop_steps))
    bins = np.arange(3) / 2.0

    # Compute alphas for each time
    inds = np.digitize(times, bins, right=True)
    inds[inds == 0] = 1
    alphas = (bins[inds] - times) / (bins[inds] - bins[inds - 1])

    # Create the new interpolated trajectory
    trajectory = (
        alphas * trajectory_via_points[:, inds - 1]
        + (1.0 - alphas) * trajectory_via_points[:, inds]
    )
    return trajectory.T


def visualize_urdf(
    use_rt, simulate, disable_self_collision, fix_root, port, agent_type
):
    # No ray tracing in Sapien 2.2, but we can set shader for rt if available
    if use_rt:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 64
        sapien.render_config.rt_use_denoiser = True

    # Setup
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    config = sapien.SceneConfig()
    config.gravity = np.array([0, 0, 0])
    scene = engine.create_scene(config=config)
    scene.set_timestep(1 / 125)

    # Ground
    render_mat = renderer.create_material()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-1, render_material=render_mat)

    # Lighting
    scene.set_ambient_light(np.array([0.6, 0.6, 0.6]))
    scene.add_directional_light(np.array([1, 1, -1]), np.array([1, 1, 1]))
    scene.add_directional_light([0, 0, -1], [1, 1, 1])
    scene.add_point_light(np.array([2, 2, 2]), np.array([1, 1, 1]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([1, 1, 1]), shadow=False)

    # Viewer
    viewer = ContactViewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(0.5, 0, 0.5)
    viewer.set_camera_rpy(0, -0.8, 3.14)
    # viewer.window.show_origin_frame = False  # No such attribute in 2.2, perhaps
    # viewer.window.move_speed = 0.01  # Assuming similar

    # Articulation
    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = False
    urdf_paths = {
        "right": "right_arm/robot.urdf",
        "right_m": "right_arm_m/robot.urdf",
        "left": "left_arm/robot.urdf",
        "left_m": "left_arm_m/robot.urdf",
    }
    urdf_path = str(
        Path(__file__).resolve().parent.parent / "urdf" / Path(urdf_paths[agent_type])
    )
    robot = loader.load(urdf_path)

    # Robot motion
    loop_steps = 600
    trajectory = np.array([])
    if simulate:
        for joint in robot.get_active_joints():
            joint.set_drive_property(1000, 50)
        trajectory = generate_joint_limit_trajectory(robot, loop_steps=loop_steps)

    robot.set_qpos(np.zeros([robot.dof]))

    agent = DynamixelAgent(
        port=port,
        urdf=urdf_path,
        ee_link_name="ee_c",
    )

    step = 0

    while not viewer.closed:
        action = agent.get_joints()
        robot.set_qpos(action)

        viewer.render()
        if simulate:
            qpos = trajectory[step]
            for joint, single_qpos in zip(robot.get_active_joints(), qpos):
                joint.set_drive_target(single_qpos)
            robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True, external=False))
            step += 1
            step = step % loop_steps

            viewer.draw_contact()


def main():
    args = parse_args()
    visualize_urdf(
        args.use_rt,
        args.simulate,
        args.disable_self_collision,
        args.fix_root,
        args.port,
        args.type,
    )


if __name__ == "__main__":
    main()