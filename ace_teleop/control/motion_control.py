import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve
from typing import List, Optional, Dict

# Import pink modules for inverse kinematics
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit

# Import qpsolvers for selecting a QP solver
import qpsolvers


class PinocchioMotionControl:
    """
    Class for controlling motion using Pinocchio for forward kinematics and Pink for inverse kinematics.
    This replaces the original CLIK-based IK with Pink's task-based IK solver.
    """

    def __init__(
        self,
        urdf_path: str,
        wrist_name: str,
        arm_init_qpos: np.ndarray,
        arm_config: Dict[str, float],
        arm_indices: Optional[List[int]] = [],
    ) -> None:
        """
        Initialize the motion control class.
        - Load the robot model from URDF.
        - Optionally reduce the model if arm_indices are provided.
        - Set up joint limits.
        - Initialize Pink configuration and tasks for IK.
        """
        # Store arm indices for model reduction if provided
        self.arm_indices = arm_indices
        # Smoothing factor for interpolating between current and target qpos
        self.alpha = float(arm_config["out_lp_alpha"])

        # Load the full Pinocchio model from URDF
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.dof = self.model.nq

        # If arm_indices are provided, create a reduced model by locking other joints
        if arm_indices:
            locked_joint_ids = list(set(range(self.dof)) - set(self.arm_indices))
            locked_joint_ids = [
                id + 1 for id in locked_joint_ids
            ]  # Account for universe joint
            self.model = pin.buildReducedModel(
                self.model, locked_joint_ids, np.zeros(self.dof)
            )
        # Update arm DOF after potential reduction
        self.arm_dof = self.model.nq

        # Store joint position limits from the model
        self.lower_limit = self.model.lowerPositionLimit
        self.upper_limit = self.model.upperPositionLimit
        # Create Pinocchio data object for computations
        self.data: pin.Data = self.model.createData()

        # Get the frame ID for the wrist (end-effector)
        self.wrist_id = self.model.getFrameId(wrist_name)

        # print(arm_init_qpos)

        # Initialize current joint positions (qpos)
        self.qpos = arm_init_qpos
        # Compute initial forward kinematics
        pin.forwardKinematics(self.model, self.data, self.qpos)
        # Get initial wrist pose
        self.wrist_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.wrist_id
        )

        # Damping factor for regularization in IK (from config)
        self.damp = float(arm_config["damp"])
        # Convergence threshold for IK error
        self.ik_eps = float(arm_config["eps"])
        # Time step for IK velocity integration
        self.dt = float(arm_config["dt"])

        # Initialize Pink-specific components
        # Create Pink configuration from Pinocchio model and initial qpos
        self.configuration = pink.Configuration(self.model, self.data, self.qpos.copy())

        # Define FrameTask for the wrist (end-effector tracking)
        self.wrist_task = FrameTask(
            wrist_name,
            position_cost=1.0,  # Cost for position error [cost/m]
            orientation_cost=1.0,  # Cost for orientation error [cost/rad]
            lm_damping=1.0,  # Levenberg-Marquardt damping for regularization
        )

        # Define PostureTask as regularization to prefer certain postures
        self.posture_task = PostureTask(
            cost=1e-3,  # Small cost for posture deviation [cost/rad]
        )

        # List of tasks for IK solver
        self.tasks = [self.wrist_task, self.posture_task]

        # Set initial targets for tasks from the current configuration
        for task in self.tasks:
            task.set_target_from_configuration(self.configuration)

        # Select QP solver (prefer 'osqp' if available, otherwise first available)
        self.solver = qpsolvers.available_solvers[0]
        if "osqp" in qpsolvers.available_solvers:
            self.solver = "osqp"

        # Configuration limits for safety in IK
        self.config_limit = ConfigurationLimit(self.model)

    def control(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """
        Compute the next joint positions to reach the target end-effector pose.
        - Set the target for the wrist task.
        - Compute IK using Pink to get target qpos.
        - Interpolate between current and target qpos.
        """
        # Create target SE3 transform for the wrist
        oMdes = pin.SE3(target_rot, target_pos)
        # Copy current qpos
        qpos = self.qpos.copy()

        # Compute target qpos using Pink IK
        ik_qpos = qpos.copy()
        # ik_qpos = self.ik_clik(ik_qpos, oMdes, self.wrist_id) # Original code
        ik_qpos = self.ik_pink(ik_qpos, oMdes, self.wrist_id)
        qpos = ik_qpos.copy()

        # Interpolate between current qpos and computed IK qpos using alpha
        self.qpos = pin.interpolate(self.model, self.qpos, qpos, self.alpha)
        self.qpos = qpos.copy()

        # Return the updated qpos
        return self.qpos.copy()

    def ik_pink(
        self, qpos: np.ndarray, oMdes: pin.SE3, wrist_id: int, max_iter: int = 1000
    ) -> np.ndarray:
        """
        Compute inverse kinematics using Pink's task-based solver with multiple iterations.
        - Iteratively update the configuration by computing velocity in each step.
        - Set the target pose for the wrist task once.
        - In each iteration: Solve IK for velocity, integrate to get incremental qpos update.
        - Check convergence after each iteration; exit early if error is below threshold.
        - This mirrors the iterative structure of ik_clik for consistency, allowing convergence
        within a single function call, rather than relying on external control loop iterations.

        Args:
            qpos (np.ndarray): Initial joint positions (configuration vector).
            oMdes (pin.SE3): Desired end-effector pose in SE3 (rotation and translation).
            wrist_id (int): Frame ID of the wrist (end-effector) in the Pinocchio model.
            max_iter (int, optional): Maximum number of iterations for convergence. Defaults to 1000.

        Returns:
            np.ndarray: Updated joint positions after iterative IK convergence.
        """
        # Update Pink configuration with initial qpos
        self.configuration.update(qpos)

        # Set the target transform for the wrist task (done once, as target is fixed)
        self.wrist_task.set_target(oMdes)

        # Iterate up to the specified number of iterations for convergence
        for _ in range(max_iter):
            # Compute velocity using Pink's solve_ik in the current configuration
            velocity = solve_ik(
                self.configuration,
                self.tasks,
                self.dt,
                solver=self.solver,
                damping=self.damp,
                limits=[self.config_limit],
                safety_break=True,  # Check for limit violations and raise if necessary
            )

            # Integrate velocity to get incremental update to qpos
            qpos = self.configuration.integrate(velocity, self.dt)

            # Update the configuration with the new qpos for the next iteration
            self.configuration.update(qpos)

            # Compute forward kinematics to check current error
            pin.forwardKinematics(self.model, self.data, qpos)
            wrist_pose = pin.updateFramePlacement(self.model, self.data, wrist_id)
            iMd = wrist_pose.actInv(oMdes)
            err = pin.log(iMd).vector

            # Check if the error is below the convergence threshold; if so, exit early
            if norm(err) < self.ik_eps:
                break

        # After loop, print a warning if not converged (for debugging)
        if norm(err) > self.ik_eps:
            print(f"motion_control.py: Not converged! Error norm: {norm(err)} (threshold: {self.ik_eps})")

        # Return the final qpos (no clipping needed, as limits are enforced in solve_ik)
        return qpos

    def ik_clik(
        self, qpos: np.ndarray, oMdes: pin.SE3, wrist_id: int, iter: int = 1000
    ) -> np.ndarray:
        """
        Compute inverse kinematics using a Closed-Loop Inverse Kinematics (CLIK) algorithm.
        - Iteratively compute joint velocities to minimize the error between the current and desired
        end-effector (wrist) pose.
        - Use Pinocchio for forward kinematics and Jacobian computation.
        - Apply damping to stabilize the solution.
        - Clip the resulting joint positions to stay within joint limits.

        Args:
            qpos (np.ndarray): Current joint positions (configuration vector).
            oMdes (pin.SE3): Desired end-effector pose in SE3 (rotation and translation).
            wrist_id (int): Frame ID of the wrist (end-effector) in the Pinocchio model.
            iter (int, optional): Maximum number of iterations for the CLIK algorithm. Defaults to 1000.

        Returns:
            np.ndarray: Updated joint positions after applying the CLIK algorithm.
        """
        # Iterate up to the specified number of iterations
        for _ in range(iter):
            # Compute forward kinematics to update the current end-effector pose
            pin.forwardKinematics(self.model, self.data, qpos)

            # Update the wrist (end-effector) pose in the world frame
            wrist_pose = pin.updateFramePlacement(self.model, self.data, wrist_id)

            # Compute the pose error (difference between current and desired pose) in SE3
            iMd = wrist_pose.actInv(oMdes)  # Transform from current to desired pose

            # Convert the pose error to a 6D vector (3D translation + 3D rotation) using logarithmic map
            err = pin.log(iMd).vector

            # Check if the error is below the convergence threshold; if so, exit early
            if norm(err) < self.ik_eps:
                break

            # Compute the frame Jacobian for the wrist in the world frame
            J = pin.computeFrameJacobian(self.model, self.data, qpos, wrist_id)

            # Transform the Jacobian to account for the pose error's frame using the logarithmic map
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            # Solve for the joint velocity using damped least-squares (pseudo-inverse with damping)
            # J^T * (J * J^T + damp * I)^(-1) * err
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))

            # Integrate the joint velocity to update the joint positions
            qpos = pin.integrate(self.model, qpos, v * self.dt)

        if norm(err) > self.ik_eps:
            print(f"motion_control.py: Not converged! Error norm: {norm(err)} (threshold: {self.ik_eps})")

        # Clip the joint positions to ensure they stay within the model's joint limits
        qpos = np.clip(qpos, self.lower_limit, self.upper_limit)

        # Return the updated joint positions
        return qpos