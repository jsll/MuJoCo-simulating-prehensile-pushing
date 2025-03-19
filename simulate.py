import argparse

import mujoco
import mujoco.viewer
import numpy as np
import utils
import yaml


class DifferentialIKController:
    def __init__(self, model, data, end_effector_id, damping=1e-5, max_angvel=0.1):
        self.model = model
        self.data = data
        self.end_effector_id = end_effector_id
        self.jac = np.zeros((6, model.nv))
        self.diag = damping * np.eye(6)
        self.max_angvel = max_angvel

    def compute_control(self, spatial_velocity):
        self.update()
        # Compute IK
        dq = self.differential_ik(spatial_velocity)

        return dq

    def update(self):
        # Get Jacobian
        mujoco.mj_jacSite(
            self.model, self.data, self.jac[:3], self.jac[3:], self.end_effector_id
        )

    def differential_ik(
        self,
        spatial_velocity,
    ):
        dq = self.jac.T @ np.linalg.solve(
            self.jac @ self.jac.T + self.diag, spatial_velocity
        )

        # Scale down joint velocities if they exceed maximum.
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

        return dq


class JointPositionController:
    def __init__(self, max_angvel, integration_dt):
        self.max_angvel = max_angvel
        self.integration_dt = integration_dt

    def compute_control(self, joint_velocity, current_q):
        # Scale down the joint velocities if they exceed maximum
        max_step = self.max_angvel * self.integration_dt
        scale_factor = 1.0
        max_change = np.abs(joint_velocity).max()
        if max_change > max_step:
            scale_factor = max_step / max_change
        # Apply scaled changes to get new positions
        dq = scale_factor * joint_velocity

        q = current_q + dq

        return q


class StateMachine:
    def __init__(
        self,
        model,
        data,
        end_effector_id,
        pick_site_id,
        gripper_actuator_ids,
        sequential_push_types,
        dof_ids,
    ):
        self.state = "init"
        self.model = model
        self.data = data
        self.end_effector_id = end_effector_id
        self.pick_site_id = pick_site_id
        self.gripper_actuator_ids = gripper_actuator_ids
        self.sequential_push_types = sequential_push_types  # If translational we align the pre-push pose with push_pose_4, if rotational we align with push_pose_2
        self.dof_ids = dof_ids

        # Add push pose site IDs
        self.fixture_pose_id = model.site("fixture_pose").id
        self.current_prehensile_pose_id = 0
        self.push_type_to_push_pose_id = {
            "translational": model.site("push_pose_4").id,
            "rotational": model.site("push_pose_2").id,
        }

        # State completion thresholds
        self.position_threshold = 0.008  # Distance threshold
        self.position_threshold_pushing = 0.017  # Distance threshold
        self.orientation_threshold = 0.01  # Quaternion distance threshold
        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = 0.0

        # Store initial pose
        self.initial_pick_pos = np.zeros(3)
        self.initial_pick_quat = np.zeros(4)

        # Initialization and grasping steps, used to wait for the simulation to stabilize from initialization and to wait for the gripper to close
        self.steps_for_initialization = 50
        self.current_initialization_step = 0
        self.wait_for_gripper = 100
        self.current_gripper_step = 0
        self.default_qpos_at_fixture = np.array(
            [
                1.0463343,
                -0.76788506,
                0.17391249,
                -1.68005005,
                0.14997465,
                0.92053087,
                -2.66448956,
            ]
        )

        # Offsets for pre-grasping, post-grasping, and returning from push
        self.pre_grasp_offset = np.array([0.0, 0.0, 0.1])
        self.post_grasp_offset = self.pre_grasp_offset
        self.offset_to_return_from_push = {
            "translational": np.array([0.0, -0.1, 0.0]),
            "rotational": np.array([0.0, 0.0, -0.1]),
        }

        # Temporary quaternion arrays for computations
        self.site_quat = np.zeros(4)
        self.target_quat = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)

        # Temporary variables for storing
        self.position_to_return_from = None

    def update_mocap(self, target_pos, target_quat):
        self.data.mocap_pos = target_pos
        self.data.mocap_quat = target_quat

    def update(self):
        if self.state == "init":
            self.current_initialization_step += 1
            # Store initial pose
            if self.current_initialization_step == self.steps_for_initialization:
                self.initial_pick_pos[:] = self.data.site(self.pick_site_id).xpos.copy()
                mujoco.mju_mat2Quat(
                    self.initial_pick_quat,
                    self.data.site(self.pick_site_id).xmat.copy(),
                )
                self.update_mocap(self.initial_pick_pos, self.initial_pick_quat)
                self.current_initialization_step = 0
                self.state = "move_to_pregrasp"
            return None

        elif self.state == "move_to_pregrasp":
            target_pos = (
                self.data.site(self.pick_site_id).xpos.copy() + self.pre_grasp_offset
            )

            mujoco.mju_mat2Quat(
                self.target_quat, self.data.site(self.pick_site_id).xmat
            )
            if self.check_pose_reached(target_pos, self.target_quat):
                self.state = "move_to_pick"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.spatial_velocity_from_poses(target_pos, self.target_quat)

        elif self.state == "move_to_pick":
            target_pos = self.data.site(self.pick_site_id).xpos.copy()
            mujoco.mju_mat2Quat(
                self.target_quat, self.data.site(self.pick_site_id).xmat
            )

            if self.check_pose_reached(target_pos, self.target_quat):
                self.state = "close_gripper"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.spatial_velocity_from_poses(target_pos, self.target_quat)

        elif self.state == "close_gripper":
            self.data.ctrl[self.gripper_actuator_ids] = 50.0
            self.current_gripper_step += 1
            if self.wait_for_gripper == self.current_gripper_step:
                self.current_gripper_step = 0
                self.state = "move_to_post_grasp_with_object"
            return None

        elif self.state == "move_to_post_grasp_with_object":
            target_pos = self.initial_pick_pos + self.post_grasp_offset

            self.target_quat = self.initial_pick_quat
            if self.check_pose_reached(target_pos, self.target_quat):
                self.state = "move_to_fixture"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.spatial_velocity_from_poses(target_pos, self.target_quat)

        elif self.state == "move_to_fixture":
            target_pos = self.data.site(self.fixture_pose_id).xpos.copy()
            mujoco.mju_mat2Quat(
                self.target_quat, self.data.site(self.fixture_pose_id).xmat
            )

            if self.check_pose_reached(target_pos, self.target_quat):
                self.state = "align_with_push_pose"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.joint_velocities_from_joint_poitions(
                    self.default_qpos_at_fixture
                )

        elif self.state == "align_with_push_pose":
            pre_push_site_id = self.model.site(
                f"pre_prehensile_pose_{self.current_prehensile_pose_id}"
            ).id
            push_type = self.sequential_push_types[self.current_prehensile_pose_id]
            target_pos_id = self.push_type_to_push_pose_id[push_type]
            target_pos = self.data.site(target_pos_id).xpos.copy()

            pre_push_pos = self.data.site(pre_push_site_id).xpos
            offset = target_pos - pre_push_pos

            current_ee_pos = self.data.site(self.end_effector_id).xpos
            target_ee_pos = current_ee_pos + offset

            # I want to compute T x O* x G, where T is the target quaternion, O is the quaternion of the pre-push pose, and G is the quaternion of the end-effector
            mujoco.mju_mat2Quat(self.target_quat, self.data.site(target_pos_id).xmat)

            pre_push_quat = np.zeros(4)
            pre_push_conj_quat = np.zeros(4)
            mujoco.mju_mat2Quat(pre_push_quat, self.data.site(pre_push_site_id).xmat)
            mujoco.mju_negQuat(pre_push_conj_quat, pre_push_quat)

            end_effector_quat = np.zeros(4)
            mujoco.mju_mat2Quat(
                end_effector_quat, self.data.site(self.end_effector_id).xmat
            )

            mujoco.mju_mulQuat(self.target_quat, self.target_quat, pre_push_conj_quat)
            mujoco.mju_mulQuat(self.target_quat, self.target_quat, end_effector_quat)

            if self.check_pose_reached(
                target_ee_pos,
                self.target_quat,
            ):
                self.position_to_return_from = self.data.site(
                    self.end_effector_id
                ).xpos.copy()
                self.state = "execute_push"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.spatial_velocity_from_poses(target_ee_pos, self.target_quat)

        elif self.state == "execute_push":
            goal_frame = f"prehensile_pose_{self.current_prehensile_pose_id + 1}"
            goal_site_id = self.model.site(goal_frame).id
            target_pos = self.data.site(goal_site_id).xpos.copy()
            mujoco.mju_mat2Quat(self.target_quat, self.data.site(goal_site_id).xmat)
            if self.check_pose_reached(
                target_pos, self.target_quat, self.position_threshold_pushing
            ):
                self.position_to_return_from = self.data.site(
                    self.end_effector_id
                ).xpos.copy()
                self.state = "return_from_push"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.spatial_velocity_from_poses(target_pos, self.target_quat)

        elif self.state == "return_from_push":
            push_type = self.sequential_push_types[self.current_prehensile_pose_id]
            target_frame_id = self.push_type_to_push_pose_id[push_type]
            # Get the rotation matrix of the target pose
            target_rot = self.data.site(target_frame_id).xmat.reshape(3, 3)
            # Transform the offset to the local frame
            local_offset = target_rot @ self.offset_to_return_from_push[push_type]

            target_pos = self.position_to_return_from + local_offset
            # The target quaternion is the same as the current gripper quaternion
            mujoco.mju_mat2Quat(
                self.target_quat, self.data.site(self.end_effector_id).xmat
            )
            if self.check_pose_reached(target_pos, target_quat=None):
                if self.current_prehensile_pose_id == (
                    len(self.sequential_push_types) - 1
                ):
                    self.state = "done"
                else:
                    self.current_prehensile_pose_id += 1
                    self.state = "move_to_fixture"
                return None
            else:
                self.update_mocap(target_pos, self.target_quat)
                return self.spatial_velocity_from_poses(target_pos, self.target_quat)

        elif self.state == "done":
            return None

    def check_pose_reached(
        self, target_pos=None, target_quat=None, position_threshold=None
    ):
        if target_pos is None and target_quat is None:
            raise ValueError("At least one of target_pos or target_quat must be set.")

        position_reached = False
        orientation_reached = False
        if target_pos is not None:
            current_pos = self.data.site(self.end_effector_id).xpos
            if position_threshold is None:
                position_threshold = self.position_threshold
            position_reached = (
                np.linalg.norm(current_pos - target_pos) < position_threshold
            )

        if target_quat is not None:
            # Get current orientation
            mujoco.mju_mat2Quat(
                self.site_quat, self.data.site(self.end_effector_id).xmat
            )
            # Compute quaternion difference
            mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
            mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
            orientation_error = np.abs(1.0 - self.error_quat[0])  # cos(theta/2)
            orientation_reached = orientation_error < self.orientation_threshold

        if target_pos is not None and target_quat is not None:
            return position_reached and orientation_reached
        elif target_pos is not None:
            return position_reached
        else:
            return orientation_reached

    def spatial_velocity_from_poses(self, target_pos, target_quat):
        # Compute spatial velocity
        current_pos = self.data.site(self.end_effector_id).xpos
        current_quat = np.zeros(4)
        rotational_velocity = np.zeros(3)
        mujoco.mju_mat2Quat(current_quat, self.data.site(self.end_effector_id).xmat)

        # Position error
        translational_velocity = target_pos - current_pos

        # Orientation error
        mujoco.mju_negQuat(self.site_quat_conj, current_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(rotational_velocity, self.error_quat, 1.0)

        spatial_velocity = np.zeros(6)
        spatial_velocity[:3] = translational_velocity
        spatial_velocity[3:] = rotational_velocity

        return spatial_velocity

    def joint_velocities_from_joint_poitions(self, target_q):
        current_q = self.data.qpos[self.dof_ids][:7]
        dq = target_q - current_q
        return dq


class Simulator:
    def __init__(
        self,
        experiment_file,
        dt,
        integration_dt,
        max_velocity,
        gravity_compensation,
        damping,
    ):
        self.model = None
        self.data = None
        self.integration_dt = integration_dt
        self.max_velocity = max_velocity
        self.damping = damping

        self.setup(experiment_file, gravity_compensation, dt)

    def setup(self, experiment_file, gravity_compensation, dt):
        mujoco_xml, sequential_push_types = self.setup_experiment(experiment_file)
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        self.model.opt.timestep = dt

        self.data = mujoco.MjData(self.model)

        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        actuator_names = [
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
        ]

        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array(
            [self.model.actuator(name).id for name in actuator_names]
        )

        gripper_actuator_ids = np.array([self.model.actuator("actuator8").id])

        # Get necessary IDs
        self.end_effector_id = self.model.site("attachment_site").id

        # Get joint ranges
        self.joint_ranges = np.array(
            [self.model.jnt_range[jnt_id] for jnt_id in self.dof_ids]
        )

        # This actually does not do anything unless we enable gravity compensation in the xml file which is weird
        if gravity_compensation:
            self.model.body_gravcomp[self.dof_ids] = 1.0

        pick_site_id = self.model.site("prehensile_pose_0").id

        # Get keyframe ID and mocap ID. The mocap ID is used to set the target pose for the end-effector
        self.key_id = self.model.key("home").id
        self.mocap_id = self.model.body("target").mocapid[0]

        # Initialize state machine
        self.state_machine = StateMachine(
            self.model,
            self.data,
            self.end_effector_id,
            pick_site_id,
            gripper_actuator_ids,
            sequential_push_types,
            self.dof_ids,
        )
        # Initialize controller
        self.differential_ik_controller = DifferentialIKController(
            self.model, self.data, self.end_effector_id, self.damping, self.max_velocity
        )
        self.joint_position_controller = JointPositionController(
            self.max_velocity, self.integration_dt
        )

    def setup_experiment(self, experiment_file: str):
        # Load experiment file
        with open(experiment_file, "r") as file:
            experiment = yaml.safe_load(file)

        object_type = experiment["object"]
        mesh_file = experiment["mesh_file"]
        trajectory_file = experiment["trajectory_file"]
        mesh = utils.load_mesh("models/" + mesh_file)
        prehensile_poses = np.load("trajectories/" + trajectory_file)
        pre_prehensile_poses = []
        push_types = []  # There are two push types: translational and rotational
        for start_pose, goal_pose in zip(prehensile_poses, prehensile_poses[1:]):
            contact_pose, push_type = utils.calculate_contact_pose(
                mesh, start_pose, goal_pose
            )
            pre_prehensile_poses.append(contact_pose)
            push_types.append(push_type)

        mujoco_xml = self.generate_model_specification(
            object_type, prehensile_poses, pre_prehensile_poses
        )
        return mujoco_xml, push_types

    def get_object_specification(self, sites, object_type="cube"):
        object_specification = ""
        if object_type == "cube":
            object_specification = f"""
            <body name="cube" pos="0.5 0 0.2" >
                <joint type="free"/>
                <geom type="box" size="0.01 0.03 0.03" rgba="1 0 0 1" 
                      friction="1.05 0.05 0.1" 
                      condim="6" solimp="1 0.5 0.2" solref="0.001 1"/>
                {sites}
            </body>
            </worldbody>
            <keyframe>
                <key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04 0.5 0 0.2 1 0 0 0" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>
            </keyframe>
            """
        elif object_type == "T-shape":
            object_specification = f"""<body name="t_shape" pos="0.5 0 0.2">
                    <joint type="free"/>
                    <!-- Vertical part of T -->
                    <geom name="t_vertical" type="box" pos="0 0 -0.015" size="0.01 0.015 0.03" rgba="1 0 0 1"
                          friction="1.05 0.05 0.1"
                          condim="6" solimp="1 0.5 0.2" solref="0.001 1"/>
                    <!-- Horizontal part of T -->
                    <geom name="t_horizontal" type="box" pos="0 0 0.015" size="0.01 0.03 0.015" rgba="1 0 0 1"
                          friction="1.05 0.05 0.1"
                          condim="6" solimp="1 0.5 0.2" solref="0.001 1"/>
 
                    <!-- Add your sites here as needed -->
                    {sites}
                </body>
                </worldbody>
                <keyframe>
                    <key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04 0.5 0 0.2 1 0 0 0" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>
                </keyframe>
                """
        elif object_type == "L-shape":
            object_specification = f"""
                <body name="l_shape" pos="0.5 0 0.2" quat="0.3826834 0.9238795 0 0">
                    <joint type="free"/>
                    <!-- Vertical part of L -->
                    <geom name="l_vertical" type="box" pos="0 0.015 0" size="0.01 0.015 0.03" rgba="1 0 0 1"
                          friction="1.05 0.05 0.1"
                          condim="6" solimp="1 0.5 0.2" solref="0.001 1"/>
                    <!-- Horizontal part of L -->
                    <geom name="l_horizontal" type="box" pos="0 0 -0.015" size="0.01 0.03 0.015" rgba="1 0 0 1"
                          friction="1.05 0.05 0.1"
                          condim="6" solimp="1 0.5 0.2" solref="0.001 1"/>
                    {sites}
                </body>
                </worldbody>
                <keyframe>
                    <key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04 0.5 0 0.2 0.3826834 0.9238795 0 0" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>
                </keyframe>
            """

        else:
            raise ValueError(f"Object type {object_type} is not supported.")
        return object_specification

    def generate_model_specification(
        self, object_type, prehensile_poses, pre_prehensile_poses
    ):
        """
        Generate MuJoCo XML model specification with mesh and poses.

        Args:
            mesh_file (str): Path to the mesh file
            prehensile_poses (List[np.ndarray]): List of 4x4 pose matrices
            pre_prehensile_poses (List[np.ndarray]): List of 4x4 pre-prehensile pose matrices

        Returns:
            str: Complete MuJoCo XML model specification
        """
        # Build prehensile pose specifications
        prehensile_sites = []
        for i, pose in enumerate(prehensile_poses):
            pos = pose[:3, 3]  # Extract position from pose matrix
            rot = pose[:3, :3]  # Extract rotation matrix
            quat = utils.rotmat2quat(rot)  # Convert rotation matrix to quaternion
            site = (
                f'<site name="prehensile_pose_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" '
                f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
                f'type="sphere" size="0.005" rgba="0 1 0 1"/>'
            )
            prehensile_sites.append(site)

        # Build pre-prehensile pose specifications
        pre_prehensile_sites = []
        for i, pose in enumerate(pre_prehensile_poses):
            pos = pose[:3, 3]
            rot = pose[:3, :3]
            quat = utils.rotmat2quat(rot)
            site = (
                f'<site name="pre_prehensile_pose_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" '
                f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" '
                f'type="sphere" size="0.005" rgba="0 1 0 1"/>'
            )
            pre_prehensile_sites.append(site)

        # Combine all site specifications
        all_sites = "\n ".join(prehensile_sites + pre_prehensile_sites)

        object_specification = self.get_object_specification(all_sites, object_type)

        xml = f"""
        <mujoco model="panda scene">

            <option integrator="implicitfast">
                <flag multiccd="enable"/>
            </option>
            <compiler angle="radian" meshdir="models/franka_emika_panda/assets" autolimits="true"/>
            <include file="models/franka_emika_panda/scene.xml"/>
            <worldbody>
            <!-- Added fixed fixture -->
            <body name="fixture" pos="0.0 0.4 0.5">
                <geom type="box" size="0.2 0.048 0.02" rgba="0 0 1 1" friction="2 0.9 0.9"/>
                <site name="push_pose_2" pos="0 0 0.02" quat="0 0 -1 0" type="sphere" size="0.005" rgba="0 1 0 1"/>
                <site name="push_pose_4" pos="0 -0.048 0" quat="0 0 -1 0" type="sphere" size="0.005" rgba="0 1 0 1"/>
                <site name="fixture_pose" pos="0 -0.2 0.2" quat="0 0 -1 0" type="sphere" size="0.005" rgba="0 1 0 1"/>
            </body>
            {object_specification}
        </mujoco>
        """

        return xml

    def simulate(self, apply_control):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            # Reset to initial keyframe
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                if apply_control:
                    self.update()
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def update(self):
        velocity = self.state_machine.update()
        q = self.data.qpos.copy()
        # If the spatial_velocity is None, we set the control to the current joint positions
        if velocity is None:
            q_robot = q[self.dof_ids]
        # If the velocity is of size 7, it means we have joint velocities
        elif velocity.size == 7:
            current_q = q[self.dof_ids][:7]
            q_robot = self.joint_position_controller.compute_control(
                velocity, current_q
            )
        # If the velocity is of size 6, it means we have spatial (end-effector) velocities
        else:
            dq = self.differential_ik_controller.compute_control(velocity)
            # Integrate velocities
            mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
            # Apply joint limits and set control
            q_robot = q[self.dof_ids]
            np.clip(
                q_robot,
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
                out=q_robot,
            )

        self.data.ctrl[self.actuator_ids] = q_robot[:7]


def main(args) -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    simulator = Simulator(
        args.filename,
        args.dt,
        args.integration_dt,
        args.max_velocity,
        not args.no_gravity_compensation,
        args.damping,
    )
    simulator.simulate(not args.no_control)


if __name__ == "__main__":
    # Create an argparser that takes a filename as one of the arguments.
    parser = argparse.ArgumentParser(
        description="Run the prehensile manipulation simulation."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="experiments/square.yaml",
        help="Path to the experiment file.",
    )
    # Add max velocity argument
    parser.add_argument(
        "--max_velocity",
        type=float,
        default=0.1,
        help="Maximum allowed joint velocity in radians/second.",
    )
    # Add argument for no control
    parser.add_argument(
        "--no_control",
        action="store_true",
        help="Run the simulation without any control.",
    )
    # Add argument for damping
    parser.add_argument(
        "--damping",
        type=float,
        default=1e-5,
        help="Damping term for the pseudoinverse. This is used to prevent joint velocities from becoming too large when the Jacobian is close to singular.",
    )
    # Add argument for integration timestep
    parser.add_argument(
        "--integration_dt",
        type=float,
        default=1.0,
        help="Integration time-step in seconds. This corresponds to the amount of time the joint velocities will be integrated for to obtain the desired joint positions.",
    )
    # Add argument for gravity compensation
    parser.add_argument(
        "--no_gravity_compensation",
        action="store_true",
        help="Disable gravity compensation.",
    )
    # Add argument for dt
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Simulation time-step in seconds.",
    )

    args = parser.parse_args()

    main(args)
