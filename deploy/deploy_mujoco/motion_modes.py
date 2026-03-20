from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


MOTION_MODE_WALK = "WALK"
MOTION_MODE_STAND_BALANCE = "STAND_BALANCE"
MOTION_MODE_MANIPULATE_STANDING = "MANIPULATE_STANDING"
MOTION_MODE_WALK_CARRY = "WALK_CARRY"

MOTION_MODE_STATUS_TRANSITIONING = "TRANSITIONING"
MOTION_MODE_STATUS_READY = "READY"
MOTION_MODE_STATUS_ERROR = "ERROR"

VALID_MOTION_MODES = {
    MOTION_MODE_WALK,
    MOTION_MODE_STAND_BALANCE,
    MOTION_MODE_MANIPULATE_STANDING,
    MOTION_MODE_WALK_CARRY,
}


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def clamp_to_joint_ranges(target: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    clipped = target.copy()
    for idx in range(clipped.shape[0]):
        lo = float(ranges[idx, 0])
        hi = float(ranges[idx, 1])
        if lo < hi:
            clipped[idx] = np.clip(clipped[idx], lo, hi)
    return clipped


def rate_limit_towards(
    current: np.ndarray,
    goal: np.ndarray,
    rate_limits: np.ndarray,
    dt: float,
    *,
    ranges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    safe_dt = max(float(dt), 1e-6)
    max_step = np.asarray(rate_limits, dtype=np.float64) * safe_dt
    delta = np.clip(np.asarray(goal, dtype=np.float64) - np.asarray(current, dtype=np.float64), -max_step, max_step)
    next_target = np.asarray(current, dtype=np.float64) + delta
    if ranges is not None:
        next_target = clamp_to_joint_ranges(next_target, ranges)
        delta = next_target - np.asarray(current, dtype=np.float64)
    return next_target, delta / safe_dt


def quat_wxyz_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.array(quat_wxyz, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    quat /= norm
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_error(desired_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    rot_err = desired_rot @ current_rot.T
    skew = np.array(
        [
            rot_err[2, 1] - rot_err[1, 2],
            rot_err[0, 2] - rot_err[2, 0],
            rot_err[1, 0] - rot_err[0, 1],
        ],
        dtype=np.float64,
    )
    return 0.5 * skew


def rotation_matrix_to_roll_pitch(rot: np.ndarray) -> tuple[float, float]:
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    pitch = math.atan2(float(-rot[2, 0]), float(np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2)))
    return roll, pitch


def resolve_upright_body_id(
    m: mujoco.MjModel,
    *,
    preferred_name: str = "torso_link",
    fallback_body_id: int | None = None,
) -> int:
    preferred_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, preferred_name)
    if preferred_body_id >= 0:
        return int(preferred_body_id)
    if fallback_body_id is not None and int(fallback_body_id) >= 0:
        return int(fallback_body_id)
    pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body_id >= 0:
        return int(pelvis_body_id)
    raise ValueError(f"Could not resolve upright body '{preferred_name}' or fallback pelvis body")


def damped_task_step(
    jacobian: np.ndarray,
    error: np.ndarray,
    nullspace: np.ndarray,
    *,
    gain: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    task_jacobian = jacobian @ nullspace
    if task_jacobian.size == 0 or not np.any(np.abs(task_jacobian) > 1e-9):
        return np.zeros(nullspace.shape[0], dtype=np.float64), nullspace
    damping_sq = float(damping * damping)
    lhs = task_jacobian @ task_jacobian.T + damping_sq * np.eye(task_jacobian.shape[0], dtype=np.float64)
    rhs = error * float(gain)
    y = np.linalg.solve(lhs, rhs)
    pinv = task_jacobian.T @ y
    dq = nullspace @ pinv
    projector = np.eye(nullspace.shape[0], dtype=np.float64) - task_jacobian.T @ np.linalg.solve(lhs, task_jacobian)
    return dq, nullspace @ projector


@dataclass
class MotionModeSnapshot:
    mode: str
    status: str
    changed: bool


class MotionModeManager:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, config: dict, ros_bridge=None):
        self.m = m
        self.d = d
        self.ros_bridge = ros_bridge
        self.base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.upright_body_id = resolve_upright_body_id(self.m, fallback_body_id=self.base_body_id)
        self.requested_mode = str(config.get("initial_motion_mode", MOTION_MODE_WALK)).strip().upper()
        if self.requested_mode not in VALID_MOTION_MODES:
            self.requested_mode = MOTION_MODE_WALK
        self.current_mode = self.requested_mode
        self.status = MOTION_MODE_STATUS_TRANSITIONING
        self._last_mode = self.current_mode
        self._last_command = None
        self._ready_since = None
        self._transition_started = 0.0
        self._idle_since = None
        self.transition_timeout_sec = float(config.get("mode_transition_timeout_sec", 6.0))
        self.ready_hold_sec = float(config.get("mode_ready_hold_sec", 0.35))
        self.walk_ready_delay_sec = float(config.get("walk_ready_delay_sec", 0.10))
        self.ready_linear_speed_mps = float(config.get("mode_ready_linear_speed_mps", 0.10))
        self.ready_angular_speed_rps = float(config.get("mode_ready_angular_speed_rps", 0.40))
        self.ready_roll_pitch_rad = float(config.get("mode_ready_roll_pitch_rad", 0.22))
        self.ready_vertical_speed_mps = float(config.get("mode_ready_vertical_speed_mps", 0.08))
        self.auto_stand_enabled = bool(config.get("auto_stand_enabled", True))
        self.auto_stand_cmd_linear_threshold = float(config.get("auto_stand_cmd_linear_threshold", 0.05))
        self.auto_stand_cmd_yaw_threshold = float(config.get("auto_stand_cmd_yaw_threshold", 0.12))
        self.auto_stand_dwell_sec = float(config.get("auto_stand_dwell_sec", 0.75))
        self.auto_stand_resume_linear_threshold = float(
            config.get("auto_stand_resume_linear_threshold", 0.08)
        )
        self.auto_stand_resume_yaw_threshold = float(
            config.get("auto_stand_resume_yaw_threshold", 0.18)
        )
        self._sync_bridge()

    def _sync_bridge(self):
        if self.ros_bridge is None:
            return
        self.ros_bridge.set_motion_mode_state(self.current_mode)
        self.ros_bridge.set_motion_mode_status(self.status)

    def _set_mode(self, mode: str, sim_time: float):
        desired = str(mode).strip().upper()
        if desired not in VALID_MOTION_MODES:
            self.status = MOTION_MODE_STATUS_ERROR
            self._sync_bridge()
            return
        if desired != self.current_mode:
            self._last_mode = self.current_mode
            self.current_mode = desired
        self.status = MOTION_MODE_STATUS_TRANSITIONING
        self._ready_since = None
        self._transition_started = float(sim_time)
        self._sync_bridge()

    def _set_requested_mode(self, mode: str, sim_time: float):
        desired = str(mode).strip().upper()
        if desired not in VALID_MOTION_MODES:
            self.status = MOTION_MODE_STATUS_ERROR
            self._sync_bridge()
            return
        self.requested_mode = desired
        self._idle_since = None
        self._set_mode(desired, sim_time)

    def consume_command(self, sim_time: float):
        if self.ros_bridge is None:
            return
        command = self.ros_bridge.get_motion_mode_command()
        if command is None or command == self._last_command:
            return
        self._last_command = command
        self._set_requested_mode(command, sim_time)

    def _command_norms(self, command_velocity: np.ndarray | None) -> tuple[float, float]:
        if command_velocity is None:
            return 0.0, 0.0
        command_velocity = np.asarray(command_velocity, dtype=np.float64)
        linear = float(np.linalg.norm(command_velocity[:2]))
        yaw = float(abs(command_velocity[2]))
        return linear, yaw

    def _update_auto_stand_mode(self, sim_time: float, command_velocity: np.ndarray | None):
        if not self.auto_stand_enabled or self.requested_mode not in (MOTION_MODE_WALK, MOTION_MODE_WALK_CARRY):
            self._idle_since = None
            return

        linear_cmd, yaw_cmd = self._command_norms(command_velocity)
        command_idle = (
            linear_cmd <= self.auto_stand_cmd_linear_threshold
            and yaw_cmd <= self.auto_stand_cmd_yaw_threshold
        )
        if command_idle:
            if self._idle_since is None:
                self._idle_since = float(sim_time)
        else:
            self._idle_since = None

        if (
            self.current_mode == MOTION_MODE_WALK
            and self._idle_since is not None
            and float(sim_time) - self._idle_since >= self.auto_stand_dwell_sec
        ):
            self._set_mode(MOTION_MODE_STAND_BALANCE, sim_time)
            return

        command_requests_walk = (
            linear_cmd >= self.auto_stand_resume_linear_threshold
            or yaw_cmd >= self.auto_stand_resume_yaw_threshold
        )
        if self.current_mode == MOTION_MODE_STAND_BALANCE and command_requests_walk:
            self._set_mode(self.requested_mode, sim_time)
            self._idle_since = None

    def _standing_ready(self) -> bool:
        upright_rot = self.d.xmat[self.upright_body_id].reshape(3, 3)
        roll, pitch = rotation_matrix_to_roll_pitch(upright_rot)
        upright_cvel = self.d.cvel[self.upright_body_id]
        pelvis_cvel = self.d.cvel[self.base_body_id]
        angular_speed = float(np.linalg.norm(upright_cvel[:3]))
        linear_speed = float(np.linalg.norm(pelvis_cvel[3:5]))
        vertical_speed = float(abs(pelvis_cvel[5]))
        return (
            abs(roll) <= self.ready_roll_pitch_rad
            and abs(pitch) <= self.ready_roll_pitch_rad
            and angular_speed <= self.ready_angular_speed_rps
            and linear_speed <= self.ready_linear_speed_mps
            and vertical_speed <= self.ready_vertical_speed_mps
        )

    def _walking_ready(self, sim_time: float) -> bool:
        return float(sim_time) - self._transition_started >= self.walk_ready_delay_sec

    def step(self, sim_time: float, command_velocity: np.ndarray | None = None) -> MotionModeSnapshot:
        previous_mode = self.current_mode
        self.consume_command(sim_time)
        self._update_auto_stand_mode(sim_time, command_velocity)
        if self.current_mode in (MOTION_MODE_WALK, MOTION_MODE_WALK_CARRY):
            ready = self._walking_ready(sim_time)
        else:
            ready = self._standing_ready()

        if ready:
            if self._ready_since is None:
                self._ready_since = float(sim_time)
            if float(sim_time) - self._ready_since >= self.ready_hold_sec:
                self.status = MOTION_MODE_STATUS_READY
        else:
            self._ready_since = None
            if float(sim_time) - self._transition_started > self.transition_timeout_sec:
                self.status = MOTION_MODE_STATUS_ERROR
            else:
                self.status = MOTION_MODE_STATUS_TRANSITIONING
        self._sync_bridge()
        return MotionModeSnapshot(
            mode=self.current_mode,
            status=self.status,
            changed=previous_mode != self.current_mode,
        )

    def should_use_walk_policy(self) -> bool:
        return self.current_mode in (MOTION_MODE_WALK, MOTION_MODE_WALK_CARRY)

    def stand_controller_active(self) -> bool:
        return self.current_mode in (MOTION_MODE_STAND_BALANCE, MOTION_MODE_MANIPULATE_STANDING)

    def allow_manipulation_ik(self) -> bool:
        return (
            self.current_mode == MOTION_MODE_MANIPULATE_STANDING
            and self.status == MOTION_MODE_STATUS_READY
        )


class WalkLegController:
    def __init__(
        self,
        *,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        qpos_adr: np.ndarray,
        qvel_adr: np.ndarray,
        pelvis_body_id: int,
        policy,
        default_angles: np.ndarray,
        action_scale: float,
        cmd_scale: np.ndarray,
        dof_pos_scale: float,
        dof_vel_scale: float,
        ang_vel_scale: float,
        control_decimation: int,
        num_actions: int,
        num_obs: int,
    ):
        self.m = m
        self.d = d
        self.qpos_adr = qpos_adr
        self.qvel_adr = qvel_adr
        self.pelvis_body_id = pelvis_body_id
        self.policy = policy
        self.default_angles = default_angles.copy()
        self.action_scale = float(action_scale)
        self.cmd_scale = np.array(cmd_scale, dtype=np.float64)
        self.dof_pos_scale = float(dof_pos_scale)
        self.dof_vel_scale = float(dof_vel_scale)
        self.ang_vel_scale = float(ang_vel_scale)
        self.control_decimation = int(control_decimation)
        self.num_actions = int(num_actions)
        self.num_obs = int(num_obs)
        self.counter = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

    def reset(self):
        self.counter = 0
        self.action.fill(0.0)
        self.target_dof_pos = self.default_angles.copy()
        self.obs.fill(0.0)

    def step(self, cmd_policy: np.ndarray):
        self.counter += 1
        if self.counter % self.control_decimation != 0:
            return self.target_dof_pos

        qj = self.d.qpos[self.qpos_adr].copy()
        dqj = self.d.qvel[self.qvel_adr].copy()
        quat = self.d.qpos[3:7]
        omega = self.d.qvel[3:6]

        qj = (qj - self.default_angles) * self.dof_pos_scale
        dqj = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * self.ang_vel_scale

        period = 0.8
        count = self.counter * float(self.m.opt.timestep)
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.obs[:3] = omega
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = cmd_policy * self.cmd_scale
        self.obs[9 : 9 + self.num_actions] = qj
        self.obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj
        self.obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
        self.obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array(
            [sin_phase, cos_phase],
            dtype=np.float32,
        )

        obs_tensor = np.asarray(self.obs, dtype=np.float32)[None, :]
        import torch

        self.action = self.policy(torch.from_numpy(obs_tensor)).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.action_scale + self.default_angles
        return self.target_dof_pos


class StandLegController:
    def __init__(
        self,
        *,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        qpos_adr: np.ndarray,
        qvel_adr: np.ndarray,
        act_joint_ids: np.ndarray,
        config: dict,
    ):
        self.m = m
        self.d = d
        self.qpos_adr = qpos_adr
        self.qvel_adr = qvel_adr
        self.act_joint_ids = act_joint_ids[: len(qpos_adr)]
        self.leg_joint_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) or ""
            for joint_id in self.act_joint_ids
        ]
        self.leg_joint_index = {name: index for index, name in enumerate(self.leg_joint_names)}
        self.joint_ranges = self.m.jnt_range[self.act_joint_ids].astype(np.float64).copy()
        self.base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.upright_body_id = resolve_upright_body_id(self.m, fallback_body_id=self.base_body_id)
        self.target = np.array(config.get("stand_leg_target_angles", []), dtype=np.float64)
        if self.target.shape != qpos_adr.shape:
            self.target = self.d.qpos[self.qpos_adr].copy()
        self.filtered_target = self.d.qpos[self.qpos_adr].copy()
        self.feedback_max_delta = float(config.get("stand_leg_max_delta", 0.025))
        self.joint_rate_limit = float(config.get("stand_leg_joint_rate_limit", 1.50))
        self.base_height_target = float(config.get("stand_base_height_target", 0.79))
        self.pitch_kp = float(config.get("stand_pitch_kp", 0.45))
        self.pitch_kd = float(config.get("stand_pitch_kd", 0.09))
        self.roll_kp = float(config.get("stand_roll_kp", 0.35))
        self.roll_kd = float(config.get("stand_roll_kd", 0.08))
        self.vx_kp = float(config.get("stand_vx_kp", 0.18))
        self.vy_kp = float(config.get("stand_vy_kp", 0.12))
        self.height_kp = float(config.get("stand_height_kp", 0.35))
        self.height_kd = float(config.get("stand_height_kd", 0.10))
        self.midfoot_offset_x = float(config.get("stand_midfoot_offset_x", 0.035))
        self.midfoot_offset_y = float(config.get("stand_midfoot_offset_y", 0.0))
        self.midfoot_target_x = float(config.get("stand_midfoot_target_x", -0.01))
        self.midfoot_target_y = float(config.get("stand_midfoot_target_y", 0.0))
        self.midfoot_pitch_kp = float(config.get("stand_midfoot_pitch_kp", 0.90))
        self.midfoot_pitch_kd = float(config.get("stand_midfoot_pitch_kd", 0.20))
        self.midfoot_roll_kp = float(config.get("stand_midfoot_roll_kp", 0.0))
        self.midfoot_roll_kd = float(config.get("stand_midfoot_roll_kd", 0.0))
        self.hip_pitch_balance_scale = float(config.get("stand_hip_pitch_balance_scale", 0.90))
        self.knee_pitch_balance_scale = float(config.get("stand_knee_pitch_balance_scale", 0.20))
        self.ankle_pitch_balance_scale = float(config.get("stand_ankle_pitch_balance_scale", 1.35))
        self.hip_roll_balance_scale = float(config.get("stand_hip_roll_balance_scale", 1.00))
        self.ankle_roll_balance_scale = float(config.get("stand_ankle_roll_balance_scale", 1.20))
        self.height_hip_pitch_scale = float(config.get("stand_height_hip_pitch_scale", 0.20))
        self.height_knee_scale = float(config.get("stand_height_knee_scale", 1.00))
        self.height_ankle_pitch_scale = float(config.get("stand_height_ankle_pitch_scale", 0.55))
        self.left_hip_pitch_idx = self._require_joint_index("left_hip_pitch_joint")
        self.left_hip_roll_idx = self._require_joint_index("left_hip_roll_joint")
        self.left_knee_idx = self._require_joint_index("left_knee_joint")
        self.left_ankle_pitch_idx = self._require_joint_index("left_ankle_pitch_joint")
        self.left_ankle_roll_idx = self._require_joint_index("left_ankle_roll_joint")
        self.right_hip_pitch_idx = self._require_joint_index("right_hip_pitch_joint")
        self.right_hip_roll_idx = self._require_joint_index("right_hip_roll_joint")
        self.right_knee_idx = self._require_joint_index("right_knee_joint")
        self.right_ankle_pitch_idx = self._require_joint_index("right_ankle_pitch_joint")
        self.right_ankle_roll_idx = self._require_joint_index("right_ankle_roll_joint")
        self.left_foot_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    def _require_joint_index(self, joint_name: str) -> int:
        if joint_name not in self.leg_joint_index:
            raise ValueError(f"StandLegController missing required leg joint: {joint_name}")
        return int(self.leg_joint_index[joint_name])

    def reset(self):
        self.filtered_target = self.d.qpos[self.qpos_adr].copy()

    def _midfoot_support_state(self) -> tuple[float, float, float, float]:
        pelvis_pos = self.d.xpos[self.base_body_id].copy()
        pelvis_vel_world = self.d.cvel[self.base_body_id][3:6].copy()
        left_rot = self.d.xmat[self.left_foot_body_id].reshape(3, 3)
        right_rot = self.d.xmat[self.right_foot_body_id].reshape(3, 3)
        left_midfoot = self.d.xpos[self.left_foot_body_id].copy() + left_rot @ np.array(
            [self.midfoot_offset_x, self.midfoot_offset_y, 0.0],
            dtype=np.float64,
        )
        right_midfoot = self.d.xpos[self.right_foot_body_id].copy() + right_rot @ np.array(
            [self.midfoot_offset_x, self.midfoot_offset_y, 0.0],
            dtype=np.float64,
        )
        support_midpoint = 0.5 * (left_midfoot + right_midfoot)
        support_forward = left_rot[:, 0] + right_rot[:, 0]
        support_forward[2] = 0.0
        norm = float(np.linalg.norm(support_forward))
        if norm < 1e-6:
            support_forward = self.d.xmat[self.base_body_id].reshape(3, 3)[:, 0].copy()
            support_forward[2] = 0.0
            norm = float(np.linalg.norm(support_forward))
        if norm < 1e-6:
            support_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            support_forward /= norm
        support_lateral = left_rot[:, 1] + right_rot[:, 1]
        support_lateral[2] = 0.0
        support_lateral = support_lateral - float(np.dot(support_lateral, support_forward)) * support_forward
        lat_norm = float(np.linalg.norm(support_lateral))
        if lat_norm < 1e-6:
            support_lateral = np.array([-support_forward[1], support_forward[0], 0.0], dtype=np.float64)
            lat_norm = float(np.linalg.norm(support_lateral))
        if lat_norm < 1e-6:
            support_lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            support_lateral /= lat_norm
        pelvis_offset_x = float(np.dot(pelvis_pos - support_midpoint, support_forward))
        pelvis_vel_x = float(np.dot(pelvis_vel_world, support_forward))
        pelvis_offset_y = float(np.dot(pelvis_pos - support_midpoint, support_lateral))
        pelvis_vel_y = float(np.dot(pelvis_vel_world, support_lateral))
        return pelvis_offset_x, pelvis_vel_x, pelvis_offset_y, pelvis_vel_y

    def compute_target(self) -> np.ndarray:
        target = self.target.copy()
        pelvis_rot = self.d.xmat[self.base_body_id].reshape(3, 3)
        pelvis_cvel = self.d.cvel[self.base_body_id]
        upright_rot = self.d.xmat[self.upright_body_id].reshape(3, 3)
        upright_cvel = self.d.cvel[self.upright_body_id]
        base_vel = pelvis_rot.T @ pelvis_cvel[3:6]
        roll, pitch = rotation_matrix_to_roll_pitch(upright_rot)
        roll_rate = float(upright_cvel[0])
        pitch_rate = float(upright_cvel[1])
        pelvis_z = float(self.d.xpos[self.base_body_id][2])
        z_vel = float(pelvis_cvel[5])
        pelvis_offset_x, pelvis_vel_x, pelvis_offset_y, pelvis_vel_y = self._midfoot_support_state()

        sagittal_cmd = (
            self.pitch_kp * pitch
            + self.pitch_kd * pitch_rate
            + self.vx_kp * float(base_vel[0])
            + self.midfoot_pitch_kp * (self.midfoot_target_x - pelvis_offset_x)
            - self.midfoot_pitch_kd * pelvis_vel_x
        )
        roll_cmd = (
            self.roll_kp * roll
            + self.roll_kd * roll_rate
            + self.vy_kp * float(base_vel[1])
            + self.midfoot_roll_kp * (self.midfoot_target_y - pelvis_offset_y)
            - self.midfoot_roll_kd * pelvis_vel_y
        )
        height_cmd = self.height_kp * (self.base_height_target - pelvis_z) - self.height_kd * z_vel

        hip_balance_cmd = self.hip_pitch_balance_scale * sagittal_cmd
        knee_balance_cmd = self.knee_pitch_balance_scale * sagittal_cmd
        ankle_balance_cmd = self.ankle_pitch_balance_scale * sagittal_cmd

        target[self.left_hip_pitch_idx] -= hip_balance_cmd
        target[self.left_knee_idx] += knee_balance_cmd
        target[self.left_ankle_pitch_idx] -= ankle_balance_cmd
        target[self.right_hip_pitch_idx] -= hip_balance_cmd
        target[self.right_knee_idx] += knee_balance_cmd
        target[self.right_ankle_pitch_idx] -= ankle_balance_cmd

        target[self.left_hip_pitch_idx] += self.height_hip_pitch_scale * height_cmd
        target[self.left_knee_idx] -= self.height_knee_scale * height_cmd
        target[self.left_ankle_pitch_idx] -= self.height_ankle_pitch_scale * height_cmd
        target[self.right_hip_pitch_idx] += self.height_hip_pitch_scale * height_cmd
        target[self.right_knee_idx] -= self.height_knee_scale * height_cmd
        target[self.right_ankle_pitch_idx] -= self.height_ankle_pitch_scale * height_cmd

        target[self.left_hip_roll_idx] += self.hip_roll_balance_scale * roll_cmd
        target[self.left_ankle_roll_idx] -= self.ankle_roll_balance_scale * roll_cmd
        target[self.right_hip_roll_idx] -= self.hip_roll_balance_scale * roll_cmd
        target[self.right_ankle_roll_idx] += self.ankle_roll_balance_scale * roll_cmd

        feedback_delta = np.clip(target - self.target, -self.feedback_max_delta, self.feedback_max_delta)
        desired_target = clamp_to_joint_ranges(self.target + feedback_delta, self.joint_ranges)
        rate_limits = np.full(self.filtered_target.shape[0], self.joint_rate_limit, dtype=np.float64)
        self.filtered_target, _ = rate_limit_towards(
            self.filtered_target,
            desired_target,
            rate_limits,
            float(self.m.opt.timestep),
            ranges=self.joint_ranges,
        )
        return self.filtered_target.copy()


class HierarchicalUpperBodyController:
    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        config: dict,
        motion_mode_manager: MotionModeManager,
        ros_bridge=None,
    ):
        self.m = m
        self.d = d
        self.ros_bridge = ros_bridge
        self.motion_mode_manager = motion_mode_manager
        self.enabled = bool(config.get("upper_body_enabled", False))
        self.leg_actuator_count = int(config.get("leg_actuator_count", 12))
        self.nu = int(self.m.nu)
        self.act_joint_ids = self.m.actuator_trnid[:, 0].copy()
        self.qpos_adr = np.array([self.m.jnt_qposadr[jid] for jid in self.act_joint_ids], dtype=int)
        self.qvel_adr = np.array([self.m.jnt_dofadr[jid] for jid in self.act_joint_ids], dtype=int)

        if not self.enabled or self.nu <= self.leg_actuator_count:
            self.enabled = False
            return

        actuator_name_to_index = {}
        self.actuator_names = []
        for actuator_id in range(self.nu):
            actuator_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if not actuator_name:
                raise ValueError(f"Actuator id {actuator_id} is missing a name")
            self.actuator_names.append(actuator_name)
            actuator_name_to_index[actuator_name] = actuator_id

        self.waist_joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        self.left_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        ]
        self.right_arm_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.right_hand_joint_names = [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.waist_ctrl_indices = resolve_ctrl_indices(actuator_name_to_index, self.waist_joint_names)
        self.left_ctrl_indices = resolve_ctrl_indices(actuator_name_to_index, self.left_joint_names)
        self.right_arm_ctrl_indices = resolve_ctrl_indices(actuator_name_to_index, self.right_arm_joint_names)
        self.right_hand_ctrl_indices = resolve_ctrl_indices(actuator_name_to_index, self.right_hand_joint_names)
        self.upper_ctrl_indices = np.concatenate(
            (
                self.waist_ctrl_indices,
                self.left_ctrl_indices,
                self.right_arm_ctrl_indices,
                self.right_hand_ctrl_indices,
            )
        ).astype(int)
        self.upper_qpos_adr = self.qpos_adr[self.upper_ctrl_indices]
        self.upper_qvel_adr = self.qvel_adr[self.upper_ctrl_indices]

        self.arm_ctrl_indices = np.concatenate((self.waist_ctrl_indices, self.right_arm_ctrl_indices)).astype(int)
        self.arm_qpos_adr = self.qpos_adr[self.arm_ctrl_indices]
        self.arm_qvel_adr = self.qvel_adr[self.arm_ctrl_indices]
        self.arm_joint_ids = self.act_joint_ids[self.arm_ctrl_indices]
        self.arm_joint_ranges = self.m.jnt_range[self.arm_joint_ids].astype(np.float64).copy()
        self.left_joint_ids = self.act_joint_ids[self.left_ctrl_indices]
        self.left_joint_ranges = self.m.jnt_range[self.left_joint_ids].astype(np.float64).copy()
        self.right_hand_qpos_adr = self.qpos_adr[self.right_hand_ctrl_indices]

        self.waist_slice = slice(0, 3)
        self.left_slice = slice(3, 17)
        self.right_arm_slice = slice(17, 24)
        self.right_hand_slice = slice(24, 31)

        self.upper_kps = np.array(config["upper_body_kps"], dtype=np.float64)
        self.upper_kds = np.array(config["upper_body_kds"], dtype=np.float64)
        self.zero_upper_dq = np.zeros_like(self.upper_kds)

        self.waist_neutral = np.array(config["waist_neutral_pose"], dtype=np.float64)
        self.left_arm_park = np.array(config["left_arm_park_pose"], dtype=np.float64)
        self.right_arm_neutral = np.array(config["right_arm_neutral_pose"], dtype=np.float64)
        self.right_arm_stow = np.array(config["right_arm_stow_pose"], dtype=np.float64)
        self.right_arm_carry = np.array(config["right_arm_carry_pose"], dtype=np.float64)
        self.right_arm_stand = np.array(config.get("right_arm_stand_pose", config["right_arm_neutral_pose"]), dtype=np.float64)
        self.right_hand_open = np.array(config["right_hand_open"], dtype=np.float64)
        self.right_hand_close = np.array(config["right_hand_close"], dtype=np.float64)

        self.hand_interp_rate = float(config.get("hand_interp_rate", 8.0))
        self.hand_state_tolerance = float(config.get("hand_state_tolerance", 0.06))
        self.arm_goal_timeout_sec = float(config.get("arm_goal_timeout_sec", 5.0))
        self.arm_goal_tolerance = float(config.get("arm_goal_tolerance", 0.03))
        self.arm_task_gain = float(config.get("arm_task_gain", 0.45))
        self.arm_task_damping = float(config.get("arm_task_damping", 0.10))
        self.torso_task_gain = float(config.get("torso_task_gain", 0.25))
        self.torso_task_damping = float(config.get("torso_task_damping", 0.08))
        self.posture_task_gain = float(config.get("posture_task_gain", 0.18))
        self.posture_task_damping = float(config.get("posture_task_damping", 0.10))
        self.arm_joint_rate_limit = float(config.get("arm_joint_rate_limit", 1.25))
        self.waist_joint_rate_limit = float(config.get("waist_joint_rate_limit", 0.8))
        self.walk_arm_joint_rate_limit = float(config.get("walk_arm_joint_rate_limit", 0.55))
        self.stand_arm_joint_rate_limit = float(config.get("stand_arm_joint_rate_limit", 0.85))
        self.left_arm_joint_rate_limit = float(config.get("left_arm_joint_rate_limit", 0.60))
        self.walk_upper_gain_scale = float(config.get("walk_upper_gain_scale", 0.45))
        self.walk_carry_upper_gain_scale = float(config.get("walk_carry_upper_gain_scale", 0.55))
        self.stand_upper_gain_scale = float(config.get("stand_upper_gain_scale", 0.80))
        self.manip_upper_gain_scale = float(config.get("manip_upper_gain_scale", 1.0))
        self.walk_waist_gain_scale = float(config.get("walk_waist_gain_scale", 0.20))
        self.walk_carry_waist_gain_scale = float(config.get("walk_carry_waist_gain_scale", 0.25))
        self.stand_waist_gain_scale = float(config.get("stand_waist_gain_scale", 0.18))
        self.manip_waist_gain_scale = float(config.get("manip_waist_gain_scale", 0.45))
        self.palm_offset_local = np.array(config["palm_offset_local"], dtype=np.float64)
        self.workspace_min = np.array(config["arm_workspace_min"], dtype=np.float64)
        self.workspace_max = np.array(config["arm_workspace_max"], dtype=np.float64)
        self.manip_orientation_weight = float(config.get("manip_orientation_weight", 0.35))
        self.grasp_capture_halfsize = np.array(config["grasp_capture_halfsize"], dtype=np.float64)
        self.grasp_contact_frames_required = int(config.get("grasp_contact_frames", 6))
        self.grasp_loss_frames_allowed = int(config.get("grasp_loss_frames", 4))

        self.upper_target = np.zeros(self.upper_ctrl_indices.size, dtype=np.float64)
        self.upper_target_dq = np.zeros(self.upper_ctrl_indices.size, dtype=np.float64)
        self.arm_joint_goal = np.concatenate((self.waist_neutral, self.right_arm_stow))
        self.arm_joint_target = self.arm_joint_goal.copy()
        self.left_arm_target = self.left_arm_park.copy()
        self.right_hand_target = self.right_hand_open.copy()
        self.right_hand_target_dq = np.zeros_like(self.right_hand_target)
        self.right_hand_desired = self.right_hand_open.copy()
        self.hand_command = "open"
        self.hand_state = "OPEN"
        self.hand_command_seq = -1
        self.arm_goal_seq = -1
        self.arm_goal_base = None
        self.arm_goal_orientation_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.arm_goal_active = False
        self.arm_goal_started = 0.0
        self.arm_status = "IDLE"
        self.grasped_object_label = ""
        self.grasp_contact_frames = 0
        self.grasp_loss_frames = 0

        self.base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.wrist_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "small_sphere__link")
        if self.ball_body_id < 0:
            self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "small_sphere")

        self.ball_geom_ids = set()
        self.hand_geom_ids = set()
        self.geom_body_names = {}
        for gid in range(self.m.ngeom):
            body_id = int(self.m.geom_bodyid[gid])
            body_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            self.geom_body_names[gid] = body_name
            if body_name.startswith("small_sphere"):
                self.ball_geom_ids.add(gid)
            if (
                body_name == "right_wrist_yaw_link"
                or body_name.startswith("right_hand_thumb")
                or body_name.startswith("right_hand_index")
                or body_name.startswith("right_hand_middle")
            ):
                self.hand_geom_ids.add(gid)

        self._refresh_upper_target(0.0)
        self._sync_bridge_state()

    def _sync_bridge_state(self):
        if self.ros_bridge is None:
            return
        self.ros_bridge.set_arm_status(self.arm_status)
        self.ros_bridge.set_hand_state(self.hand_state)
        self.ros_bridge.set_grasped_object_label(self.grasped_object_label)

    def _consume_ros_commands(self, sim_time: float):
        if self.ros_bridge is None:
            return
        command_state = self.ros_bridge.get_manipulation_commands()
        hand_command_seq = int(command_state["hand_command_seq"])
        if hand_command_seq != self.hand_command_seq:
            self.hand_command_seq = hand_command_seq
            self.hand_command = str(command_state["hand_command"]).strip().lower()
            self.right_hand_desired = (
                self.right_hand_close.copy() if self.hand_command == "close" else self.right_hand_open.copy()
            )
            self.hand_state = "MOVING"

        arm_goal = command_state["arm_goal"]
        if arm_goal is None:
            return
        if int(arm_goal["seq"]) == self.arm_goal_seq:
            return
        self.arm_goal_seq = int(arm_goal["seq"])
        if not self.motion_mode_manager.allow_manipulation_ik():
            self.arm_goal_active = False
            self.arm_status = "FAIL"
            return
        position = np.clip(np.array(arm_goal["position"], dtype=np.float64), self.workspace_min, self.workspace_max)
        self.arm_goal_base = position
        self.arm_goal_orientation_base = np.array(arm_goal["orientation"], dtype=np.float64)
        self.arm_goal_active = True
        self.arm_goal_started = float(sim_time)
        self.arm_status = "BUSY"

    def _current_palm_pose_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wrist_pos = self.d.xpos[self.wrist_body_id].copy()
        wrist_rot = self.d.xmat[self.wrist_body_id].reshape(3, 3).copy()
        palm_pos = wrist_pos + wrist_rot @ self.palm_offset_local
        return wrist_pos, wrist_rot, palm_pos

    def _goal_pose_world(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.arm_goal_base is None:
            return None
        base_pos = self.d.xpos[self.base_body_id].copy()
        base_rot = self.d.xmat[self.base_body_id].reshape(3, 3).copy()
        goal_pos_world = base_pos + base_rot @ self.arm_goal_base
        goal_rot_world = base_rot @ quat_wxyz_to_rot(self.arm_goal_orientation_base)
        return goal_pos_world, goal_rot_world

    def _target_arm_posture(self) -> np.ndarray:
        mode = self.motion_mode_manager.current_mode
        if mode == MOTION_MODE_WALK:
            return np.concatenate((self.waist_neutral, self.right_arm_stow))
        if mode == MOTION_MODE_WALK_CARRY:
            return np.concatenate((self.waist_neutral, self.right_arm_carry))
        if mode == MOTION_MODE_STAND_BALANCE:
            if self.motion_mode_manager.requested_mode == MOTION_MODE_WALK_CARRY or self.grasped_object_label:
                return np.concatenate((self.waist_neutral, self.right_arm_carry))
            return np.concatenate((self.waist_neutral, self.right_arm_stand))
        if self.arm_goal_active:
            return self.arm_joint_goal.copy()
        return np.concatenate((self.waist_neutral, self.right_arm_stand))

    def _active_arm_rate_limits(self) -> np.ndarray:
        rate_limit = self.stand_arm_joint_rate_limit
        mode = self.motion_mode_manager.current_mode
        if mode == MOTION_MODE_WALK:
            rate_limit = self.walk_arm_joint_rate_limit
        elif mode == MOTION_MODE_WALK_CARRY:
            rate_limit = min(self.stand_arm_joint_rate_limit, self.arm_joint_rate_limit)
        elif mode == MOTION_MODE_MANIPULATE_STANDING:
            rate_limit = self.arm_joint_rate_limit
        rate_limits = np.full(self.arm_joint_goal.shape[0], rate_limit, dtype=np.float64)
        rate_limits[:3] = self.waist_joint_rate_limit
        return rate_limits

    def _active_gain_scale(self) -> float:
        mode = self.motion_mode_manager.current_mode
        if mode == MOTION_MODE_WALK:
            return self.walk_upper_gain_scale
        if mode == MOTION_MODE_WALK_CARRY:
            return self.walk_carry_upper_gain_scale
        if mode == MOTION_MODE_STAND_BALANCE:
            return self.stand_upper_gain_scale
        return self.manip_upper_gain_scale

    def _active_waist_gain_scale(self) -> float:
        mode = self.motion_mode_manager.current_mode
        if mode == MOTION_MODE_WALK:
            return self.walk_waist_gain_scale
        if mode == MOTION_MODE_WALK_CARRY:
            return self.walk_carry_waist_gain_scale
        if mode == MOTION_MODE_STAND_BALANCE:
            return self.stand_waist_gain_scale
        return self.manip_waist_gain_scale

    def _refresh_upper_target(self, dt: float):
        if not self.motion_mode_manager.allow_manipulation_ik():
            self.arm_goal_active = False
            self.arm_joint_goal = self._target_arm_posture()
            if self.motion_mode_manager.current_mode != MOTION_MODE_MANIPULATE_STANDING:
                self.arm_status = "IDLE"
        previous_arm_target = self.arm_joint_target.copy()
        previous_left_target = self.left_arm_target.copy()
        self.arm_joint_target, arm_joint_target_dq = rate_limit_towards(
            previous_arm_target,
            self.arm_joint_goal,
            self._active_arm_rate_limits(),
            dt,
            ranges=self.arm_joint_ranges,
        )
        self.left_arm_target, left_arm_target_dq = rate_limit_towards(
            previous_left_target,
            self.left_arm_park,
            np.full(self.left_arm_park.shape[0], self.left_arm_joint_rate_limit, dtype=np.float64),
            dt,
            ranges=self.left_joint_ranges,
        )
        self.upper_target[self.waist_slice] = self.arm_joint_target[:3]
        self.upper_target[self.left_slice] = self.left_arm_target
        self.upper_target[self.right_arm_slice] = self.arm_joint_target[3:]
        self.upper_target[self.right_hand_slice] = self.right_hand_target
        self.upper_target_dq[self.waist_slice] = arm_joint_target_dq[:3]
        self.upper_target_dq[self.left_slice] = left_arm_target_dq
        self.upper_target_dq[self.right_arm_slice] = arm_joint_target_dq[3:]
        self.upper_target_dq[self.right_hand_slice] = self.right_hand_target_dq

    def _step_hand_controller(self, dt: float):
        previous_target = self.right_hand_target.copy()
        alpha = float(np.clip(dt * self.hand_interp_rate, 0.0, 1.0))
        self.right_hand_target += alpha * (self.right_hand_desired - self.right_hand_target)
        self.right_hand_target_dq = (self.right_hand_target - previous_target) / max(float(dt), 1e-6)
        actual = self.d.qpos[self.right_hand_qpos_adr].copy()
        desired_err = np.max(np.abs(actual - self.right_hand_desired))
        target_err = np.max(np.abs(self.right_hand_target - self.right_hand_desired))
        if desired_err <= self.hand_state_tolerance and target_err <= self.hand_state_tolerance:
            self.hand_state = "CLOSED" if self.hand_command == "close" else "OPEN"
        else:
            self.hand_state = "MOVING"

    def _step_arm_controller(self, sim_time: float, dt: float):
        if not self.arm_goal_active:
            return
        if not self.motion_mode_manager.allow_manipulation_ik():
            self.arm_goal_active = False
            self.arm_status = "FAIL"
            return
        goal = self._goal_pose_world()
        if goal is None:
            return
        target_palm_world, target_rot_world = goal
        wrist_pos, wrist_rot, palm_pos = self._current_palm_pose_world()
        palm_err = target_palm_world - palm_pos
        palm_dist = float(np.linalg.norm(palm_err))
        rot_err = rotation_error(target_rot_world, wrist_rot)
        if palm_dist <= self.arm_goal_tolerance and float(np.linalg.norm(rot_err)) <= 0.12:
            self.arm_goal_active = False
            self.arm_status = "SUCCESS"
            self.arm_joint_goal = self.d.qpos[self.arm_qpos_adr].copy()
            return
        if sim_time - self.arm_goal_started > self.arm_goal_timeout_sec:
            self.arm_goal_active = False
            self.arm_status = "FAIL"
            self.arm_joint_goal = self.d.qpos[self.arm_qpos_adr].copy()
            return

        desired_wrist_world = target_palm_world - target_rot_world @ self.palm_offset_local
        pos_err = desired_wrist_world - wrist_pos

        jacp_torso = np.zeros((3, self.m.nv), dtype=np.float64)
        jacr_torso = np.zeros((3, self.m.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.m, self.d, jacp_torso, jacr_torso, self.torso_body_id)
        jacp_wrist = np.zeros((3, self.m.nv), dtype=np.float64)
        jacr_wrist = np.zeros((3, self.m.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.m, self.d, jacp_wrist, jacr_wrist, self.wrist_body_id)

        q_current = self.d.qpos[self.arm_qpos_adr].copy()
        desired_torso_rot = self.d.xmat[self.base_body_id].reshape(3, 3).copy()
        torso_rot = self.d.xmat[self.torso_body_id].reshape(3, 3).copy()
        torso_err = rotation_error(desired_torso_rot, torso_rot)

        nullspace = np.eye(q_current.shape[0], dtype=np.float64)
        dq_total = np.zeros(q_current.shape[0], dtype=np.float64)

        dq_task, nullspace = damped_task_step(
            jacr_torso[:, self.arm_qvel_adr],
            torso_err,
            nullspace,
            gain=self.torso_task_gain,
            damping=self.torso_task_damping,
        )
        dq_total += dq_task

        hand_error = np.concatenate((pos_err, self.manip_orientation_weight * rot_err))
        hand_jacobian = np.vstack((jacp_wrist[:, self.arm_qvel_adr], jacr_wrist[:, self.arm_qvel_adr]))
        dq_task, nullspace = damped_task_step(
            hand_jacobian,
            hand_error,
            nullspace,
            gain=self.arm_task_gain,
            damping=self.arm_task_damping,
        )
        dq_total += dq_task

        posture_target = self._target_arm_posture()
        posture_error = posture_target - q_current
        dq_task, _ = damped_task_step(
            np.eye(q_current.shape[0], dtype=np.float64),
            posture_error,
            nullspace,
            gain=self.posture_task_gain,
            damping=self.posture_task_damping,
        )
        dq_total += dq_task

        dq_limit = self._active_arm_rate_limits() * float(dt)
        dq_total = np.clip(dq_total, -dq_limit, dq_limit)
        self.arm_joint_goal = clamp_to_joint_ranges(q_current + dq_total, self.arm_joint_ranges)
        self.arm_status = "BUSY"

    def _ball_inside_capture_volume(self) -> bool:
        if self.ball_body_id < 0:
            return False
        _, wrist_rot, palm_pos = self._current_palm_pose_world()
        ball_pos = self.d.xpos[self.ball_body_id].copy()
        local_ball = wrist_rot.T @ (ball_pos - palm_pos)
        return bool(np.all(np.abs(local_ball) <= self.grasp_capture_halfsize))

    def _clear_grasp(self):
        self.grasped_object_label = ""
        self.grasp_contact_frames = 0
        self.grasp_loss_frames = 0

    def _update_grasp_state(self):
        if self.ball_body_id < 0 or not self.ball_geom_ids or not self.hand_geom_ids:
            self._clear_grasp()
            return
        if self.hand_command != "close":
            self._clear_grasp()
            return
        thumb_contact = False
        finger_contact = False
        for contact_idx in range(int(self.d.ncon)):
            contact = self.d.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.ball_geom_ids and geom2 in self.hand_geom_ids:
                hand_geom = geom2
            elif geom2 in self.ball_geom_ids and geom1 in self.hand_geom_ids:
                hand_geom = geom1
            else:
                continue
            body_name = self.geom_body_names.get(hand_geom, "")
            if body_name.startswith("right_hand_thumb"):
                thumb_contact = True
            if body_name.startswith("right_hand_index") or body_name.startswith("right_hand_middle") or body_name == "right_wrist_yaw_link":
                finger_contact = True
        contact_ok = thumb_contact and finger_contact and self._ball_inside_capture_volume()
        if contact_ok:
            self.grasp_contact_frames += 1
            self.grasp_loss_frames = 0
        else:
            self.grasp_contact_frames = 0
            if self.grasped_object_label:
                self.grasp_loss_frames += 1
        if not self.grasped_object_label and contact_ok and self.grasp_contact_frames >= self.grasp_contact_frames_required:
            self.grasped_object_label = "orangeball"
            return
        if self.grasped_object_label and not contact_ok and self.grasp_loss_frames >= self.grasp_loss_frames_allowed:
            self._clear_grasp()

    def step(self, sim_time: float, dt: float):
        if not self.enabled:
            return
        self._consume_ros_commands(sim_time)
        self._step_hand_controller(dt)
        self._step_arm_controller(sim_time, dt)
        self._update_grasp_state()
        self._refresh_upper_target(dt)
        self._sync_bridge_state()

    def compute_torque(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros(0, dtype=np.float64)
        q_upper = self.d.qpos[self.upper_qpos_adr].copy()
        dq_upper = self.d.qvel[self.upper_qvel_adr].copy()
        gain_scale = self._active_gain_scale()
        waist_gain_scale = self._active_waist_gain_scale()
        kps = self.upper_kps.copy() * gain_scale
        kds = self.upper_kds.copy() * gain_scale
        kps[self.waist_slice] *= waist_gain_scale
        kds[self.waist_slice] *= waist_gain_scale
        target_dq = (
            self.upper_target_dq
            if self.motion_mode_manager.allow_manipulation_ik()
            else self.zero_upper_dq
        )
        return pd_control(
            self.upper_target,
            q_upper,
            kps,
            target_dq,
            dq_upper,
            kds,
        )


def resolve_ctrl_indices(actuator_name_to_index: dict[str, int], required_names: list[str]) -> np.ndarray:
    missing = [name for name in required_names if name not in actuator_name_to_index]
    if missing:
        raise ValueError(f"Missing actuator names: {', '.join(missing)}")
    return np.asarray([actuator_name_to_index[name] for name in required_names], dtype=int)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation
