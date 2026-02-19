from __future__ import annotations

import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from pathlib import Path
from warnings import warn
import re

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

ROS2_ENABLED = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import String

    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState, Imu
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped, Twist
    from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
    ROS2_ENABLED = True
except Exception:
    ROS2_ENABLED = False
    class Node:
        def __init__(self, *args, **kwargs):
            raise ImportError("Node is not installed.")


def sim_time_to_sec_nsec(sim_time: float) -> tuple[int, int]:
    sec = int(sim_time)
    nsec = int((sim_time - sec) * 1e9)
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    elif nsec < 0:
        sec -= 1
        nsec += 1_000_000_000
    return sec, nsec


class MujocoROS2Bridge(Node):
    """
    Publishes:
      /clock
      /tf (odom -> base_link)
      /tf_static (base_link -> livox_frame, base_link -> camera_link)
      /joint_states
      /odom
      /imu
    """

    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        *,
        base_body_name="pelvis",
        livox_body_name="lidar_frame",
        camera_body_name="depth_camera_frame",
        odom_frame="odom",
        base_frame="base_link",
        livox_frame="livox_frame",
        camera_frame="camera_link",
        cmd_vel_topic="/unitree/cmd_vel",
        cmd_vel_timeout_sec=0.8,
        cmd_lock: threading.Lock | None = None,
        cmd_shared: np.ndarray | None = None,
        clock_hz: float | None = None,
        tf_hz: float | None = None,
        odom_hz: float | None = None,
        imu_hz: float | None = None,
        joint_hz: float | None = None,
        profile_enabled: bool = False,
    ):
        super().__init__("mujoco_ros2_bridge")
        self.m = m
        self.d = d
        self.p = Path(str(Path(LEGGED_GYM_ROOT_DIR) / "resources/robots/g1_description/g1_12dof.urdf")).expanduser().resolve()
        self.p29 = Path(str(Path(LEGGED_GYM_ROOT_DIR) / "resources/robots/g1_description/g1_29dof.urdf")).expanduser().resolve()
        
        # QoS: sensor-style (best effort, low latency)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publishers
        self.pub_clock = self.create_publisher(Clock, "/clock", qos)
        self.pub_joint = self.create_publisher(JointState, "/joint_states", qos)
        self.pub_odom = self.create_publisher(Odometry, "/odom", qos)
        self.pub_imu = self.create_publisher(Imu, "/imu", qos)

        # TF broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Description
        self.pub_robot_description = self.create_publisher(String, "/robot_description", 1)

        # Frames
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.livox_frame = livox_frame
        self.camera_frame = camera_frame
        self.cmd_lock = cmd_lock
        self.cmd_shared = cmd_shared
        self.last_cmd_vel_walltime = time.time()
        self.cmd_vel_timeout_sec = float(cmd_vel_timeout_sec)

        # IDs from MJCF
        self.base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.livox_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, livox_body_name)
        self.camera_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, camera_body_name)

        if self.base_body_id < 0:
            raise ValueError(f"Body '{base_body_name}' not found in MJCF")
        if self.livox_body_id < 0:
            raise ValueError(f"Body '{livox_body_name}' not found in MJCF")
        if self.camera_body_id < 0:
            raise ValueError(f"Body '{camera_body_name}' not found in MJCF")

        # Cache joint indexing for /joint_states
        self._js_names, self._js_qposadr, self._js_dofadr = self._build_joint_state_index()

        # Publish static TF & Description once
        self._publish_static_tf_once()
        self._publish_robot_description_once()

        # sim clock accumulator
        self.sim_time = 0.0
        self._profile_enabled = bool(profile_enabled)
        self._stats = self._new_stats()

        self._clock_period = self._hz_to_period(clock_hz)
        self._tf_period = self._hz_to_period(tf_hz)
        self._odom_period = self._hz_to_period(odom_hz)
        self._imu_period = self._hz_to_period(imu_hz)
        self._joint_period = self._hz_to_period(joint_hz)

        self._last_clock_t = -1e12
        self._last_tf_t = -1e12
        self._last_odom_t = -1e12
        self._last_imu_t = -1e12
        self._last_joint_t = -1e12

        # MuJoCo joint names present in this MJCF
        mj_joints = set()
        for jid in range(self.m.njnt):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name:
                mj_joints.add(name)

        # URDF movable joints (includes upper body joints in 29dof URDF)
        urdf_movable = self._parse_urdf_movable_joint_names(str(self.p29))

        # Joints that exist in URDF but not in MuJoCo -> placeholders
        self.placeholder_joint_names = [jn for jn in urdf_movable if jn not in mj_joints]

        # You can also optionally exclude the floating base joint if it exists in URDF
        # (some URDFs do not include it as a "joint" in the same way)
        self.placeholder_joint_names = [
            jn for jn in self.placeholder_joint_names
            if jn != "floating_base_joint"
        ]
        print(f"[upper-body placeholder] {len(self.placeholder_joint_names)} placeholder joints")

        self.cmd_sub = None
        if self.cmd_lock is not None and self.cmd_shared is not None:
            self.cmd_sub = self.create_subscription(Twist, cmd_vel_topic, self._cmd_vel_cb, 10)
            self.get_logger().info(
                f"Subscribed to velocity command topic: {cmd_vel_topic} (timeout={self.cmd_vel_timeout_sec:.2f}s)"
            )

    @staticmethod
    def _hz_to_period(hz: float | None) -> float | None:
        if hz is None:
            return None
        hz = float(hz)
        if hz <= 0.0:
            return None
        return 1.0 / hz

    @staticmethod
    def _new_stats() -> dict:
        return {
            "publish_calls": 0,
            "sim_time_s": 0.0,
            "clock_msgs": 0,
            "tf_msgs": 0,
            "odom_msgs": 0,
            "imu_msgs": 0,
            "joint_msgs": 0,
            "t_clock_s": 0.0,
            "t_tf_s": 0.0,
            "t_odom_s": 0.0,
            "t_imu_s": 0.0,
            "t_joint_s": 0.0,
            "t_total_s": 0.0,
        }

    def _should_publish(self, period: float | None, sim_time: float, attr_name: str) -> bool:
        if period is None:
            return True
        last_t = getattr(self, attr_name)
        if sim_time + 1e-12 >= last_t + period:
            setattr(self, attr_name, sim_time)
            return True
        return False

    def _add_stat_time(self, key: str, dt: float) -> None:
        if self._profile_enabled:
            self._stats[key] += float(dt)

    def get_stats(self, reset: bool = False) -> dict:
        stats = dict(self._stats)
        sim_time = max(float(stats["sim_time_s"]), 1e-9)
        for k in ("clock_msgs", "tf_msgs", "odom_msgs", "imu_msgs", "joint_msgs"):
            stats[f"{k}_per_sec"] = float(stats[k]) / sim_time

        total_calls = max(int(stats["publish_calls"]), 1)
        stats["total_ms_per_call"] = 1e3 * float(stats["t_total_s"]) / total_calls
        stats["clock_ms_per_msg"] = 1e3 * float(stats["t_clock_s"]) / max(int(stats["clock_msgs"]), 1)
        stats["tf_ms_per_msg"] = 1e3 * float(stats["t_tf_s"]) / max(int(stats["tf_msgs"]), 1)
        stats["odom_ms_per_msg"] = 1e3 * float(stats["t_odom_s"]) / max(int(stats["odom_msgs"]), 1)
        stats["imu_ms_per_msg"] = 1e3 * float(stats["t_imu_s"]) / max(int(stats["imu_msgs"]), 1)
        stats["joint_ms_per_msg"] = 1e3 * float(stats["t_joint_s"]) / max(int(stats["joint_msgs"]), 1)

        if reset:
            self._stats = self._new_stats()

        return stats

    def _augment_joint_state(self, js_msg):
        """
        Mutates a sensor_msgs/JointState: appends placeholder joints at 0 position/velocity.
        Only adds joints that aren't already present in js_msg.name.
        """
        existing = set(js_msg.name)
        for jn in self.placeholder_joint_names:
            if jn in existing:
                continue
            js_msg.name.append(jn)
            js_msg.position.append(0.0)
            # keep arrays aligned if you publish velocity/effort
            if js_msg.velocity is not None:
                js_msg.velocity.append(0.0)
            if js_msg.effort is not None:
                js_msg.effort.append(0.0)

    def _parse_urdf_movable_joint_names(self, urdf_path: str) -> list[str]:
        """
        Return URDF joint names that are not 'fixed'.
        """
        txt = Path(urdf_path).read_text(encoding="utf-8")
        # <joint name="..." type="...">
        matches = re.findall(r'<joint\s+name="([^"]+)"\s+type="([^"]+)"', txt)
        return [name for (name, jtype) in matches if jtype.strip().lower() != "fixed"]

    def _cmd_vel_cb(self, msg: Twist):
        with self.cmd_lock:
            self.cmd_shared[0] = float(msg.linear.x)
            self.cmd_shared[1] = float(msg.linear.y)
            self.cmd_shared[2] = float(msg.angular.z)
        self.last_cmd_vel_walltime = time.time()

    def enforce_cmd_vel_timeout(self):
        if self.cmd_sub is None:
            return
        if time.time() - self.last_cmd_vel_walltime <= self.cmd_vel_timeout_sec:
            return
        with self.cmd_lock:
            # Support both numpy arrays and Python lists.
            if hasattr(self.cmd_shared, "fill"):
                self.cmd_shared.fill(0.0)
            else:
                for i in range(min(3, len(self.cmd_shared))):
                    self.cmd_shared[i] = 0.0

    # --------------------------
    # Utilities
    # --------------------------
    def _mat_to_quat_wxyz(self, R: np.ndarray) -> np.ndarray:
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
        return q  # wxyz

    def _build_joint_state_index(self):
        names = []
        qposadr = []
        dofadr = []
        for jid in range(self.m.njnt):
            jtype = int(self.m.jnt_type[jid])
            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                continue  # base
            # ignore ball joints etc for now (rare in this model)
            # hinge/slide have 1 qpos, 1 dof
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not name:
                continue
            names.append(name)
            qposadr.append(int(self.m.jnt_qposadr[jid]))
            dofadr.append(int(self.m.jnt_dofadr[jid]))
        return names, np.asarray(qposadr, dtype=int), np.asarray(dofadr, dtype=int)

    def _publish_static_tf_once(self):
        """
        Publish:
          base_link -> livox_frame  (from MJCF relative body transform)
          base_link -> camera_link (from MJCF relative body transform)
        These bodies are rigidly attached (no joints), so static is correct.
        """
        stamp = self.get_clock().now().to_msg()

        # We assume lidar_frame and depth_camera_frame are direct children of pelvis in your XML.
        # In that case, their model-local pose (m.body_pos/body_quat) is already in base frame.
        # If later you nest them deeper, we can compute parent chain, but your XML is direct.

        static_msgs = []

        # base -> lidar_frame
        t1 = TransformStamped()
        t1.header.stamp = stamp
        t1.header.frame_id = self.base_frame
        t1.child_frame_id = self.livox_frame

        p = self.m.body_pos[self.livox_body_id]
        q = self.m.body_quat[self.livox_body_id]  # wxyz in MuJoCo

        t1.transform.translation.x = float(p[0])
        t1.transform.translation.y = float(p[1])
        t1.transform.translation.z = float(p[2])
        t1.transform.rotation.w = float(q[0])
        t1.transform.rotation.x = float(q[1])
        t1.transform.rotation.y = float(q[2])
        t1.transform.rotation.z = float(q[3])
        static_msgs.append(t1)

        # base -> camera_link
        t2 = TransformStamped()
        t2.header.stamp = stamp
        t2.header.frame_id = self.base_frame
        t2.child_frame_id = self.camera_frame

        p = self.m.body_pos[self.camera_body_id]
        q = self.m.body_quat[self.camera_body_id]  # wxyz

        t2.transform.translation.x = float(p[0])
        t2.transform.translation.y = float(p[1])
        t2.transform.translation.z = float(p[2])
        t2.transform.rotation.w = float(q[0])
        t2.transform.rotation.x = float(q[1])
        t2.transform.rotation.y = float(q[2])
        t2.transform.rotation.z = float(q[3])
        static_msgs.append(t2)

        self.tf_static_broadcaster.sendTransform(static_msgs)

        t = TransformStamped()
        t.header.frame_id = "odom"
        t.child_frame_id = "world"
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform([t])


    def _publish_robot_description_once(self):
        """
        Publish a minimal URDF so RViz can display a RobotModel.
        This includes:
        base_link
        livox_frame
        camera_link
        and fixed joints from base_link to sensors.

        Replace this later with the real G1 URDF for full visualization.
        """
        if not self.p.exists():
            self.get_logger().error(f"/robot_description URDF not found: {self.p}")
            return

        urdf = self.p.read_text(encoding="utf-8")
        msg = String()
        msg.data = urdf
        self.pub_robot_description.publish(msg)

    # --------------------------
    # Publish per sim step
    # --------------------------
    def publish_step(self, dt: float, *, stamp_msg=None, sim_time: float | None = None):
        """
        Call once per MuJoCo step AFTER mj_step.
        """
        t_call = time.perf_counter()
        if sim_time is None:
            self.sim_time += float(dt)
        else:
            self.sim_time = float(sim_time)

        if stamp_msg is None:
            stamp = self.get_clock().now().to_msg()
            sec, nsec = sim_time_to_sec_nsec(self.sim_time)
            stamp.sec = sec
            stamp.nanosec = nsec
        else:
            stamp = stamp_msg

        self._stats["publish_calls"] += 1
        self._stats["sim_time_s"] += float(dt)

        publish_clock = self._should_publish(self._clock_period, self.sim_time, "_last_clock_t")
        publish_tf = self._should_publish(self._tf_period, self.sim_time, "_last_tf_t")
        publish_odom = self._should_publish(self._odom_period, self.sim_time, "_last_odom_t")
        publish_imu = self._should_publish(self._imu_period, self.sim_time, "_last_imu_t")
        publish_joint = self._should_publish(self._joint_period, self.sim_time, "_last_joint_t")

        if publish_clock:
            t0 = time.perf_counter()
            clk = Clock()
            sec, nsec = sim_time_to_sec_nsec(self.sim_time)
            clk.clock.sec = sec
            clk.clock.nanosec = nsec
            self.pub_clock.publish(clk)
            self._stats["clock_msgs"] += 1
            self._add_stat_time("t_clock_s", time.perf_counter() - t0)

        if publish_tf or publish_odom or publish_imu:
            # base pose in world/odom
            p_w = self.d.xpos[self.base_body_id].copy()
            R_w_base = self.d.xmat[self.base_body_id].reshape(3, 3).copy()
            q_wxyz = self._mat_to_quat_wxyz(R_w_base)

            # base spatial velocity (MuJoCo provides cvel: [ang; lin] in world frame)
            v6 = self.d.cvel[self.base_body_id].copy()
            w_w = v6[0:3]
            v_w = v6[3:6]

            if publish_tf:
                t0 = time.perf_counter()
                tfmsg = TransformStamped()
                tfmsg.header.stamp = stamp
                tfmsg.header.frame_id = self.odom_frame
                tfmsg.child_frame_id = self.base_frame
                tfmsg.transform.translation.x = float(p_w[0])
                tfmsg.transform.translation.y = float(p_w[1])
                tfmsg.transform.translation.z = float(p_w[2])
                tfmsg.transform.rotation.w = float(q_wxyz[0])
                tfmsg.transform.rotation.x = float(q_wxyz[1])
                tfmsg.transform.rotation.y = float(q_wxyz[2])
                tfmsg.transform.rotation.z = float(q_wxyz[3])
                self.tf_broadcaster.sendTransform(tfmsg)
                self._stats["tf_msgs"] += 1
                self._add_stat_time("t_tf_s", time.perf_counter() - t0)

            if publish_odom:
                t0 = time.perf_counter()
                odom = Odometry()
                odom.header.stamp = stamp
                odom.header.frame_id = self.odom_frame
                odom.child_frame_id = self.base_frame
                odom.pose.pose.position.x = float(p_w[0])
                odom.pose.pose.position.y = float(p_w[1])
                odom.pose.pose.position.z = float(p_w[2])
                odom.pose.pose.orientation.w = float(q_wxyz[0])
                odom.pose.pose.orientation.x = float(q_wxyz[1])
                odom.pose.pose.orientation.y = float(q_wxyz[2])
                odom.pose.pose.orientation.z = float(q_wxyz[3])
                odom.twist.twist.linear.x = float(v_w[0])
                odom.twist.twist.linear.y = float(v_w[1])
                odom.twist.twist.linear.z = float(v_w[2])
                odom.twist.twist.angular.x = float(w_w[0])
                odom.twist.twist.angular.y = float(w_w[1])
                odom.twist.twist.angular.z = float(w_w[2])
                self.pub_odom.publish(odom)
                self._stats["odom_msgs"] += 1
                self._add_stat_time("t_odom_s", time.perf_counter() - t0)

            if publish_imu:
                t0 = time.perf_counter()
                imu = Imu()
                imu.header.stamp = stamp
                imu.header.frame_id = self.base_frame
                imu.orientation.w = float(q_wxyz[0])
                imu.orientation.x = float(q_wxyz[1])
                imu.orientation.y = float(q_wxyz[2])
                imu.orientation.z = float(q_wxyz[3])

                # angular velocity: express in base frame
                w_b = (R_w_base.T @ w_w.reshape(3, 1)).reshape(3)
                imu.angular_velocity.x = float(w_b[0])
                imu.angular_velocity.y = float(w_b[1])
                imu.angular_velocity.z = float(w_b[2])

                # linear acceleration: use body spatial acceleration if available
                if hasattr(self.d, "cacc"):
                    a6 = self.d.cacc[self.base_body_id].copy()
                    a_w = a6[3:6]
                else:
                    a_w = np.zeros(3, dtype=np.float64)

                g_w = np.array(self.m.opt.gravity, dtype=np.float64)
                proper_a_w = a_w - g_w
                proper_a_b = (R_w_base.T @ proper_a_w.reshape(3, 1)).reshape(3)
                imu.linear_acceleration.x = float(proper_a_b[0])
                imu.linear_acceleration.y = float(proper_a_b[1])
                imu.linear_acceleration.z = float(proper_a_b[2])

                # unknown covariances
                imu.orientation_covariance[0] = -1.0
                imu.angular_velocity_covariance[0] = -1.0
                imu.linear_acceleration_covariance[0] = -1.0

                self.pub_imu.publish(imu)
                self._stats["imu_msgs"] += 1
                self._add_stat_time("t_imu_s", time.perf_counter() - t0)

        if publish_joint:
            t0 = time.perf_counter()
            js = JointState()
            js.header.stamp = stamp
            js.name = list(self._js_names)
            if len(self._js_qposadr) > 0:
                js.position = self.d.qpos[self._js_qposadr].astype(np.float64).tolist()
            if len(self._js_dofadr) > 0:
                js.velocity = self.d.qvel[self._js_dofadr].astype(np.float64).tolist()
            self._augment_joint_state(js)
            self.pub_joint.publish(js)
            self._stats["joint_msgs"] += 1
            self._add_stat_time("t_joint_s", time.perf_counter() - t0)

        self._add_stat_time("t_total_s", time.perf_counter() - t_call)

class LivoxPublisher(Node):
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        super().__init__("livox_mid360_sim")
        self.m = m
        self.d = d

        # Use SensorDataQoS so RViz / typical pipelines behave well for live sensors
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        from sensor_msgs.msg import PointCloud2
        self.pub = self.create_publisher(PointCloud2, "/livox/points", qos)

        # IDs from MJCF
        self.body_lidar_frame = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "lidar_frame")
        self.site_livox = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "livox_mid360")
        if self.body_lidar_frame < 0 or self.site_livox < 0:
            raise ValueError("Could not find lidar_frame body or livox_mid360 site in MJCF")

        # TF broadcaster
        from tf2_ros import TransformBroadcaster
        from geometry_msgs.msg import TransformStamped
        self._tf_pub = TransformBroadcaster(self)
        self._TransformStamped = TransformStamped

    def publish_tf(self, stamp_msg):
        # world -> lidar_frame (body)
        p = self.d.xpos[self.body_lidar_frame].copy()
        R = self.d.xmat[self.body_lidar_frame].reshape(3, 3).copy()
        self._send_tf("world", "lidar_frame", p, R, stamp_msg)

        # world -> livox_mid360 (site)
        p = self.d.site_xpos[self.site_livox].copy()
        R = self.d.site_xmat[self.site_livox].reshape(3, 3).copy()
        self._send_tf("world", "livox_mid360", p, R, stamp_msg)

    def _send_tf(self, parent, child, p_w, R_w_child, stamp_msg):
        t = self._TransformStamped()
        t.header.stamp = stamp_msg
        t.header.frame_id = parent
        t.child_frame_id = child

        t.transform.translation.x = float(p_w[0])
        t.transform.translation.y = float(p_w[1])
        t.transform.translation.z = float(p_w[2])

        q_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q_wxyz, R_w_child.reshape(-1))
        t.transform.rotation.w = float(q_wxyz[0])
        t.transform.rotation.x = float(q_wxyz[1])
        t.transform.rotation.y = float(q_wxyz[2])
        t.transform.rotation.z = float(q_wxyz[3])

        self._tf_pub.sendTransform(t)

class D435iPublisher(Node):
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        super().__init__("d435i_sim")
        self.m = m
        self.d = d

        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )

        from sensor_msgs.msg import Image, CameraInfo
        self.pub_color = self.create_publisher(Image, "/intel/D435i/color", qos)
        self.pub_depth = self.create_publisher(Image, "/intel/D435i/depth", qos)
        self.pub_aligned = self.create_publisher(Image, "/intel/D435i/aligned_depth_to_color", qos)

        self.pub_camera_info = self.create_publisher(CameraInfo, "/intel/D435i/camera_info", qos)

        # TF
        from tf2_ros import TransformBroadcaster
        from geometry_msgs.msg import TransformStamped
        self._tf_pub = TransformBroadcaster(self)
        self._TransformStamped = TransformStamped

        # IDs from your XML names
        self.body_pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_lidar_frame = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "lidar_frame")
        self.body_depth_cam_frame = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "depth_camera_frame")
        self.cam_depth = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "d435i_depth_cam")

        if self.body_pelvis < 0 or self.body_lidar_frame < 0 or self.body_depth_cam_frame < 0 or self.cam_depth < 0:
            raise ValueError("Could not find required bodies/camera from MJCF: pelvis/lidar_frame/depth_camera_frame/d435i_depth_cam")

        # Frame names (reflect MJCF)
        self.frame_world = "world"
        self.frame_pelvis = "pelvis"
        self.frame_lidar = "lidar_frame"
        self.frame_depth_cam_frame = "depth_camera_frame"
        self.frame_cam = "d435i_depth_cam"
        self.frame_cam_optical = "d435i_depth_cam_optical"  # ROS optical convention

        # Optical transform relative to MuJoCo camera frame:
        # Optical: +X right, +Y down, +Z forward
        # MuJoCo cam: +X right, +Y up, -Z forward
        # So optical = R * mjcam where R = diag(1,-1,-1)
        self.R_opt_mjcam = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        # Images are rotated 90deg clockwise before publishing. That changes the
        # implied optical basis by +90deg about +Z, so TF must include the inverse
        # rotation to keep DepthCloud geometry consistent.
        self.R_optical_framecorr = np.array(
            [[0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def publish_frames_and_images(self, frame_dict: dict, stamp_msg=None):
        stamp = stamp_msg if stamp_msg is not None else self.get_clock().now().to_msg()

        # ----- TF: world -> pelvis, lidar_frame, depth_camera_frame -----
        self._publish_body_tf(self.frame_world, self.frame_pelvis, self.body_pelvis, stamp)
        self._publish_body_tf(self.frame_world, self.frame_lidar, self.body_lidar_frame, stamp)
        self._publish_body_tf(self.frame_world, self.frame_depth_cam_frame, self.body_depth_cam_frame, stamp)

        # ----- TF: world -> camera frame (from cam_xpos/xmat) -----
        p = self.d.cam_xpos[self.cam_depth].copy()
        R = self.d.cam_xmat[self.cam_depth].reshape(3, 3).copy()  # camera->world
        self._publish_pose_tf(self.frame_world, self.frame_cam, p, R, stamp)

        # ----- TF: camera -> optical (fixed) -----
        # We publish world->optical using: R_w_opt = R_w_cam @ R_cam_opt
        # where R_cam_opt = (R_opt_mjcam)^T because we defined R_opt_mjcam mapping mjcam->optical.
        R_cam_opt = self.R_opt_mjcam.T
        R_w_opt = R @ R_cam_opt @ self.R_optical_framecorr
        self._publish_pose_tf(self.frame_world, self.frame_cam_optical, p, R_w_opt, stamp)

        # ----- Images -----
        rgb = frame_dict.get("rgb_u8", None)
        depth_mm = frame_dict.get("depth_mm_u16", None)

        rgb = rot90_cw(rgb)
        depth_mm = rot90_cw(depth_mm)

        # Use optical frame_id for images (most ROS stacks expect *_optical_frame)
        img_frame_id = self.frame_cam_optical

        if rgb is not None:
            self.pub_color.publish(ros_image_from_numpy(rgb, frame_id=img_frame_id, stamp_msg=stamp, encoding="rgb8"))

        if depth_mm is not None:
            depth_msg = ros_image_from_numpy(depth_mm, frame_id=img_frame_id, stamp_msg=stamp, encoding="16UC1")
            self.pub_depth.publish(depth_msg)
            # In your sim, depth is already from the same rendered viewpoint; treat as aligned
            self.pub_aligned.publish(depth_msg)

        # ----- CameraInfo -----
        intr = frame_dict.get("intrinsics", None)
        if intr is not None:
            fx2, fy2, cx2, cy2, W2, H2 = rotate_intrinsics_90cw(intr["fx"], intr["fy"], intr["cx"], intr["cy"], frame_dict["width"], frame_dict["height"])
            info = camera_info_from_intrinsics(
                W2, H2,
                fx2, fy2, cx2, cy2,
                frame_id=img_frame_id, stamp_msg=stamp
            )
            self.pub_camera_info.publish(info)
            
    def _publish_body_tf(self, parent: str, child: str, body_id: int, stamp):
        p = self.d.xpos[body_id].copy()
        R = self.d.xmat[body_id].reshape(3, 3).copy()  # body->world
        self._publish_pose_tf(parent, child, p, R, stamp)

    def _publish_pose_tf(self, parent: str, child: str, p_w: np.ndarray, R_w_child: np.ndarray, stamp):
        t = self._TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child

        t.transform.translation.x = float(p_w[0])
        t.transform.translation.y = float(p_w[1])
        t.transform.translation.z = float(p_w[2])

        q_xyzw = _mat_to_quat_xyzw(R_w_child)
        t.transform.rotation.x = float(q_xyzw[0])
        t.transform.rotation.y = float(q_xyzw[1])
        t.transform.rotation.z = float(q_xyzw[2])
        t.transform.rotation.w = float(q_xyzw[3])

        self._tf_pub.sendTransform(t)

def rot90_cw(img: np.ndarray) -> np.ndarray:
    # 90° clockwise
    return np.rot90(img, k=-1)  # same as k=3

def rotate_intrinsics_90cw(fx, fy, cx, cy, width, height):
    new_width  = height
    new_height = width
    fx2 = fy
    fy2 = fx
    cx2 = cy
    cy2 = (width - 1) - cx
    return fx2, fy2, cx2, cy2, new_width, new_height


def pointcloud2_from_xyz(
    points_xyz: np.ndarray,
    *,
    frame_id: str,
    stamp_msg,
    intensity: np.ndarray | None = None,
):
    """
    Build a sensor_msgs/PointCloud2 with fields:
      - x, y, z (float32)
      - intensity (float32) [optional; if not provided, uses 0.0]

    points_xyz: (N,3) float array
    intensity: (N,) optional
    stamp_msg: builtin_interfaces/msg/Time (e.g., node.get_clock().now().to_msg())
    """
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {pts.shape}")

    n = pts.shape[0]
    if intensity is None:
        inten = None
    else:
        inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if inten.shape[0] != n:
            raise ValueError(f"intensity must have length {n}, got {inten.shape[0]}")

    # Pack as little-endian float32: x y z intensity
    # point_step = 16 bytes
    packed = np.empty((n, 4), dtype=np.dtype("<f4"))
    if n > 0:
        packed[:, :3] = pts
        if inten is None:
            packed[:, 3] = 0.0
        else:
            packed[:, 3] = inten

    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.is_dense = True

    msg.fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.data = packed.tobytes()
    return msg

def ros_image_from_numpy(arr: np.ndarray, *, frame_id: str, stamp_msg, encoding: str):
    """
    Create sensor_msgs/Image from numpy array.
    - rgb8: (H,W,3) uint8
    - 16UC1: (H,W) uint16
    """
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id

    if encoding == "rgb8":
        a = np.asarray(arr)
        assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3, f"rgb8 expects (H,W,3) uint8, got {a.shape} {a.dtype}"
        msg.height, msg.width = a.shape[0], a.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = a.tobytes()

    elif encoding == "16UC1":
        a = np.asarray(arr)
        assert a.dtype == np.uint16 and a.ndim == 2, f"16UC1 expects (H,W) uint16, got {a.shape} {a.dtype}"
        msg.height, msg.width = a.shape[0], a.shape[1]
        msg.encoding = "16UC1"
        msg.is_bigendian = False
        msg.step = msg.width * 2
        msg.data = a.tobytes()

    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return msg

def camera_info_from_intrinsics(width: int, height: int, fx: float, fy: float, cx: float, cy: float, *, frame_id: str, stamp_msg):
    from sensor_msgs.msg import CameraInfo
    msg = CameraInfo()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    msg.width = int(width)
    msg.height = int(height)

    # K (3x3) row-major
    msg.k = [
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    ]

    # P (3x4) row-major, assume no stereo baseline
    msg.p = [
        fx, 0.0, cx, 0.0,
        0.0, fy, cy, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]

    # Identity rotation
    msg.r = [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]

    # Distortion unknown for sim; set plumb_bob with zeros (common)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return msg

def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """MuJoCo gives xmat (3x3) but we need ROS quaternion (x,y,z,w)."""
    # MuJoCo mju_mat2Quat returns (w,x,y,z)
    q_wxyz = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q_wxyz, R.reshape(-1))
    # convert to ROS (x,y,z,w)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

def start_cmd_web_ui(cmd_shared: np.ndarray, cmd_lock: threading.Lock, cmd_init: np.ndarray, 
                     rgb_jpeg_shared, rgb_lock,
                     host="127.0.0.1", port=8000,
                     slider_min=-1.0, slider_max=1.0):
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>cmd sliders</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
    .row {{ margin: 14px 0; }}
    input[type=range] {{ width: 420px; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
    button {{ padding: 8px 12px; }}
  </style>
</head>
<body>
  <h2>MuJoCo cmd sliders</h2>
  <p>Adjust <code>cmd = [vx, vy, yaw_rate]</code> live.</p>

  <div class="row">
    <label>vx: <span id="vxv"></span></label><br/>
    <input id="vx" type="range" min="{slider_min}" max="{slider_max}" step="0.001" value="{float(cmd_init[0])}">
  </div>

  <div class="row">
    <label>vy: <span id="vyv"></span></label><br/>
    <input id="vy" type="range" min="{slider_min}" max="{slider_max}" step="0.001" value="{float(cmd_init[1])}">
  </div>

  <div class="row">
    <label>yaw_rate: <span id="yawv"></span></label><br/>
    <input id="yaw" type="range" min="{slider_min}" max="{slider_max}" step="0.001" value="{float(cmd_init[2])}">
  </div>

  <button id="reset">Reset to cmd_init</button>

  <p>Current cmd: <code id="cmd"></code></p>

  <h3>D435i RGB (live)</h3>
  <img id="rgb" style="transform: rotate(90deg); max-width: 900px; width: 100%; border: 1px solid #ddd; border-radius: 8px; margin-top: 51px; margin-left: -53px;"
       src="/d435/rgb.jpg" />

<script>
const vx = document.getElementById("vx");
const vy = document.getElementById("vy");
const yaw = document.getElementById("yaw");
const vxv = document.getElementById("vxv");
const vyv = document.getElementById("vyv");
const yawv = document.getElementById("yawv");
const cmdEl = document.getElementById("cmd");
const resetBtn = document.getElementById("reset");

function render() {{
  vxv.textContent = (+vx.value).toFixed(3);
  vyv.textContent = (+vy.value).toFixed(3);
  yawv.textContent = (+yaw.value).toFixed(3);
  cmdEl.textContent = `[${{(+vx.value).toFixed(3)}}, ${{(+vy.value).toFixed(3)}}, ${{(+yaw.value).toFixed(3)}}]`;
}}

async function send() {{
  render();
  await fetch("/cmd", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ vx:+vx.value, vy:+vy.value, yaw:+yaw.value }})
  }});
}}

vx.addEventListener("input", send);
vy.addEventListener("input", send);
yaw.addEventListener("input", send);

resetBtn.addEventListener("click", async () => {{
  await fetch("/reset", {{ method:"POST" }});
  const r = await fetch("/cmd");
  const c = await r.json();
  vx.value = c.vx; vy.value = c.vy; yaw.value = c.yaw;
  render();
}});

(async () => {{
  const r = await fetch("/cmd");
  const c = await r.json();
  vx.value = c.vx; vy.value = c.vy; yaw.value = c.yaw;
  render();
}})();

const img = document.getElementById("rgb");
setInterval(() => {{
  img.src = "/d435/rgb.jpg?ts=" + Date.now();
}}, 100); // ~20 FPS refresh (camera is 30fps; this is fine)
</script>
</body>
</html>
"""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, content_type, body_bytes: bytes):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))
                return
            if self.path == "/cmd":
                with cmd_lock:
                    data = {"vx": float(cmd_shared[0]), "vy": float(cmd_shared[1]), "yaw": float(cmd_shared[2])}
                self._send(200, "application/json", json.dumps(data).encode("utf-8"))
                return
            if self.path.startswith("/d435/rgb.jpg"):
                with rgb_lock:
                    jpg = rgb_jpeg_shared["jpeg"]
                if not jpg:
                    self._send(204, "text/plain", b"")  # no content yet
                    return
                self._send(200, "image/jpeg", jpg)
                return
            self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path == "/cmd":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    with cmd_lock:
                        cmd_shared[0] = float(payload.get("vx", cmd_shared[0]))
                        cmd_shared[1] = float(payload.get("vy", cmd_shared[1]))
                        cmd_shared[2] = float(payload.get("yaw", cmd_shared[2]))
                    self._send(200, "application/json", b'{"ok":true}')
                except Exception as e:
                    self._send(400, "application/json", json.dumps({"ok": False, "err": str(e)}).encode("utf-8"))
                return

            if self.path == "/reset":
                with cmd_lock:
                    cmd_shared[:] = cmd_init
                self._send(200, "application/json", b'{"ok":true}')
                return

            self._send(404, "text/plain", b"not found")

        # quiet logs
        def log_message(self, format, *args):
            return

    httpd = HTTPServer((host, port), Handler)
    print(f"[cmd web ui] Open: http://{host}:{port}")
    httpd.serve_forever()

def rgb_u8_to_jpeg_bytes(rgb_u8: np.ndarray, quality: int = 80) -> bytes:
    """
    Encode (H,W,3) uint8 RGB to JPEG bytes.
    Requires Pillow: pip install pillow
    """
    from PIL import Image
    import io
    im = Image.fromarray(rgb_u8, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()

from lidar_livox_mid360 import LivoxMid360Sim
from camera_d435i import D435iDepthSim

def draw_multiple_world_point_sets(
    viewer,
    m,
    d,
    *,
    # LiDAR-style input (points in site frame)
    lidar_site_id: int | None = None,
    lidar_points_site: np.ndarray | None = None,
    lidar_rgba=(0.0, 1.0, 0.0, 1.0),
    lidar_radius: float = 0.0025,
    lidar_max: int = 1000,

    # Camera-style input (points already in world frame)
    cam_points_world: np.ndarray | None = None,
    cam_colors_rgb: np.ndarray | None = None,   # (N,3) uint8 or float
    cam_radius: float = 0.005,
    cam_alpha: float = 1.0,
    cam_max: int = 1000,

    # Global cap (must not exceed scn.maxgeom)
    maxgeom_cap: int | None = None,
):
    """
    Draw multiple point sets into viewer.user_scn in one pass (no overwriting).

    - LiDAR points are assumed to be in the LiDAR SITE frame and will be transformed to world.
    - Camera points are assumed to be already in WORLD coordinates.
    - Colors for camera points are per-point RGB; LiDAR is a constant RGBA.

    Args:
      viewer: mujoco.viewer viewer object
      m, d: model/data (needed for site transform)
      lidar_site_id: MuJoCo site id for LiDAR
      lidar_points_site: (N,3) points in site frame
      cam_points_world: (N,3) points in world frame
      cam_colors_rgb: (N,3) RGB colors (uint8 0..255 or float 0..1)
      maxgeom_cap: optional hard limit on markers drawn (<= scn.maxgeom)
    """
    if viewer is None:
        return

    # Prepare LiDAR world points (subsampled)
    lidar_pts_w = None
    if lidar_site_id is not None and lidar_points_site is not None:
        pts = np.asarray(lidar_points_site)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
            n = pts.shape[0]
            step = max(1, n // int(lidar_max))
            pts_s = pts[::step][:int(lidar_max)]
            p_w = d.site_xpos[lidar_site_id].copy()
            R_w_s = d.site_xmat[lidar_site_id].reshape(3, 3).copy()  # site->world
            lidar_pts_w = p_w[None, :] + (pts_s @ R_w_s.T)

    # Prepare camera points (subsampled) + colors (aligned)
    cam_pts_w = None
    cam_cols = None
    if cam_points_world is not None:
        pts = np.asarray(cam_points_world)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
            n = pts.shape[0]
            if n > int(cam_max):
                idx = np.linspace(0, n - 1, num=int(cam_max), dtype=np.int32)
                cam_pts_w = pts[idx]
                if cam_colors_rgb is not None:
                    cam_cols = np.asarray(cam_colors_rgb)[idx]
            else:
                cam_pts_w = pts
                if cam_colors_rgb is not None:
                    cam_cols = np.asarray(cam_colors_rgb)

            if cam_cols is not None:
                if cam_cols.ndim != 2 or cam_cols.shape[1] != 3 or cam_cols.shape[0] != cam_pts_w.shape[0]:
                    cam_cols = None
                else:
                    cam_cols = cam_cols.astype(np.float32)
                    if cam_cols.max() > 1.0:
                        cam_cols /= 255.0
                    cam_cols = np.clip(cam_cols, 0.0, 1.0)

    # If nothing to draw, clear overlay
    
    if lidar_pts_w is None and cam_pts_w is None:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
        return

    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0

        # Determine capacity
        capacity = scn.maxgeom
        if maxgeom_cap is not None:
            capacity = min(capacity, int(maxgeom_cap))

        mat = np.eye(3, dtype=np.float64).reshape(-1)

        geom_i = 0

        # Draw LiDAR first (constant color)
        if lidar_pts_w is not None:
            rgba = np.array(lidar_rgba, dtype=np.float64)
            M = min(lidar_pts_w.shape[0], capacity - geom_i)
            for i in range(M):
                g = scn.geoms[geom_i]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([float(lidar_radius), 0, 0], dtype=np.float64),
                    lidar_pts_w[i].astype(np.float64),
                    mat,
                    rgba,
                )
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                geom_i += 1
                if geom_i >= capacity:
                    scn.ngeom = geom_i
                    return

        # Draw camera points (per-point RGB)
        if cam_pts_w is not None:
            M = min(cam_pts_w.shape[0], capacity - geom_i)
            for i in range(M):
                rgba = np.array([1.0, 1.0, 1.0, float(cam_alpha)], dtype=np.float64)
                if cam_cols is not None:
                    rgba[:3] = cam_cols[i].astype(np.float64)

                g = scn.geoms[geom_i]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([float(cam_radius), 0, 0], dtype=np.float64),
                    cam_pts_w[i].astype(np.float64),
                    mat,
                    rgba,
                )
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                geom_i += 1
                if geom_i >= capacity:
                    scn.ngeom = geom_i
                    return

        scn.ngeom = geom_i


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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument(
        "--mapping-mode",
        action="store_true",
        help="Enable mapping-oriented runtime defaults without changing legacy behavior",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation loop without MuJoCo GUI viewer",
    )
    parser.add_argument(
        "--web-ui",
        action="store_true",
        help="Enable command web UI in mapping mode (legacy mode already enables it by default)",
    )
    # Sensor visualization controls (MuJoCo overlay markers)
    parser.add_argument(
        "--show-sensors",
        action="store_true",
        help="Draw sensor outputs (LiDAR + D435 point clouds) in MuJoCo viewer overlay",
    )
    parser.add_argument(
        "--profile-lidar",
        action="store_true",
        help="Print LiDAR timing counters once per second",
    )
    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Print unified runtime profile (RTF + ROS/camera timings) once per second",
    )
    parser.add_argument("--ros-clock-hz", type=float, default=None, help="Max /clock publish rate (Hz)")
    parser.add_argument("--ros-tf-hz", type=float, default=None, help="Max /tf publish rate (Hz)")
    parser.add_argument("--ros-odom-hz", type=float, default=None, help="Max /odom publish rate (Hz)")
    parser.add_argument("--ros-imu-hz", type=float, default=None, help="Max /imu publish rate (Hz)")
    parser.add_argument("--ros-joint-hz", type=float, default=None, help="Max /joint_states publish rate (Hz)")
    args = parser.parse_args()

    if args.show_sensors and args.headless:
        warn("--show-sensors has no effect in --headless mode.")

    ros_clock_hz = args.ros_clock_hz
    ros_tf_hz = args.ros_tf_hz
    ros_odom_hz = args.ros_odom_hz
    ros_imu_hz = args.ros_imu_hz
    ros_joint_hz = args.ros_joint_hz

    if args.mapping_mode:
        if ros_clock_hz is None:
            ros_clock_hz = 50.0
        if ros_tf_hz is None:
            ros_tf_hz = 30.0
        if ros_odom_hz is None:
            ros_odom_hz = 30.0
        if ros_imu_hz is None:
            ros_imu_hz = 100.0
        if ros_joint_hz is None:
            ros_joint_hz = 20.0

    # Keep legacy default behavior unchanged: web UI on in legacy mode.
    enable_web_ui = (not args.mapping_mode) or bool(args.web_ui)

    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd_lock = threading.Lock()
        cmd_init = config["cmd_init"]
        cmd_shared = cmd_init.copy()

        rgb_lock = threading.Lock()
        rgb_jpeg_shared = {
            "jpeg": b"",     # latest JPEG bytes
            "t": 0.0,        # wall time of last update
            "w": 0,
            "h": 0,
        }

        if enable_web_ui:
            web_thread = threading.Thread(
                target=start_cmd_web_ui,
                args=(cmd_shared, cmd_lock, cmd_init, rgb_jpeg_shared, rgb_lock),
                kwargs=dict(host="127.0.0.1", port=8000, slider_min=-10.0, slider_max=10.0),
                daemon=True,
            )
            web_thread.start()
        else:
            print("[cmd web ui] Disabled for this run")

    if not hasattr(mujoco, "mj_multiRay"):
        warn("No mj_multiRay capability, this run will be much slower.")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # LiDAR    
    lidar = LivoxMid360Sim(
        m,
        site_name="livox_mid360",
        frame_rate_hz=10.0,
        points_per_second=200_000,
        max_points_per_frame=800,   # start smaller for speed; raise once stable
        range_max=30.0,              # indoor-ish cap; raise if needed
        range_noise_sigma=0.02,
        output_frame="site"
    )

    # Depth Camera
    d435 = D435iDepthSim(
        m, d,
        camera_name="d435i_depth_cam",
        mount_site_name="d435i_mount",
        width=640, height=480,
        fps=30.0,
        z_near=0.28,
        z_far=3.0,
        depth_noise_sigma_m=0.002,
        output_pointcloud=True,
        output_frame="site",
        mount_roll_deg=0.0,
        mount_pitch_deg=-45.0,
        mount_yaw_deg=0.0,
        raycast_stride=1,
        depth_generation_mode="render_fast" if args.mapping_mode else "raycast",
        profile_enabled=args.profile_runtime,
    )

    # ---------------------------
    # ROBUST DOF INDEXING FIX
    # ---------------------------
    # Map actuators -> joints -> qpos/qvel indices so extra joints (e.g., freejoint props)
    # do NOT break our assumptions.
    #
    # - actuator_trnid: (nu, 2) array; first column is joint id
    # - jnt_qposadr: index into qpos for that joint
    # - jnt_dofadr:  index into qvel for that joint
    act_joint_ids = m.actuator_trnid[:, 0].copy()
    qpos_adr = np.array([m.jnt_qposadr[jid] for jid in act_joint_ids], dtype=int)
    qvel_adr = np.array([m.jnt_dofadr[jid] for jid in act_joint_ids], dtype=int)

    # Optional sanity checks (won’t stop execution unless you want it to)
    if qpos_adr.shape[0] != num_actions or qvel_adr.shape[0] != num_actions:
        print(
            f"[WARN] actuator count ({qpos_adr.shape[0]}) != num_actions ({num_actions}). "
            "This may indicate extra actuators or a mismatch in config."
        )

    # load policy
    policy = torch.jit.load(policy_path)

    last_lidar_pts_site = None  # lidar output_frame="site"
    last_cam_pts_world = None
    last_cam_cols = None
    lidar_profile_last_wall = time.perf_counter()
    runtime_profile_last_wall = time.perf_counter()
    runtime_profile = {
        "steps": 0,
        "sim_s": 0.0,
        "wall_s": 0.0,
        "spin_ros_s": 0.0,
        "control_s": 0.0,
        "mj_step_s": 0.0,
        "viewer_s": 0.0,
        "bridge_pub_s": 0.0,
        "lidar_s": 0.0,
        "lidar_ros_s": 0.0,
        "camera_step_s": 0.0,
        "camera_ros_s": 0.0,
        "jpeg_s": 0.0,
        "draw_s": 0.0,
        "sleep_s": 0.0,
    }

    livox_node = None
    d435_node = None
    if ROS2_ENABLED:
        rclpy.init(args=None)
        ros_bridge = MujocoROS2Bridge(
            m,
            d,
            cmd_vel_topic="/unitree/cmd_vel",
            cmd_lock=cmd_lock,
            cmd_shared=cmd_shared,
            clock_hz=ros_clock_hz,
            tf_hz=ros_tf_hz,
            odom_hz=ros_odom_hz,
            imu_hz=ros_imu_hz,
            joint_hz=ros_joint_hz,
            profile_enabled=args.profile_runtime,
        )
        livox_node = LivoxPublisher(m, d)
        d435_node = D435iPublisher(m, d)
        print("[ROS2] Publishing /clock, /tf, /tf_static, /joint_states, /odom, /imu")
        print("[ROS2] Publishing /livox/points (sensor_msgs/PointCloud2)")
        print("[ROS2] Subscribed /unitree/cmd_vel (geometry_msgs/Twist -> [vx, vy, yaw_rate])")
        print("[ROS2] Publishing D435i topics:")
        print("  /intel/D435i/color (sensor_msgs/Image rgb8)")
        print("  /intel/D435i/depth (sensor_msgs/Image 16UC1, mm)")
        print("  /intel/D435i/aligned_depth_to_color (sensor_msgs/Image 16UC1, mm)")
        print("  + camera_info and TF frames from MJCF names")
    else:
        ros_bridge = None
        print("[ROS2] rclpy not available; ROS publishing")

    def maybe_print_runtime_profile(now_wall: float):
        global runtime_profile_last_wall
        if not args.profile_runtime:
            return
        if now_wall - runtime_profile_last_wall < 1.0:
            return

        wall = max(runtime_profile["wall_s"], 1e-9)
        steps = max(runtime_profile["steps"], 1)
        rtf = runtime_profile["sim_s"] / wall

        print(
            "[Runtime profile] "
            f"wall_window={wall:.3f}s sim_window={runtime_profile['sim_s']:.3f}s "
            f"rtf={rtf:.3f} steps={runtime_profile['steps']} "
            f"ms/step spin_ros={1e3*runtime_profile['spin_ros_s']/steps:.3f} "
            f"control={1e3*runtime_profile['control_s']/steps:.3f} "
            f"mj_step={1e3*runtime_profile['mj_step_s']/steps:.3f} "
            f"viewer={1e3*runtime_profile['viewer_s']/steps:.3f} "
            f"bridge={1e3*runtime_profile['bridge_pub_s']/steps:.3f} "
            f"lidar={1e3*runtime_profile['lidar_s']/steps:.3f} "
            f"lidar_ros={1e3*runtime_profile['lidar_ros_s']/steps:.3f} "
            f"camera={1e3*runtime_profile['camera_step_s']/steps:.3f} "
            f"camera_ros={1e3*runtime_profile['camera_ros_s']/steps:.3f} "
            f"jpeg={1e3*runtime_profile['jpeg_s']/steps:.3f} "
            f"draw={1e3*runtime_profile['draw_s']/steps:.3f} "
            f"sleep={1e3*runtime_profile['sleep_s']/steps:.3f}"
        )

        if d435 is not None:
            cam_stats = d435.get_stats(reset=True)
            print(
                "[D435 profile] "
                f"mode={cam_stats['depth_mode']} "
                f"sim_window={cam_stats['sim_time_s']:.3f}s frames={cam_stats['frames_emitted']} "
                f"points_per_sec={cam_stats['points_per_sec']:.1f} "
                f"pts/frame={cam_stats['points_per_frame_avg']:.1f} "
                f"ms/frame depth={cam_stats['render_depth_ms_per_frame']:.3f} "
                f"rgb={cam_stats['render_rgb_ms_per_frame']:.3f} "
                f"depth_model={cam_stats['depth_model_ms_per_frame']:.3f} "
                f"depth_to_pc={cam_stats['depth_to_pc_ms_per_frame']:.3f} "
                f"world_xform={cam_stats['world_transform_ms_per_frame']:.3f} "
                f"total={cam_stats['total_ms_per_frame']:.3f}"
            )

        if ros_bridge is not None:
            bridge_stats = ros_bridge.get_stats(reset=True)
            print(
                "[ROS bridge profile] "
                f"sim_window={bridge_stats['sim_time_s']:.3f}s calls={bridge_stats['publish_calls']} "
                f"rate/s clock={bridge_stats['clock_msgs_per_sec']:.1f} "
                f"tf={bridge_stats['tf_msgs_per_sec']:.1f} "
                f"odom={bridge_stats['odom_msgs_per_sec']:.1f} "
                f"imu={bridge_stats['imu_msgs_per_sec']:.1f} "
                f"joint={bridge_stats['joint_msgs_per_sec']:.1f} "
                f"ms/msg clock={bridge_stats['clock_ms_per_msg']:.3f} "
                f"tf={bridge_stats['tf_ms_per_msg']:.3f} "
                f"odom={bridge_stats['odom_ms_per_msg']:.3f} "
                f"imu={bridge_stats['imu_ms_per_msg']:.3f} "
                f"joint={bridge_stats['joint_ms_per_msg']:.3f}"
            )

        for key in runtime_profile:
            runtime_profile[key] = 0.0
        runtime_profile["steps"] = 0
        runtime_profile_last_wall = now_wall

    def run_loop(viewer=None):
        global counter, action, target_dof_pos, obs
        global last_lidar_pts_site, last_cam_pts_world, last_cam_cols, lidar_profile_last_wall

        sim_time = 0.0
        start = time.time()
        while (viewer is None or viewer.is_running()) and time.time() - start < simulation_duration:
            loop_wall_t0 = time.perf_counter()
            step_start = time.time()

            if ros_bridge is not None:
                t0 = time.perf_counter()
                rclpy.spin_once(ros_bridge, timeout_sec=0.0)
                ros_bridge.enforce_cmd_vel_timeout()
                runtime_profile["spin_ros_s"] += time.perf_counter() - t0

            # --- Robust joint state extraction for PD control ---
            t0 = time.perf_counter()
            qj_raw = d.qpos[qpos_adr]
            dqj_raw = d.qvel[qvel_adr]
            tau = pd_control(
                target_dof_pos,
                qj_raw,
                kps,
                np.zeros_like(kds),
                dqj_raw,
                kds,
            )
            d.ctrl[:] = tau
            runtime_profile["control_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            mujoco.mj_step(m, d)
            runtime_profile["mj_step_s"] += time.perf_counter() - t0
            sim_time += m.opt.timestep

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
                qj = d.qpos[qpos_adr].copy()
                dqj = d.qvel[qvel_adr].copy()
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                with cmd_lock:
                    cmd = cmd_shared.copy()
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            if viewer is not None:
                t0 = time.perf_counter()
                viewer.sync()
                runtime_profile["viewer_s"] += time.perf_counter() - t0

            sim_stamp = None
            if ros_bridge is not None:
                sim_stamp = ros_bridge.get_clock().now().to_msg()
                sec, nsec = sim_time_to_sec_nsec(sim_time)
                sim_stamp.sec = sec
                sim_stamp.nanosec = nsec

                t0 = time.perf_counter()
                ros_bridge.publish_step(m.opt.timestep, stamp_msg=sim_stamp, sim_time=sim_time)
                runtime_profile["bridge_pub_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            cloud = lidar.step(d, dt=m.opt.timestep)
            runtime_profile["lidar_s"] += time.perf_counter() - t0
            if cloud is not None:
                last_lidar_pts_site = cloud  # (N,3) in site frame
                if livox_node is not None:
                    t1 = time.perf_counter()
                    stamp = sim_stamp if sim_stamp is not None else livox_node.get_clock().now().to_msg()
                    livox_node.publish_tf(stamp)

                    frame_id = "livox_mid360"
                    pack_t0 = time.perf_counter()
                    msg = pointcloud2_from_xyz(
                        cloud,
                        frame_id=frame_id,
                        stamp_msg=stamp,
                        intensity=None,
                    )
                    lidar.record_pack_time(time.perf_counter() - pack_t0)
                    livox_node.pub.publish(msg)
                    runtime_profile["lidar_ros_s"] += time.perf_counter() - t1

            t0 = time.perf_counter()
            frame = d435.step(dt=m.opt.timestep)
            runtime_profile["camera_step_s"] += time.perf_counter() - t0
            if frame is not None and frame.get("pointcloud_world") is not None:
                last_cam_pts_world = frame["pointcloud_world"]
                last_cam_cols = frame.get("point_colors_rgb", None)
                if enable_web_ui and frame.get("rgb_u8") is not None:
                    t1 = time.perf_counter()
                    try:
                        jpg = rgb_u8_to_jpeg_bytes(frame["rgb_u8"], quality=80)
                        with rgb_lock:
                            rgb_jpeg_shared["jpeg"] = jpg
                            rgb_jpeg_shared["t"] = time.time()
                            rgb_jpeg_shared["h"], rgb_jpeg_shared["w"] = frame["rgb_u8"].shape[:2]
                    except Exception as e:
                        print(e)
                    runtime_profile["jpeg_s"] += time.perf_counter() - t1
            if frame is not None and d435_node is not None:
                t1 = time.perf_counter()
                d435_node.publish_frames_and_images(frame, stamp_msg=sim_stamp)
                runtime_profile["camera_ros_s"] += time.perf_counter() - t1

            if args.show_sensors and viewer is not None:
                t0 = time.perf_counter()
                draw_multiple_world_point_sets(
                    viewer, m, d,
                    lidar_site_id=lidar.site_id,
                    lidar_points_site=last_lidar_pts_site if last_lidar_pts_site is not None else None,
                    lidar_radius=0.005,
                    lidar_max=1000,
                    cam_points_world=last_cam_pts_world if last_cam_pts_world is not None else None,
                    cam_colors_rgb=last_cam_cols if last_cam_cols is not None else None,
                    cam_radius=0.01,
                    cam_alpha=1.0,
                    cam_max=2400,
                )
                runtime_profile["draw_s"] += time.perf_counter() - t0

            if args.profile_lidar:
                now_wall = time.perf_counter()
                if now_wall - lidar_profile_last_wall >= 1.0:
                    stats = lidar.get_stats(reset=True)
                    print(
                        "[LiDAR profile] "
                        f"sim_window={stats['sim_time_s']:.3f}s "
                        f"frames={stats['frames_emitted']} "
                        f"rays={stats['rays_cast']} ({stats['rays_per_sec']:.1f}/s) "
                        f"points={stats['points_kept']} ({stats['points_per_sec']:.1f}/s, "
                        f"{stats['points_per_frame_avg']:.1f}/frame) "
                        f"ms/frame sample={stats['sample_ms_per_frame']:.3f} "
                        f"transform={stats['transform_ms_per_frame']:.3f} "
                        f"raycast={stats['raycast_ms_per_frame']:.3f} "
                        f"post={stats['postprocess_ms_per_frame']:.3f} "
                        f"pack={stats['pack_ms_per_frame']:.3f}"
                    )
                    lidar_profile_last_wall = now_wall

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                sleep_t0 = time.perf_counter()
                time.sleep(time_until_next_step)
                runtime_profile["sleep_s"] += time.perf_counter() - sleep_t0

            now = time.perf_counter()
            runtime_profile["steps"] += 1
            runtime_profile["sim_s"] += m.opt.timestep
            runtime_profile["wall_s"] += now - loop_wall_t0
            maybe_print_runtime_profile(now)

    if args.headless:
        run_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            run_loop(viewer=viewer)

    if ros_bridge is not None or livox_node is not None or d435_node is not None:
        if ros_bridge is not None:
            ros_bridge.destroy_node()
        if livox_node is not None:
            livox_node.destroy_node()
        if d435_node is not None:
            d435_node.destroy_node()
        rclpy.shutdown()
