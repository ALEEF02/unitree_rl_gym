import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import struct

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

ROS2_ENABLED = False
try:
    import rclpy
    from rclpy.node import Node
    ROS2_ENABLED = True
except Exception:
    ROS2_ENABLED = False


class LivoxPublisher(Node):
    def __init__(self):
        super().__init__("livox_mid360_sim")
        # Use SensorDataQoS so RViz / typical pipelines behave well for live sensors
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        from sensor_msgs.msg import PointCloud2
        self.pub = self.create_publisher(PointCloud2, "/livox/points", qos)

class D435iPublisher(Node):
    def __init__(self):
        super().__init__("d435i_sim")

        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )

        from sensor_msgs.msg import Image

        self.pub_color = self.create_publisher(Image, "/intel/D435i/color", qos)
        self.pub_depth = self.create_publisher(Image, "/intel/D435i/depth", qos)
        self.pub_aligned = self.create_publisher(Image, "/intel/D435i/aligned_depth_to_color", qos)

        # Optional but recommended for Nav2 / SLAM:
        from sensor_msgs.msg import CameraInfo
        self.pub_color_info = self.create_publisher(CameraInfo, "/intel/D435i/color/camera_info", qos)
        self.pub_depth_info = self.create_publisher(CameraInfo, "/intel/D435i/depth/camera_info", qos)
        self.pub_aligned_info = self.create_publisher(CameraInfo, "/intel/D435i/aligned_depth_to_color/camera_info", qos)

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
        inten = np.zeros((n,), dtype=np.float32)
    else:
        inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if inten.shape[0] != n:
            raise ValueError(f"intensity must have length {n}, got {inten.shape[0]}")

    # Pack as little-endian float32: x y z intensity
    # point_step = 16 bytes
    data = bytearray(n * 16)
    off = 0
    pack = struct.Struct("<ffff").pack
    for i in range(n):
        data[off:off+16] = pack(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]), float(inten[i]))
        off += 16

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
    msg.data = bytes(data)
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
    args = parser.parse_args()
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

        web_thread = threading.Thread(
            target=start_cmd_web_ui,
            args=(cmd_shared, cmd_lock, cmd_init, rgb_jpeg_shared, rgb_lock),
            kwargs=dict(host="127.0.0.1", port=8000, slider_min=-10.0, slider_max=10.0),
            daemon=True,
        )
        web_thread.start()


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
        raycast_stride=8,
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

    livox_node = None
    d435_node = None
    if ROS2_ENABLED:
        rclpy.init(args=None)
        livox_node = LivoxPublisher()
        d435_node = D435iPublisher()
        print("[ROS2] Publishing /livox/points (sensor_msgs/PointCloud2)")
        print("[ROS2] Publishing D435i topics:")
        print("  /intel/D435i/color (sensor_msgs/Image rgb8)")
        print("  /intel/D435i/depth (sensor_msgs/Image 16UC1, mm)")
        print("  /intel/D435i/aligned_depth_to_color (sensor_msgs/Image 16UC1, mm)")
    else:
        print("[ROS2] rclpy not available; skipping /livox/points publishing")
        print("[ROS2] rclpy not available; skipping D435i ROS publishing")


    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # --- Robust joint state extraction for PD control ---
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

            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # --- Robust joint state extraction for observations ---
                qj = d.qpos[qpos_adr].copy()
                dqj = d.qvel[qvel_adr].copy()

                # Base orientation and angular velocity are still taken from the floating base.
                # (Assumes the robot root is the first free body in the model.)
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
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            cloud = lidar.step(d, dt=m.opt.timestep)
            if cloud is not None:
                last_lidar_pts_site = cloud  # (N,3) in site frame
                if livox_node is not None:
                    # Decide what frame_id should be:
                    # - If lidar.output_frame == "sensor": frame_id like "livox_frame"
                    # - If lidar.output_frame == "site":   frame_id like "livox_mid360" (site frame)
                    frame_id = "livox_mid360"  # choose a stable frame name; match your TF later

                    msg = pointcloud2_from_xyz(
                        cloud,  # (N,3)
                        frame_id=frame_id,
                        stamp_msg=livox_node.get_clock().now().to_msg(),
                        intensity=None,  # or np.ones((cloud.shape[0],), np.float32)
                    )
                    livox_node.pub.publish(msg)

                    # Keep ROS2 responsive without blocking your sim
                    rclpy.spin_once(livox_node, timeout_sec=0.0)

            frame = d435.step(dt=m.opt.timestep)
            if frame is not None and frame.get("pointcloud_world") is not None:
                last_cam_pts_world = frame["pointcloud_world"]
                last_cam_cols = frame.get("point_colors_rgb", None)
                if frame.get("rgb_u8") is not None:
                    try:
                        jpg = rgb_u8_to_jpeg_bytes(frame["rgb_u8"], quality=80)
                        with rgb_lock:
                            rgb_jpeg_shared["jpeg"] = jpg
                            rgb_jpeg_shared["t"] = time.time()
                            rgb_jpeg_shared["h"], rgb_jpeg_shared["w"] = frame["rgb_u8"].shape[:2]
                    except Exception as e:
                        print(e)
                        pass
            if frame is not None and d435_node is not None:
                # Choose a stable TF frame for these images:
                # For RealSense convention you might eventually use: "D435i_color_optical_frame"
                frame_id = "D435i_color_optical_frame"

                stamp = d435_node.get_clock().now().to_msg()

                # Color
                if frame.get("rgb_u8") is not None:
                    msg_color = ros_image_from_numpy(frame["rgb_u8"], frame_id=frame_id, stamp_msg=stamp, encoding="rgb8")
                    d435_node.pub_color.publish(msg_color)

                # Depth (Z16-style millimeters)
                if frame.get("depth_mm_u16") is not None:
                    msg_depth = ros_image_from_numpy(frame["depth_mm_u16"], frame_id=frame_id, stamp_msg=stamp, encoding="16UC1")
                    d435_node.pub_depth.publish(msg_depth)

                    # In your current sim, depth is already aligned to the rendered RGB view.
                    # Publish the same image for aligned_depth_to_color.
                    d435_node.pub_aligned.publish(msg_depth)

                # Keep ROS2 responsive
                rclpy.spin_once(d435_node, timeout_sec=0.0)


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

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if livox_node is not None:
        livox_node.destroy_node()
        rclpy.shutdown()

    if d435_node is not None:
        d435_node.destroy_node()
        rclpy.shutdown()