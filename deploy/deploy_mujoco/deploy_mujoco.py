import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

def start_cmd_web_ui(cmd_shared: np.ndarray, cmd_lock: threading.Lock, cmd_init: np.ndarray,
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


from lidar_livox_mid360 import LivoxMid360Sim

def draw_pointcloud_markers(viewer, m, d, lidar_site_id, cloud_lidar,
                            max_markers=500, sphere_radius=0.01):
    """
    Draw cloud points as spheres in the viewer overlay (viewer.user_scn).
    cloud_lidar: (N,3) points in LiDAR local frame.
    """
    if viewer is None or cloud_lidar is None:
        return

    # Subsample to keep it fast
    n = cloud_lidar.shape[0]
    if n == 0:
        with viewer.lock():
            viewer.user_scn.ngeom = 0
        return

    step = max(1, n // max_markers)
    pts_l = cloud_lidar[::step][:max_markers]  # (M,3)

    # LiDAR pose in world
    p_w = d.site_xpos[lidar_site_id].copy()                    # (3,)
    R_w_l = d.site_xmat[lidar_site_id].reshape(3, 3).copy()    # lidar->world

    # Transform to world: p_world = p_w + R_w_l @ p_lidar
    pts_w = p_w[None, :] + (pts_l @ R_w_l.T)                   # (M,3)

    # Populate overlay geoms
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0

        # Safety: user_scn has a max capacity
        M = min(pts_w.shape[0], scn.maxgeom)

        for i in range(M):
            g = scn.geoms[i]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([sphere_radius, 0, 0], dtype=np.float64),
                pts_w[i].astype(np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),  # green
            )
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            scn.ngeom += 1


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

        web_thread = threading.Thread(
            target=start_cmd_web_ui,
            args=(cmd_shared, cmd_lock, cmd_init),
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
                # "see" it: print summary + a few points
                draw_pointcloud_markers(viewer, m, d, lidar.site_id, cloud,
                            max_markers=1000, sphere_radius=0.005)


            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
