import time
import math
import numpy as np
import mujoco

# Some versions support render(depth=True)
import inspect

class D435iDepthSim:
    """
    D435i-like depth camera simulation using MuJoCo offscreen rendering.

    Outputs (by default) a uint16 depth image in millimeters similar to RealSense depth output,
    plus a point cloud computed from intrinsics (optional).

    This is meant to be a SWAP-IN backend: later you can replace the render step with pyrealsense2
    and keep the same output contract.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str = "d435i_depth_cam",
        mount_site_name: str = "d435i_mount",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0, # These are the values used when aligning depth to RGB

        # Intrinsics: you should replace these with the actual intrinsics you read from the real D435i profile later.
        # The defaults below are "reasonable" for a 640x480 pinhole model; don't treat them as exact hardware values.
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,

        # Depth range & noise
        z_near: float = 0.28,
        z_far: float = 3.0,
        depth_noise_sigma_m: float = 0.002,   # ~2mm baseline noise (tunable)
        dropout_prob: float = 0.0,

        # Output
        output_pointcloud: bool = True,
        max_points: int = 200_000,            # safety cap for dense clouds
        output_frame: str = "camera",         # "camera" | "world"
        seed: int = 1,
        rendered_depth_is_range: bool = True,

        # Mounting offset (like we did for LiDAR): sensor frame -> site frame
        mount_roll_deg: float = 0.0,
        mount_pitch_deg: float = 0.0,
        mount_yaw_deg: float = 0.0,

        raycast_stride: int = 2,             # cast every Nth pixel (2 => 320x240 rays)
    ):
        self.m = model
        self.d = data
        self.cam_name = camera_name

        self.cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in MJCF.")
                # Save the camera's base local pose from the MJCF (relative to its parent body)
        self._cam_pos0 = self.m.cam_pos[self.cam_id].copy()
        self._cam_quat0 = self.m.cam_quat[self.cam_id].copy()

        self.site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, mount_site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{mount_site_name}' not found in MJCF.")

        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame_dt = 1.0 / self.fps
        self._accum_t = 0.0

        # Intrinsics defaults: derive from MuJoCo camera fovy so unprojection matches rendering
        fovy_deg = float(self.m.cam_fovy[self.cam_id])  # vertical FOV in degrees
        fovy = math.radians(fovy_deg)
        fy_default = 0.5 * self.height / math.tan(0.5 * fovy)
        fx_default = fy_default * (self.width / self.height)

        if fx is None: fx = fx_default
        if fy is None: fy = fy_default
        if cx is None: cx = (self.width - 1) * 0.5
        if cy is None: cy = (self.height - 1) * 0.5

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        # RealSense optical frame: +X right, +Y down, +Z forward
        # MuJoCo camera frame (OpenGL-like): +X right, +Y up, -Z forward
        # Mapping optical -> mujoco_cam: x stays, y flips, z flips
        self.R_mjcam_optical = np.array([[1, 0, 0],
                                        [0,-1, 0],
                                        [0, 0,-1]], dtype=np.float64)

        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.depth_noise_sigma_m = float(depth_noise_sigma_m)
        self.dropout_prob = float(dropout_prob)

        self.raycast_stride = int(raycast_stride)
        if self.raycast_stride < 1:
            self.raycast_stride = 1

        self.output_pointcloud = bool(output_pointcloud)
        self.max_points = int(max_points)
        self.output_frame = output_frame.strip().lower()
        if self.output_frame not in ("camera", "site", "world"):
            raise ValueError("output_frame must be 'camera', 'site', or 'world'")

        self.rng = np.random.default_rng(seed)
        self.rendered_depth_is_range = rendered_depth_is_range

        # Mount rotation: SENSOR(camera) -> SITE
        self.R_site_cam = self._rotmat_zyx_deg(mount_yaw_deg, mount_pitch_deg, mount_roll_deg)

        # Renderer (offscreen)
        self.renderer = mujoco.Renderer(self.m, height=self.height, width=self.width)

        # Probe renderer API differences across mujoco versions
        self._renderer_has_depth_kw = False
        self._renderer_can_enable_depth = False

        # Probe renderer API differences
        try:
            sig = inspect.signature(self.renderer.render)
            self._renderer_has_depth_kw = ("depth" in sig.parameters)
        except Exception:
            self._renderer_has_depth_kw = False

        self._renderer_has_enable_depth = hasattr(self.renderer, "enable_depth_rendering")
        self._renderer_has_disable_depth = hasattr(self.renderer, "disable_depth_rendering")

        # Track current mode if we need to toggle
        self._depth_enabled = False

        if self._renderer_has_enable_depth:
            # We'll toggle as needed
            self._depth_enabled = False

        # Precompute pixel grid for fast pointcloud unprojection
        u = np.arange(self.width, dtype=np.float32)
        v = np.arange(self.height, dtype=np.float32)
        self._uu, self._vv = np.meshgrid(u, v)  # (H,W)

        # Precompute per-pixel factor to convert range-along-ray -> z-depth
        # If a pixel ray direction in optical frame is [x, y, 1] (unnormalized),
        # unit ray z-component is 1 / sqrt(x^2 + y^2 + 1).
        x = (self._uu - self.cx) / self.fx
        y = (self._vv - self.cy) / self.fy
        self._ray_unit_z = (1.0 / np.sqrt(x*x + y*y + 1.0)).astype(np.float32)  # (H,W)


    @staticmethod
    def _rot_x(a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0],
                         [0, ca, -sa],
                         [0, sa,  ca]], dtype=np.float64)

    @staticmethod
    def _rot_y(a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[ ca, 0, sa],
                         [  0, 1,  0],
                         [-sa, 0, ca]], dtype=np.float64)

    @staticmethod
    def _rot_z(a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[ca, -sa, 0],
                         [sa,  ca, 0],
                         [ 0,   0, 1]], dtype=np.float64)

    @classmethod
    def _rotmat_zyx_deg(cls, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        return cls._rot_z(yaw) @ cls._rot_y(pitch) @ cls._rot_x(roll)

    def _quat_to_mat(self, q: np.ndarray) -> np.ndarray:
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, q.astype(np.float64))
        return mat.reshape(3, 3)

    def _mat_to_quat(self, R: np.ndarray) -> np.ndarray:
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
        return q

    def _apply_mount_to_model_camera(self):
        """
        Applies mount rotation to the *rendered camera orientation* by updating model.cam_quat
        and then recomputing kinematics so rendering uses the new orientation.
        """
        # Base camera local rotation from MJCF
        R_base = self._quat_to_mat(self._cam_quat0)

        # Apply mount rotation in camera local axes
        R_new = R_base @ self.R_site_cam

        self.m.cam_quat[self.cam_id] = self._mat_to_quat(R_new)
        self.m.cam_pos[self.cam_id] = self._cam_pos0

        # CRITICAL: recompute derived quantities (cam_xmat/xpos) after modifying model
        mujoco.mj_forward(self.m, self.d)

    def _site_pose_world(self):
        # site->world
        p = self.d.site_xpos[self.site_id].copy()
        R = self.d.site_xmat[self.site_id].reshape(3, 3).copy()
        return p, R

    def _depthbuf_to_meters(self, zbuf: np.ndarray) -> np.ndarray:
        """
        Convert OpenGL-style normalized depth buffer z in [0,1] to metric depth (meters)
        using the near/far planes we use for clipping.

        If your renderer already returns meters, you should not call this.
        """
        z = zbuf.astype(np.float32)

        # Guard: avoid div-by-zero and invalid z
        z = np.clip(z, 1e-6, 1.0 - 1e-6)

        n = float(self.z_near)
        f = float(self.z_far)

        # Standard OpenGL perspective depth mapping
        # z_ndc in [-1, 1]
        z_ndc = 2.0 * z - 1.0
        depth = (2.0 * n * f) / (f + n - z_ndc * (f - n))
        return depth.astype(np.float32)

    def _render_depth_m(self) -> np.ndarray:
        self._apply_mount_to_model_camera()
        self.renderer.update_scene(self.d, camera=self.cam_name)

        if self._renderer_has_depth_kw:
            depth = np.asarray(self.renderer.render(depth=True), dtype=np.float32)
        else:
            self._set_depth_mode(True)
            depth = np.asarray(self.renderer.render(), dtype=np.float32)

        # Heuristic: if depth looks like a normalized buffer, convert it
        # (Most normalized buffers are in [0,1]. Metric depth will usually exceed 1.0 sometimes.)
        if np.isfinite(depth).any():
            dmax = float(np.nanmax(depth))
            dmin = float(np.nanmin(depth))
            if 0.0 <= dmin and dmax <= 1.0:
                depth = self._depthbuf_to_meters(depth)

        return depth

    def _render_rgb_u8(self) -> np.ndarray:
        self._apply_mount_to_model_camera()
        self.renderer.update_scene(self.d, camera=self.cam_name)

        if self._renderer_has_depth_kw:
            rgb = self.renderer.render()  # default is RGB
            rgb = np.asarray(rgb)
            # Expect (H,W,3) uint8
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return rgb

        # Older API: ensure depth mode is off then render()
        self._set_depth_mode(False)
        rgb = self.renderer.render()
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _set_depth_mode(self, enabled: bool):
        """Toggle depth rendering mode for older Renderer APIs."""
        if not self._renderer_has_enable_depth:
            return
        if enabled and not self._depth_enabled:
            self.renderer.enable_depth_rendering()
            self._depth_enabled = True
        elif (not enabled) and self._depth_enabled and self._renderer_has_disable_depth:
            self.renderer.disable_depth_rendering()
            self._depth_enabled = False
        elif (not enabled) and self._depth_enabled and not self._renderer_has_disable_depth:
            # If disable isn't available, we can still get RGB by re-creating renderer if needed,
            # but most builds do have disable_depth_rendering(). If yours doesn't, tell me and
            # I'll give a tiny workaround.
            self._depth_enabled = False

    def _apply_depth_model(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Apply clipping, dropout, and noise.
        """
        d = depth_m.copy()

        # Clip to near/far and mark invalid as 0 (RealSense often uses 0 for invalid)
        invalid = (d < self.z_near) | (d > self.z_far) | ~np.isfinite(d)
        d[invalid] = 0.0

        # Optional dropout
        if self.dropout_prob > 0.0:
            mask = self.rng.random(d.shape) < self.dropout_prob
            d[mask] = 0.0

        # Add noise only where valid
        valid = d > 0.0
        if self.depth_noise_sigma_m > 0.0:
            d[valid] += self.rng.normal(0.0, self.depth_noise_sigma_m, size=valid.sum())

        d[valid] = np.clip(d[valid], self.z_near, self.z_far)
        return d.astype(np.float32)

    def _raycast_pointcloud_optical(self):
        """
        Returns:
          pts_optical: (N,3) points in RealSense optical frame (+X right, +Y down, +Z forward)
          pix: (N,2) pixel coords (row,col) aligned with points
        Uses true ray intersections (like LiDAR), so geometry is rigid under any rotation.
        """
        # Ensure camera pose reflects mount
        self._apply_mount_to_model_camera()

        # Camera pose in world (MuJoCo camera frame)
        p_cw, R_w_cam = self._camera_pose_world()

        s = self.raycast_stride
        uu = self._uu[::s, ::s]
        vv = self._vv[::s, ::s]

        # Build rays in optical frame (unnormalized): [x, y, 1]
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fy
        ones = np.ones_like(x, dtype=np.float64)

        dirs_opt = np.stack([x, y, ones], axis=-1).reshape(-1, 3)  # (M,3)
        # normalize
        dirs_opt /= np.linalg.norm(dirs_opt, axis=1, keepdims=True)

        # Optical -> MuJoCo camera frame
        dirs_cam = (self.R_mjcam_optical @ dirs_opt.T).T  # (M,3)

        # Camera -> world
        dirs_w = (R_w_cam @ dirs_cam.T).T
        dirs_w /= np.linalg.norm(dirs_w, axis=1, keepdims=True)

        # Cast rays
        n = dirs_w.shape[0]
        pnt = np.asarray(p_cw, dtype=np.float64).reshape(3, 1)
        vec = np.asarray(dirs_w, dtype=np.float64).reshape(3 * n, 1)

        geomid = np.empty((n, 1), dtype=np.int32)
        dist = np.full((n, 1), -1.0, dtype=np.float64)

        mujoco.mj_multiRay(
            self.m,
            self.d,
            pnt,
            vec,
            None,
            1,
            -1,
            geomid,
            dist,
            n,
            float(self.z_far),
        )
        dists = dist[:, 0]
        hit = dists >= 0.0
        if not np.any(hit):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

        # Points in optical frame: P = range * ray_hat_optical
        pts_opt = dirs_opt[hit] * dists[hit, None]

        # Pixel indices aligned with points
        pix = np.stack([vv.reshape(-1)[hit].astype(np.int32),
                        uu.reshape(-1)[hit].astype(np.int32)], axis=1)

        return pts_opt.astype(np.float32), pix

    def _camera_pose_world(self):
        p = self.d.cam_xpos[self.cam_id].copy()
        R = self.d.cam_xmat[self.cam_id].reshape(3, 3).copy()  # camera->world
        return p, R

    def _raycast_depth_image_mm_u16(self, rgb_u8: np.ndarray):
        """
        Build a RealSense-like Z-depth image (16UC1, millimeters) from raycast hits.
        - Depth is Z in the RealSense optical frame (+Z forward).
        - Unhit pixels are 0 (like RealSense invalid depth).
        Also returns per-hit colors and the raycast pointcloud (optical).
        """
        pts_opt, pix = self._raycast_pointcloud_optical()  # pts in optical frame
        H, W = self.height, self.width

        depth_z = np.zeros((H, W), dtype=np.float32)

        if pts_opt.shape[0] > 0:
            z = pts_opt[:, 2].astype(np.float32)  # optical Z depth
            # apply near/far clipping
            valid = (z >= self.z_near) & (z <= self.z_far) & np.isfinite(z)
            if np.any(valid):
                vv = pix[valid, 0]
                uu = pix[valid, 1]
                z = z[valid]

                # optional dropout + noise in meters
                if self.dropout_prob > 0.0:
                    keep = self.rng.random(z.shape[0]) >= self.dropout_prob
                    vv, uu, z = vv[keep], uu[keep], z[keep]

                if self.depth_noise_sigma_m > 0.0 and z.shape[0] > 0:
                    z = z + self.rng.normal(0.0, self.depth_noise_sigma_m, size=z.shape[0]).astype(np.float32)
                    z = np.clip(z, self.z_near, self.z_far)

                depth_z[vv, uu] = z

        depth_mm_u16 = np.clip(depth_z * 1000.0, 0, 65535).astype(np.uint16)

        # colors aligned to points (same pix)
        if pts_opt.shape[0] > 0:
            cols = rgb_u8[pix[:, 0], pix[:, 1], :]
        else:
            cols = np.zeros((0, 3), dtype=np.uint8)

        return depth_z, depth_mm_u16, pts_opt, pix, cols


    def step(self, dt: float) -> dict | None:
        """
        Call this every sim step. Returns a frame dict at the target FPS, else None.

        Frame dict includes:
          depth_m: float32 meters (H,W)
          depth_mm_u16: uint16 millimeters (H,W)
          intrinsics: dict
          pointcloud: (N,3) float32 in camera/world frame (optional)
        """
        self._accum_t += float(dt)
        if self._accum_t < self.frame_dt:
            return None
        self._accum_t -= self.frame_dt

        depth_m = self._render_depth_m()
        depth_m = self._apply_depth_model(depth_m)
        rgb_u8 = self._render_rgb_u8()

        # Match RealSense-like depth representation: uint16 in millimeters
        # Convert rendered depth to RealSense-style Z-depth if the renderer returns range depth
        depth_z_m, depth_mm_u16, pts_cam_optical, pix, colors = self._raycast_depth_image_mm_u16(rgb_u8)

        frame = {
            "t_wall": time.time(),
            "width": self.width,
            "height": self.height,
            "depth_m": depth_z_m,
            "depth_range_m": depth_m,
            "depth_mm_u16": depth_mm_u16,
            "rgb_u8": rgb_u8,
            "intrinsics": {
                "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy,
            },
        }

        if self.output_pointcloud:
            #pts_cam_optical, pix = self._raycast_pointcloud_optical()

            #colors = rgb_u8[pix[:, 0], pix[:, 1], :] if pix.shape[0] else np.zeros((0,3), np.uint8)

            # Optical -> MuJoCo cam frame -> world
            pts_cam_mj = (self.R_mjcam_optical @ pts_cam_optical.T).T
            p_cw, R_w_cam = self._camera_pose_world()
            pts_world = p_cw[None, :] + (R_w_cam @ pts_cam_mj.T).T

            frame["pointcloud_world"] = pts_world.astype(np.float32)
            frame["point_colors_rgb"] = colors

            if self.output_frame == "camera":
                frame["pointcloud"] = pts_cam_optical.astype(np.float32)
                frame["pointcloud_frame"] = "camera_optical"
            elif self.output_frame == "site":
                p_s, R_w_site = self._site_pose_world()
                pts_site = (R_w_site.T @ (pts_world - p_s[None, :]).T).T
                frame["pointcloud"] = pts_site.astype(np.float32)
                frame["pointcloud_frame"] = "site"
            else:
                frame["pointcloud"] = frame["pointcloud_world"]
                frame["pointcloud_frame"] = "world"


        return frame
