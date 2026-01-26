import math
import numpy as np
import mujoco

class LivoxMid360Sim:
    """
    MID-360-like point cloud generator for MuJoCo.
    - Uses mj_multiRay if available (fast), otherwise falls back to looping mj_ray.
    - Matches key output characteristics: FOV, point rate, frame rate, range, noise.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "livox_mid360",
        # MID-360 typical specs:
        frame_rate_hz: float = 10.0,          # typical :contentReference[oaicite:3]{index=3}
        points_per_second: int = 200_000,     # first return :contentReference[oaicite:4]{index=4}
	# Mounting: sensor mounted upside-down and pitched forward (nose-down)
        mount_roll_deg: float = 180.0,     # upside-down (about +X)
        mount_pitch_deg: float = 2.3,     # tilt forward / nose-down (about +Y)
        mount_yaw_deg: float = 0.0,        # typically 0 unless you have yaw offset
        output_frame: str = "sensor",  # "sensor" | "site" | "world"
        # FOV: Horizontal 360°, Vertical -7°~52° :contentReference[oaicite:5]{index=5}
        v_fov_down_deg: float = -7.0,
        v_fov_up_deg: float = 52.0,
        range_min: float = 0.1,               # blind zone ~0.1m :contentReference[oaicite:6]{index=6}
        range_max: float = 70.0,              # depends reflectivity; choose a cap :contentReference[oaicite:7]{index=7}
        # Noise (rough; tune to your needs). MID-360 precision is cm-level. :contentReference[oaicite:8]{index=8}
        range_noise_sigma: float = 0.02,
        dropout_prob: float = 0.00,
        # Performance controls
        max_points_per_frame: int = 20_000,   # 200k/s @10Hz = 20k/frame; you can lower for speed
        seed: int = 1,
    ):
        self.m = model
        self.site_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MJCF.")

        self.frame_rate_hz = float(frame_rate_hz)
        self.points_per_second = int(points_per_second)
        self.points_per_frame_target = int(round(self.points_per_second / self.frame_rate_hz))
        self.points_per_frame = int(min(self.points_per_frame_target, max_points_per_frame))

        self.vmin = math.radians(v_fov_down_deg)
        self.vmax = math.radians(v_fov_up_deg)

        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.range_noise_sigma = float(range_noise_sigma)
        self.dropout_prob = float(dropout_prob)

        self.rng = np.random.default_rng(seed)

        # Fixed rotation mapping SENSOR frame -> SITE frame (mounting offset)
        self.R_site_sensor = self._rotmat_zyx_deg(
            yaw_deg=mount_yaw_deg,
            pitch_deg=mount_pitch_deg,
            roll_deg=mount_roll_deg,
        )
        self.output_frame = output_frame.lower().strip()
        if self.output_frame not in ("sensor", "site", "world"):
            raise ValueError("output_frame must be one of: 'sensor', 'site', 'world'")


        # Accumulate points until we hit a "frame boundary" (1/frame_rate_hz)
        self._frame_dt = 1.0 / self.frame_rate_hz
        self._accum_t = 0.0
        self._accum_points = []

        # Precompute direction set per frame (quasi-uniform random within FOV)
        # Livox is non-repetitive; random sampling is a reasonable approximation.
        self._dirs_lidar = self._sample_dirs_lidar(self.points_per_frame)

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
        """
        Returns R = Rz(yaw) * Ry(pitch) * Rx(roll).
        Interpreted as: apply roll about +X, then pitch about +Y, then yaw about +Z.
        """
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        return cls._rot_z(yaw) @ cls._rot_y(pitch) @ cls._rot_x(roll)
    

    def _sample_dirs_lidar(self, n: int) -> np.ndarray:
        """
        Sample ray directions in the LiDAR local frame.
        Horizontal: [0, 2pi)
        Vertical: [vmin, vmax]
        Return unit vectors (n,3).
        """
        az = self.rng.uniform(0.0, 2.0 * math.pi, size=n)
        el = self.rng.uniform(self.vmin, self.vmax, size=n)

        # Convention: x forward, y left, z up in the LiDAR local frame
        ce = np.cos(el)
        dirs = np.stack([
            ce * np.cos(az),
            ce * np.sin(az),
            np.sin(el),
        ], axis=1).astype(np.float64)

        # Normalize for safety
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs

    def _lidar_pose_world(self, data: mujoco.MjData):
        """
        Returns (pos_world (3,), R_world_lidar (3,3))
        site_xmat is row-major 9 elements representing rotation from site->world.
        """
        pos = data.site_xpos[self.site_id].copy()
        xmat = data.site_xmat[self.site_id].reshape(3, 3).copy()  # site->world
        return pos, xmat
    def _cast_rays(self, data: mujoco.MjData, dirs_world: np.ndarray) -> np.ndarray:
        """
        Cast rays and return distances (n,). -1 means no hit.
        Uses mj_multiRay if available (fast), otherwise loops mj_ray.
        """

        pnt, _ = self._lidar_pose_world(data)
        pnt = np.asarray(pnt, dtype=np.float64).reshape(3, 1)  # (3,1) per binding

        n = int(dirs_world.shape[0])

        # Try mj_multiRay first (fast)
        if hasattr(mujoco, "mj_multiRay"):
            # vec must be flattened to (3*n, 1) in this binding
            vec = np.asarray(dirs_world, dtype=np.float64).reshape(3 * n, 1)

            # outputs must be (n,1) and writeable
            geomid = np.empty((n, 1), dtype=np.int32)
            dist = np.full((n, 1), -1.0, dtype=np.float64)

            mujoco.mj_multiRay(
                self.m,
                data,
                pnt,
                vec,
                None,          # geomgroup
                1,             # flg_static
                -1,            # bodyexclude
                geomid,
                dist,
                n,
                float(self.range_max),
            )
            return dist[:, 0].copy()

        # Fallback: loop mj_ray (slower)
        dists = np.full((n,), -1.0, dtype=np.float64)
        geomid_out = np.zeros((1,), dtype=np.int32)
        pnt_flat = pnt[:, 0]  # mj_ray expects (3,) vector
        for i in range(n):
            dist = mujoco.mj_ray(self.m, data, pnt_flat, dirs_world[i], None, 1, -1, geomid_out)
            dists[i] = float(dist)
        return dists

    def step(self, data: mujoco.MjData, dt: float) -> np.ndarray | None:
        self._accum_t += float(dt)

        # refresh directions (non-repetitive-ish)
        self._dirs_lidar = self._sample_dirs_lidar(self.points_per_frame)

        pos_w, R_w_site = self._lidar_pose_world(data)  # site->world

        # SENSOR -> SITE using mount rotation
        dirs_sensor = self._dirs_lidar                              # (N,3)
        dirs_site   = (self.R_site_sensor @ dirs_sensor.T).T        # (N,3)

        # SITE -> WORLD using live site orientation
        dirs_w = (R_w_site @ dirs_site.T).T                         # (N,3)

        dists = self._cast_rays(data, dirs_w)

        # dropout + noise + clamp
        if self.dropout_prob > 0.0:
            drop = self.rng.random(dists.shape[0]) < self.dropout_prob
            dists[drop] = -1.0

        hit = dists >= 0.0
        dists_hit = dists[hit].copy()
        dists_hit += self.rng.normal(0.0, self.range_noise_sigma, size=dists_hit.shape[0])
        dists_hit = np.clip(dists_hit, self.range_min, self.range_max)

        # Compute points in desired frame
        if self.output_frame == "sensor":
            # points in sensor intrinsic frame (like a real driver would publish)
            pts = dirs_sensor[hit] * dists_hit[:, None]

        elif self.output_frame == "site":
            # points in MJCF site frame (mount applied)
            pts = dirs_site[hit] * dists_hit[:, None]

        else:  # "world"
            # world hit points (best for visualization)
            pts = pos_w[None, :] + dirs_w[hit] * dists_hit[:, None]

        self._accum_points.append(pts)

        if self._accum_t >= self._frame_dt:
            self._accum_t -= self._frame_dt
            cloud = np.concatenate(self._accum_points, axis=0) if self._accum_points else np.zeros((0, 3), dtype=np.float64)
            self._accum_points.clear()
            return cloud

        return None
