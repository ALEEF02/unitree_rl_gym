import math
import time
from warnings import warn

import mujoco
import numpy as np

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
        stabilize_roll_pitch: bool = True,
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
        if self.frame_rate_hz <= 0.0:
            raise ValueError("frame_rate_hz must be > 0")
        self.points_per_second = int(points_per_second)
        self.points_per_frame_target = int(round(self.points_per_second / self.frame_rate_hz))
        self.points_per_frame = max(1, int(min(self.points_per_frame_target, max_points_per_frame)))
        self.effective_points_per_second = float(self.points_per_frame * self.frame_rate_hz)

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
        self.stabilize_roll_pitch = bool(stabilize_roll_pitch)

        self._has_multi_ray = hasattr(mujoco, "mj_multiRay")
        self._warned_mj_ray_fallback = False

        # Time accumulators.
        self._frame_dt = 1.0 / self.frame_rate_hz
        self._accum_t = 0.0
        self._point_budget = 0.0

        # Frame buffer (bounded by points_per_frame).
        self._frame_points = np.empty((self.points_per_frame, 3), dtype=np.float64)
        self._frame_count = 0

        # Reusable hot-path buffers.
        self._pnt = np.zeros((3, 1), dtype=np.float64)
        self._dirs_sensor = np.empty((self.points_per_frame, 3), dtype=np.float64)
        self._dirs_site = np.empty((self.points_per_frame, 3), dtype=np.float64)
        self._dirs_world = np.empty((self.points_per_frame, 3), dtype=np.float64)
        self._dists = np.empty((self.points_per_frame,), dtype=np.float64)
        self._tmp_dists = np.empty((self.points_per_frame,), dtype=np.float64)
        self._tmp_points = np.empty((self.points_per_frame, 3), dtype=np.float64)

        if self._has_multi_ray:
            self._vec = np.empty((3 * self.points_per_frame, 1), dtype=np.float64)
            self._multi_geomid = np.empty((self.points_per_frame, 1), dtype=np.int32)
            self._multi_dist = np.empty((self.points_per_frame, 1), dtype=np.float64)
        else:
            self._mjray_geomid_out = np.zeros((1,), dtype=np.int32)

        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict:
        return {
            "steps": 0,
            "sim_time_s": 0.0,
            "frames_emitted": 0,
            "rays_cast": 0,
            "points_kept": 0,
            "pack_calls": 0,
            "t_sample_s": 0.0,
            "t_transform_s": 0.0,
            "t_raycast_s": 0.0,
            "t_postprocess_s": 0.0,
            "t_pack_s": 0.0,
        }

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
    

    def _sample_dirs_lidar(self, n: int, out_dirs: np.ndarray) -> None:
        """
        Sample ray directions in the LiDAR local frame.
        Horizontal: [0, 2pi)
        Vertical: [vmin, vmax]
        Writes unit vectors to out_dirs with shape (n,3).
        """
        az = self.rng.uniform(0.0, 2.0 * math.pi, size=n)
        el = self.rng.uniform(self.vmin, self.vmax, size=n)

        # Convention: x forward, y left, z up in the LiDAR local frame
        ce = np.cos(el)
        out_dirs[:, 0] = ce * np.cos(az)
        out_dirs[:, 1] = ce * np.sin(az)
        out_dirs[:, 2] = np.sin(el)

    def _lidar_pose_world(self, data: mujoco.MjData):
        """
        Returns (pos_world (3,), R_world_lidar (3,3))
        site_xmat is row-major 9 elements representing rotation from site->world.
        """
        pos = data.site_xpos[self.site_id]
        xmat = data.site_xmat[self.site_id].reshape(3, 3)  # site->world
        return pos, xmat

    def _cast_rays(self, data: mujoco.MjData, dirs_world: np.ndarray, n: int) -> np.ndarray:
        """
        Cast rays and return distances (n,). -1 means no hit.
        Uses mj_multiRay if available (fast), otherwise loops mj_ray.
        """
        self._pnt[:, 0] = data.site_xpos[self.site_id]

        if self._has_multi_ray:
            vec = self._vec[: 3 * n]
            geomid = self._multi_geomid[:n]
            dist = self._multi_dist[:n]
            vec[:, 0] = dirs_world[:n].reshape(3 * n)
            dist[:, 0].fill(-1.0)
            mujoco.mj_multiRay(
                self.m,
                data,
                self._pnt,
                vec,
                None,          # geomgroup
                1,             # flg_static
                -1,            # bodyexclude
                geomid,
                dist,
                n,
                float(self.range_max),
            )
            return dist[:, 0]

        if not self._warned_mj_ray_fallback and self.effective_points_per_second >= 5_000:
            warn(
                "LivoxMid360Sim is using mj_ray fallback without mj_multiRay at a high effective point rate; "
                "performance may degrade significantly."
            )
            self._warned_mj_ray_fallback = True

        dists = self._dists[:n]
        dists.fill(-1.0)
        pnt_flat = self._pnt[:, 0]
        for i in range(n):
            dists[i] = float(
                mujoco.mj_ray(self.m, data, pnt_flat, dirs_world[i], None, 1, -1, self._mjray_geomid_out)
            )
        return dists

    def record_pack_time(self, seconds: float) -> None:
        dt = float(seconds)
        if dt > 0.0:
            self._stats["t_pack_s"] += dt
            self._stats["pack_calls"] += 1

    def get_stats(self, reset: bool = False) -> dict:
        stats = dict(self._stats)
        sim_time = max(float(stats["sim_time_s"]), 1e-9)
        frame_count = int(stats["frames_emitted"])
        frame_denom = max(frame_count, 1)

        stats["effective_points_per_second"] = self.effective_points_per_second
        stats["points_per_frame_limit"] = self.points_per_frame
        stats["rays_per_sec"] = float(stats["rays_cast"]) / sim_time
        stats["points_per_sec"] = float(stats["points_kept"]) / sim_time
        stats["points_per_frame_avg"] = float(stats["points_kept"]) / frame_denom
        stats["sample_ms_per_frame"] = 1e3 * float(stats["t_sample_s"]) / frame_denom
        stats["transform_ms_per_frame"] = 1e3 * float(stats["t_transform_s"]) / frame_denom
        stats["raycast_ms_per_frame"] = 1e3 * float(stats["t_raycast_s"]) / frame_denom
        stats["postprocess_ms_per_frame"] = 1e3 * float(stats["t_postprocess_s"]) / frame_denom
        stats["pack_ms_per_frame"] = 1e3 * float(stats["t_pack_s"]) / frame_denom

        if reset:
            self._stats = self._new_stats()

        return stats

    def step(self, data: mujoco.MjData, dt: float) -> np.ndarray | None:
        dt = float(dt)
        self._stats["steps"] += 1
        self._stats["sim_time_s"] += dt
        self._accum_t += dt

        # Continuous budgeted raycasting (balanced speed/fidelity).
        self._point_budget += self.effective_points_per_second * dt
        rays_to_cast = int(self._point_budget)
        if rays_to_cast > self.points_per_frame:
            rays_to_cast = self.points_per_frame
        if rays_to_cast > 0:
            self._point_budget -= rays_to_cast

            t0 = time.perf_counter()
            dirs_sensor = self._dirs_sensor[:rays_to_cast]
            self._sample_dirs_lidar(rays_to_cast, dirs_sensor)
            self._stats["t_sample_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            pos_w, R_w_site = self._lidar_pose_world(data)  # site->world
            dirs_site = self._dirs_site[:rays_to_cast]
            dirs_world = self._dirs_world[:rays_to_cast]
            np.matmul(dirs_sensor, self.R_site_sensor.T, out=dirs_site)
            if self.stabilize_roll_pitch:
                yaw = math.atan2(float(R_w_site[1, 0]), float(R_w_site[0, 0]))
                cy, sy = math.cos(yaw), math.sin(yaw)
                R_w_cast = np.array(
                    [[cy, -sy, 0.0],
                     [sy,  cy, 0.0],
                     [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            else:
                R_w_cast = R_w_site
            np.matmul(dirs_site, R_w_cast.T, out=dirs_world)
            self._stats["t_transform_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            dists = self._cast_rays(data, dirs_world, rays_to_cast)
            self._stats["t_raycast_s"] += time.perf_counter() - t0
            self._stats["rays_cast"] += rays_to_cast

            t0 = time.perf_counter()
            if self.dropout_prob > 0.0:
                drop = self.rng.random(rays_to_cast) < self.dropout_prob
                dists[drop] = -1.0

            hit_idx = np.flatnonzero(dists >= 0.0)
            n_hit = int(hit_idx.shape[0])
            if n_hit > 0:
                dists_hit = self._tmp_dists[:n_hit]
                dists_hit[:] = dists[hit_idx]
                if self.range_noise_sigma > 0.0:
                    dists_hit += self.rng.normal(0.0, self.range_noise_sigma, size=n_hit)
                np.clip(dists_hit, self.range_min, self.range_max, out=dists_hit)

                pts = self._tmp_points[:n_hit]
                if self.output_frame == "sensor":
                    np.multiply(dirs_sensor[hit_idx], dists_hit[:, None], out=pts)
                elif self.output_frame == "site":
                    np.multiply(dirs_site[hit_idx], dists_hit[:, None], out=pts)
                else:  # "world"
                    np.multiply(dirs_world[hit_idx], dists_hit[:, None], out=pts)
                    pts += pos_w[None, :]

                free_slots = self.points_per_frame - self._frame_count
                if free_slots > 0:
                    take = min(free_slots, n_hit)
                    self._frame_points[self._frame_count : self._frame_count + take] = pts[:take]
                    self._frame_count += take
                    self._stats["points_kept"] += take

            self._stats["t_postprocess_s"] += time.perf_counter() - t0

        if self._accum_t >= self._frame_dt:
            self._accum_t -= self._frame_dt
            self._stats["frames_emitted"] += 1
            if self._frame_count > 0:
                cloud = self._frame_points[: self._frame_count].copy()
            else:
                cloud = np.zeros((0, 3), dtype=np.float64)
            self._frame_count = 0
            return cloud

        return None
