from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorRuntimeConfig:
    sensor_mode: str
    enable_lidar: bool
    enable_d435: bool
    enable_web_ui: bool
    d435_depth_hz: float
    d435_depth_mode: str
    d435_rgb_hz: float | None
    d435_output_pointcloud: bool
    lidar_points_per_second: int
    lidar_max_points_per_frame: int
    ros_clock_hz: float | None
    ros_tf_hz: float | None
    ros_odom_hz: float | None
    ros_imu_hz: float | None
    ros_joint_hz: float | None


def _override(value, default):
    return default if value is None else value


def resolve_sensor_runtime_config(args) -> SensorRuntimeConfig:
    """Resolve CLI sensor/performance defaults in one testable place."""
    sensor_mode = (getattr(args, "sensor_mode", None) or "realtime").strip().lower()
    if getattr(args, "mapping_mode", False) and getattr(args, "sensor_mode", None) is None:
        sensor_mode = "realtime"

    if sensor_mode not in ("realtime", "fidelity", "off"):
        raise ValueError("sensor_mode must be one of: realtime, fidelity, off")

    show_sensors = bool(getattr(args, "show_sensors", False))
    explicit_d435_pointcloud = bool(getattr(args, "d435_pointcloud", False))

    if sensor_mode == "fidelity":
        default_ros_clock_hz = None
        default_ros_tf_hz = None
        default_ros_odom_hz = None
        default_ros_imu_hz = None
        default_ros_joint_hz = None
        default_d435_depth_hz = 30.0
        default_d435_depth_mode = "raycast"
        default_d435_rgb_hz = None
        default_d435_pointcloud = True
        default_web_ui = False
        default_lidar_points_per_second = 200_000
        default_lidar_max_points_per_frame = 8000
    else:
        default_ros_clock_hz = 100.0
        default_ros_tf_hz = 100.0
        default_ros_odom_hz = 100.0
        default_ros_imu_hz = 100.0
        default_ros_joint_hz = 50.0
        default_d435_depth_hz = 10.0
        default_d435_depth_mode = "render_fast"
        default_d435_rgb_hz = 2.0
        default_d435_pointcloud = show_sensors or explicit_d435_pointcloud
        default_web_ui = False
        default_lidar_points_per_second = 200_000
        default_lidar_max_points_per_frame = 8000

    d435_depth_mode = _override(getattr(args, "d435_depth_mode", None), default_d435_depth_mode)
    if d435_depth_mode not in ("render_fast", "raycast"):
        raise ValueError("d435_depth_mode must be one of: render_fast, raycast")

    d435_depth_hz = float(_override(getattr(args, "d435_depth_hz", None), default_d435_depth_hz))
    if d435_depth_hz <= 0.0:
        raise ValueError("d435_depth_hz must be > 0")

    lidar_points_per_second = int(_override(getattr(args, "lidar_points_per_second", None), default_lidar_points_per_second))
    lidar_max_points_per_frame = int(_override(getattr(args, "lidar_max_points_per_frame", None), default_lidar_max_points_per_frame))
    if lidar_points_per_second <= 0:
        raise ValueError("lidar_points_per_second must be > 0")
    if lidar_max_points_per_frame <= 0:
        raise ValueError("lidar_max_points_per_frame must be > 0")

    d435_output_pointcloud = (default_d435_pointcloud or explicit_d435_pointcloud) and sensor_mode != "off"

    return SensorRuntimeConfig(
        sensor_mode=sensor_mode,
        enable_lidar=sensor_mode != "off",
        enable_d435=sensor_mode != "off",
        enable_web_ui=bool(getattr(args, "web_ui", False)) or default_web_ui,
        d435_depth_hz=d435_depth_hz,
        d435_depth_mode=d435_depth_mode,
        d435_rgb_hz=_override(getattr(args, "d435_rgb_hz", None), default_d435_rgb_hz),
        d435_output_pointcloud=d435_output_pointcloud,
        lidar_points_per_second=lidar_points_per_second,
        lidar_max_points_per_frame=lidar_max_points_per_frame,
        ros_clock_hz=_override(getattr(args, "ros_clock_hz", None), default_ros_clock_hz),
        ros_tf_hz=_override(getattr(args, "ros_tf_hz", None), default_ros_tf_hz),
        ros_odom_hz=_override(getattr(args, "ros_odom_hz", None), default_ros_odom_hz),
        ros_imu_hz=_override(getattr(args, "ros_imu_hz", None), default_ros_imu_hz),
        ros_joint_hz=_override(getattr(args, "ros_joint_hz", None), default_ros_joint_hz),
    )
