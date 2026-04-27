"""Sensor datatypes used by the TerraScout simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorConfig:
    """Noise and geometry parameters for the simulated sensor suite."""

    lidar_fov_deg: float = 270.0
    lidar_angular_resolution_deg: float = 0.5
    lidar_range_noise_m: float = 0.02
    gyro_bias_rps: float = 0.004
    gyro_noise_rps: float = 0.003
    accel_noise_mps2: float = 0.025
    encoder_noise_fraction: float = 0.03


@dataclass(frozen=True)
class LidarScan:
    """Polar 2D lidar scan in the rover frame."""

    angles_rad: list[float]
    ranges_m: list[float]
    max_range_m: float


@dataclass(frozen=True)
class ImuSample:
    """Minimal planar IMU sample."""

    yaw_rate_rps: float
    longitudinal_accel_mps2: float


@dataclass(frozen=True)
class EncoderSample:
    """Wheel-encoder distance increment over one simulator tick."""

    left_delta_m: float
    right_delta_m: float


@dataclass(frozen=True)
class SensorFrame:
    """Synchronized sensor frame for one control tick."""

    lidar: LidarScan
    imu: ImuSample
    encoders: EncoderSample
