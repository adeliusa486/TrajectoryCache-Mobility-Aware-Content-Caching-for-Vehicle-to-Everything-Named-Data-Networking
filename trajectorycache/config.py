"""Configuration management for TrajectoryCache.

Loads and validates JSON/YAML config files. All parameters
correspond to those described in the paper (Section IV).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryCacheConfig:
    """Hyperparameters for the TrajectoryCache eviction algorithm."""

    # Composite score weight (0 = pure LRU, 1 = pure MRS)
    lambda_weight: float = 0.75

    # Arrival urgency decay coefficient α (s⁻¹), Eq. 6
    alpha: float = 0.5

    # High-water mark: trigger eviction when CS ≥ η_hw · C
    eta_hw: float = 0.90

    # Low-water mark: evict until CS ≤ η_lw · C
    eta_lw: float = 0.70

    # Geographic Relevance Zone radius (meters)
    r_grz: float = 300.0

    # Minimum turn rate magnitude to use CTRV vs CV fallback (rad/s)
    epsilon_turn: float = 1e-4

    # Content affinity EMA window (seconds) — β = 1/affinity_window_s
    affinity_window_s: float = 300.0

    # Eviction cycle period (seconds)
    eviction_cycle_s: float = 1.0

    # BSM reception rate (Hz) per vehicle
    bsm_rate_hz: float = 10.0

    # Trajectory prediction step size (seconds)
    delta_tau: float = 0.5

    # Stale neighbor timeout: remove if not heard for N BSM periods
    stale_timeout_periods: int = 2

    @property
    def bsm_period_s(self) -> float:
        return 1.0 / self.bsm_rate_hz

    @property
    def stale_timeout_s(self) -> float:
        return self.stale_timeout_periods * self.bsm_period_s

    @property
    def affinity_beta(self) -> float:
        """EMA learning rate β = 1/W."""
        return 1.0 / self.affinity_window_s

    def validate(self) -> None:
        assert 0.0 <= self.lambda_weight <= 1.0, "lambda must be in [0,1]"
        assert 0.0 < self.alpha, "alpha must be positive"
        assert 0.0 < self.eta_lw < self.eta_hw <= 1.0, "0 < eta_lw < eta_hw <= 1"
        assert self.r_grz > 0, "r_grz must be positive"
        assert self.epsilon_turn > 0, "epsilon_turn must be positive"
        assert self.affinity_window_s > 0, "affinity_window_s must be positive"
        assert self.eviction_cycle_s > 0, "eviction_cycle_s must be positive"


@dataclass
class RsuConfig:
    """RSU hardware / network parameters."""

    # Coverage radius (meters)
    coverage_radius_m: float = 300.0

    # Content Store capacity (MB)
    cs_capacity_mb: float = 500.0

    # Backhaul bandwidth (Gbps)
    backhaul_bandwidth_gbps: float = 1.0

    # Wireless channel rate (Mbps)
    wireless_rate_mbps: float = 54.0

    # RSU position (meters, Cartesian)
    position_x: float = 0.0
    position_y: float = 0.0

    @property
    def cs_capacity_bytes(self) -> int:
        return int(self.cs_capacity_mb * 1024 * 1024)

    def validate(self) -> None:
        assert self.coverage_radius_m > 0
        assert self.cs_capacity_mb > 0
        assert self.backhaul_bandwidth_gbps > 0


@dataclass
class SimulationConfig:
    """Simulation scenario parameters."""

    scenario_name: str = "highway"
    duration_s: float = 600.0
    warmup_s: float = 120.0
    num_vehicles: int = 100
    random_seed: int = 42

    # Content catalog size
    catalog_size: int = 12500
    zipf_alpha: float = 0.8

    # GPS noise standard deviation (meters, 0 = perfect GPS)
    gps_noise_sigma: float = 0.0

    @property
    def measurement_duration_s(self) -> float:
        return self.duration_s - self.warmup_s


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    tc: TrajectoryCacheConfig = field(default_factory=TrajectoryCacheConfig)
    rsu: RsuConfig = field(default_factory=RsuConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> None:
        self.tc.validate()
        self.rsu.validate()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectConfig":
        tc_data = d.get("trajectoryCache", {})
        rsu_data = d.get("rsu", {})
        sim_data = d.get("simulation", {})

        tc = TrajectoryCacheConfig(
            lambda_weight=tc_data.get("lambda", 0.75),
            alpha=tc_data.get("alpha", 0.5),
            eta_hw=tc_data.get("eta_hw", 0.90),
            eta_lw=tc_data.get("eta_lw", 0.70),
            r_grz=tc_data.get("r_grz", 300.0),
            epsilon_turn=tc_data.get("epsilon_turn", 1e-4),
            affinity_window_s=tc_data.get("affinity_window_s", 300.0),
            eviction_cycle_s=tc_data.get("eviction_cycle_s", 1.0),
            bsm_rate_hz=tc_data.get("bsm_rate_hz", 10.0),
        )

        rsu = RsuConfig(
            coverage_radius_m=rsu_data.get("coverage_radius_m", 300.0),
            cs_capacity_mb=rsu_data.get("cs_capacity_mb", 500.0),
            backhaul_bandwidth_gbps=rsu_data.get("backhaul_bandwidth_gbps", 1.0),
            wireless_rate_mbps=rsu_data.get("wireless_rate_mbps", 54.0),
            position_x=rsu_data.get("position_x", 0.0),
            position_y=rsu_data.get("position_y", 0.0),
        )

        sim = SimulationConfig(
            scenario_name=sim_data.get("scenario_name", "highway"),
            duration_s=sim_data.get("duration_s", 600.0),
            warmup_s=sim_data.get("warmup_s", 120.0),
            num_vehicles=sim_data.get("num_vehicles", 100),
            random_seed=sim_data.get("random_seed", 42),
            catalog_size=sim_data.get("catalog_size", 12500),
            zipf_alpha=sim_data.get("zipf_alpha", 0.8),
            gps_noise_sigma=sim_data.get("gps_noise_sigma", 0.0),
        )

        return cls(tc=tc, rsu=rsu, sim=sim)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a JSON or YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    config = ProjectConfig.from_dict(data)
    config.validate()
    logger.info("Loaded config from %s", path)
    return config


def default_highway_config() -> ProjectConfig:
    """Return the default highway scenario configuration."""
    cfg = ProjectConfig()
    cfg.sim.scenario_name = "highway"
    cfg.sim.num_vehicles = 300
    cfg.rsu.position_x = 2500.0
    cfg.rsu.position_y = 0.0
    return cfg


def default_urban_config() -> ProjectConfig:
    """Return the default urban grid scenario configuration."""
    cfg = ProjectConfig()
    cfg.sim.scenario_name = "urban"
    cfg.sim.num_vehicles = 300
    cfg.rsu.position_x = 500.0
    cfg.rsu.position_y = 500.0
    return cfg
