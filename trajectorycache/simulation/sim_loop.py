"""Python Simulation Loop.

A pure-Python discrete-event simulation of the TrajectoryCache system.
Does NOT require SUMO or NS-3 — uses synthetic vehicle mobility traces.

This is the primary tool for:
  - Algorithm validation and debugging
  - Rapid parameter sweeps
  - Reproducing the paper's key plots without the full co-simulation stack

For full experimental accuracy, use the SUMO + ndnSIM co-simulation
pipeline described in docs/SETUP.md.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectorycache.catalog.generator import (
    ContentChunk,
    generate_catalog,
    generate_highway_segments,
    generate_urban_segments,
)
from trajectorycache.config import ProjectConfig
from trajectorycache.core.affinity_estimator import AffinityEstimator
from trajectorycache.core.bsm_listener import BsmListener, VehicleState
from trajectorycache.core.content_store import ContentStore
from trajectorycache.core.ctrv_predictor import CTRVPredictor
from trajectorycache.core.eviction_engine import EvictionEngine
from trajectorycache.core.mrs_scorer import MrsScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic vehicle mobility model
# ---------------------------------------------------------------------------


class SyntheticVehicle:
    """Simple kinematic vehicle for synthetic mobility traces."""

    def __init__(
        self,
        vehicle_id: str,
        x: float,
        y: float,
        speed: float,
        heading: float,
        rng: random.Random,
        scenario: str = "highway",
    ) -> None:
        self.vehicle_id = vehicle_id
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading
        self.turn_rate = 0.0
        self.rng = rng
        self.scenario = scenario

    def step(self, dt: float) -> None:
        """Update position using CTRV kinematics with random turn-rate noise."""
        # Randomly perturb turn rate (simulates urban driving)
        if self.scenario == "urban":
            self.turn_rate += self.rng.gauss(0, 0.01) * dt
            self.turn_rate = max(-0.3, min(0.3, self.turn_rate))
        else:
            self.turn_rate *= 0.98  # Decay toward straight-line

        epsilon = 1e-4
        if abs(self.turn_rate) >= epsilon:
            r = self.speed / self.turn_rate
            self.x += r * (math.sin(self.heading + self.turn_rate * dt) - math.sin(self.heading))
            self.y += r * (math.cos(self.heading) - math.cos(self.heading + self.turn_rate * dt))
            self.heading += self.turn_rate * dt
        else:
            self.x += self.speed * dt * math.cos(self.heading)
            self.y += self.speed * dt * math.sin(self.heading)

        # Wrap heading
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # Slightly vary speed
        self.speed += self.rng.gauss(0, 0.1) * dt
        self.speed = max(5.0, min(40.0, self.speed))


class ScenarioMobility:
    """Generates and steps synthetic vehicle mobility for a scenario."""

    def __init__(
        self,
        config: ProjectConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng
        scenario = config.sim.scenario_name
        n = config.sim.num_vehicles
        self.vehicles: Dict[str, SyntheticVehicle] = {}
        self._spawn_vehicles(n, scenario)

    def _spawn_vehicles(self, n: int, scenario: str) -> None:
        for i in range(n):
            vid = f"veh_{i:04d}"
            if scenario == "highway":
                x = self.rng.uniform(0, 5000)
                y = self.rng.gauss(0, 5)  # Small lateral spread
                speed = self.rng.uniform(25, 35)  # ~90–126 km/h
                heading = 0.0 if self.rng.random() > 0.5 else math.pi
            else:  # urban
                x = self.rng.uniform(0, 1000)
                y = self.rng.uniform(0, 1000)
                speed = self.rng.uniform(8, 15)  # ~30–54 km/h
                heading = self.rng.choice([0, math.pi / 2, math.pi, -math.pi / 2])
            self.vehicles[vid] = SyntheticVehicle(vid, x, y, speed, heading, self.rng, scenario)

    def step(self, dt: float) -> None:
        for v in self.vehicles.values():
            v.step(dt)

    def get_states(self) -> Dict[str, SyntheticVehicle]:
        return self.vehicles


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


@dataclass
class SimulationMetrics:
    """Aggregated metrics over the measurement window."""

    total_interests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    backhaul_requests: int = 0
    eviction_cycles: int = 0
    total_evicted: int = 0

    # Latency tracking (ms)
    latency_samples: List[float] = field(default_factory=list)

    # Per-cycle overhead tracking (ms)
    cycle_time_samples: List[float] = field(default_factory=list)

    @property
    def miss_rate(self) -> float:
        if self.total_interests == 0:
            return 0.0
        return self.cache_misses / self.total_interests

    @property
    def hit_rate(self) -> float:
        return 1.0 - self.miss_rate

    @property
    def isr(self) -> float:
        """Interest Satisfaction Ratio: fraction satisfied within 50 ms."""
        if not self.latency_samples:
            return 0.0
        return sum(1 for l in self.latency_samples if l <= 50.0) / len(self.latency_samples)

    @property
    def mean_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        return float(np.mean(self.latency_samples))

    @property
    def mean_cycle_time_ms(self) -> float:
        if not self.cycle_time_samples:
            return 0.0
        return float(np.mean(self.cycle_time_samples))

    @property
    def per_interest_overhead_us(self) -> float:
        """Estimated per-Interest overhead attributable to TrajectoryCache (µs)."""
        if self.total_interests == 0 or not self.cycle_time_samples:
            return 0.0
        total_cycle_ms = sum(self.cycle_time_samples)
        return (total_cycle_ms / self.total_interests) * 1000.0

    def summary(self) -> dict:
        return {
            "miss_rate": self.miss_rate,
            "hit_rate": self.hit_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "isr_50ms": self.isr,
            "per_interest_overhead_us": self.per_interest_overhead_us,
            "total_interests": self.total_interests,
            "cache_hits": self.cache_hits,
            "eviction_cycles": self.eviction_cycles,
            "total_evicted": self.total_evicted,
            "mean_cycle_time_ms": self.mean_cycle_time_ms,
        }


# ---------------------------------------------------------------------------
# Main simulation class
# ---------------------------------------------------------------------------

LOCAL_HIT_LATENCY_MS = 3.0     # CS hit: ~3 ms
BACKHAUL_LATENCY_MS = 80.0     # Cache miss backhaul round-trip
BACKHAUL_STD_MS = 10.0         # Jitter on backhaul


class TrajectorySimulation:
    """Discrete-event TrajectoryCache simulation.

    Runs a fixed-duration simulation with synthetic BSMs,
    Zipf-distributed content requests, and the full TrajectoryCache pipeline.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        tc = config.tc
        rsu = config.rsu
        sim = config.sim

        self.rng = random.Random(sim.random_seed)
        self.np_rng = np.random.default_rng(sim.random_seed)

        # Build content catalog
        scenario = sim.scenario_name
        if scenario == "highway":
            segments = generate_highway_segments()
        else:
            segments = generate_urban_segments()

        self.catalog = generate_catalog(
            segments,
            zipf_alpha=sim.zipf_alpha,
            random_seed=sim.random_seed,
            r_grz=tc.r_grz,
            rsu_x=rsu.position_x,
            rsu_y=rsu.position_y,
        )

        # Zipf request weights (by popularity rank)
        sorted_catalog = sorted(self.catalog, key=lambda c: c.popularity_rank)
        n = len(sorted_catalog)
        raw_weights = 1.0 / (np.arange(1, n + 1) ** sim.zipf_alpha)
        self._request_weights = raw_weights / raw_weights.sum()
        self._sorted_catalog = sorted_catalog

        # Core components
        self.cs = ContentStore(
            capacity_bytes=rsu.cs_capacity_bytes,
            current_time=0.0,
        )
        self.predictor = CTRVPredictor(
            epsilon_turn=tc.epsilon_turn,
            delta_tau=tc.delta_tau,
        )
        self.affinity = AffinityEstimator(beta=tc.affinity_beta)
        self.scorer = MrsScorer(
            predictor=self.predictor,
            affinity=self.affinity,
            r_grz=tc.r_grz,
            alpha=tc.alpha,
        )
        self.engine = EvictionEngine(
            content_store=self.cs,
            scorer=self.scorer,
            predictor=self.predictor,
            affinity=self.affinity,
            lambda_weight=tc.lambda_weight,
            eta_hw=tc.eta_hw,
            eta_lw=tc.eta_lw,
            rsu_x=rsu.position_x,
            rsu_y=rsu.position_y,
            coverage_radius=rsu.coverage_radius_m,
        )
        self.bsm_listener = BsmListener(
            rsu_x=rsu.position_x,
            rsu_y=rsu.position_y,
            coverage_radius=rsu.coverage_radius_m,
            gps_noise_sigma=sim.gps_noise_sigma,
        )
        self.mobility = ScenarioMobility(config, self.rng)

        self.metrics = SimulationMetrics()
        self._in_warmup = True

    def run(self) -> SimulationMetrics:
        """Run the full simulation. Returns metrics after warmup period."""
        tc = self.config.tc
        sim = self.config.sim

        bsm_dt = tc.bsm_period_s
        eviction_dt = tc.eviction_cycle_s
        t = 0.0
        t_last_eviction = 0.0
        t_last_bsm = 0.0
        interests_per_s = sim.num_vehicles * 2.0  # ~2 interests/vehicle/s

        logger.info(
            "Starting simulation: scenario=%s vehicles=%d duration=%.0fs seed=%d",
            sim.scenario_name,
            sim.num_vehicles,
            sim.duration_s,
            sim.random_seed,
        )

        wall_start = time.perf_counter()
        dt = 0.1  # Simulation step size (seconds)

        while t < sim.duration_s:
            self._in_warmup = (t < sim.warmup_s)

            # --- BSM processing step ---
            if t - t_last_bsm >= bsm_dt:
                self._process_bsms(t)
                t_last_bsm = t

            # --- Generate content requests ---
            n_interests = self.np_rng.poisson(interests_per_s * dt)
            for _ in range(int(n_interests)):
                self._handle_interest(t)

            # --- Eviction cycle ---
            if t - t_last_eviction >= eviction_dt:
                states = self.bsm_listener.neighbor_table.snapshot()
                result = self.engine.run_cycle(states, t)
                if result.triggered and not self._in_warmup:
                    self.metrics.eviction_cycles += 1
                    self.metrics.total_evicted += result.n_evicted
                    self.metrics.cycle_time_samples.append(result.cycle_time_ms)
                t_last_eviction = t

            # --- Advance mobility ---
            self.mobility.step(dt)
            self.bsm_listener.tick(t)

            t += dt

        wall_elapsed = time.perf_counter() - wall_start
        logger.info(
            "Simulation complete in %.1fs wall-clock. "
            "miss_rate=%.3f isr=%.3f mean_latency=%.1fms",
            wall_elapsed,
            self.metrics.miss_rate,
            self.metrics.isr,
            self.metrics.mean_latency_ms,
        )
        return self.metrics

    def _process_bsms(self, t: float) -> None:
        """Inject synthetic BSMs from all vehicles into the BSM listener."""
        for vid, veh in self.mobility.vehicles.items():
            self.bsm_listener.process_bsm(
                vehicle_id=vid,
                x=veh.x,
                y=veh.y,
                speed=veh.speed,
                heading=veh.heading,
                timestamp=t,
                drop=False,
            )

    def _handle_interest(self, t: float) -> None:
        """Simulate a vehicle issuing an NDN Interest."""
        # Pick a chunk according to Zipf distribution
        idx = self.np_rng.choice(len(self._sorted_catalog), p=self._request_weights)
        chunk = self._sorted_catalog[idx]

        # Pick a random vehicle as the requester
        vehicle_ids = list(self.mobility.vehicles.keys())
        if not vehicle_ids:
            return
        vid = self.rng.choice(vehicle_ids)

        # Update content affinity
        self.affinity.update(vid, chunk.name, requested=True)

        # CS lookup
        hit = self.cs.lookup(chunk.name)

        if not self._in_warmup:
            self.metrics.total_interests += 1

            if hit is not None:
                self.metrics.cache_hits += 1
                latency = LOCAL_HIT_LATENCY_MS + self.np_rng.normal(0, 0.5)
            else:
                self.metrics.cache_misses += 1
                self.metrics.backhaul_requests += 1
                latency = BACKHAUL_LATENCY_MS + self.np_rng.normal(0, BACKHAUL_STD_MS)
                # Cache the fetched chunk
                states = self.bsm_listener.neighbor_table.snapshot()
                self.engine.insert_and_maybe_evict(chunk, states, t)

            self.metrics.latency_samples.append(max(0.1, latency))
