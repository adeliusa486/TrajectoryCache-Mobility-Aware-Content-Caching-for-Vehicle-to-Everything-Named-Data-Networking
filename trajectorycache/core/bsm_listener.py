"""BSM Listener and Neighbor Table Management.

Processes IEEE 802.11p Basic Safety Messages (SAE J2735 format) to maintain
a real-time neighbor table of vehicles within RSU coverage.

In production, this module interfaces with the OS-level 802.11p socket.
In simulation, BSMs are injected via the SUMO TraCI adapter.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """State of a single vehicle as derived from BSM messages.

    Coordinates are in Cartesian meters relative to the simulation origin.
    """

    vehicle_id: str
    x: float           # Position x (m)
    y: float           # Position y (m)
    speed: float       # Speed (m/s)
    heading: float     # Heading angle θ (radians, 0 = East, counterclockwise)
    turn_rate: float   # Angular velocity θ̇ (rad/s), estimated by finite difference
    timestamp: float   # Time of last BSM reception (simulation seconds)
    t_arrive: float    # Estimated time to enter RSU coverage (s); 0 if already inside
    last_heading: Optional[float] = None  # Previous heading for turn rate estimation

    def is_inside_coverage(self, rsu_x: float, rsu_y: float, radius: float) -> bool:
        """Return True if vehicle is currently within RSU coverage radius."""
        dist = math.hypot(self.x - rsu_x, self.y - rsu_y)
        return dist <= radius

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from vehicle to a point."""
        return math.hypot(self.x - x, self.y - y)


class NeighborTable:
    """Thread-safe neighbor table keyed by vehicle_id.

    Maintains the latest VehicleState for each vehicle heard within
    the stale timeout window.
    """

    def __init__(
        self,
        stale_timeout_s: float = 0.2,
        rsu_x: float = 0.0,
        rsu_y: float = 0.0,
        coverage_radius: float = 300.0,
    ) -> None:
        self._table: Dict[str, VehicleState] = {}
        self._lock = threading.RLock()
        self.stale_timeout_s = stale_timeout_s
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.coverage_radius = coverage_radius

    def update(self, state: VehicleState) -> None:
        """Insert or update a vehicle entry."""
        with self._lock:
            self._table[state.vehicle_id] = state

    def get(self, vehicle_id: str) -> Optional[VehicleState]:
        """Retrieve current state for a vehicle."""
        with self._lock:
            return self._table.get(vehicle_id)

    def prune_stale(self, current_time: float) -> int:
        """Remove vehicles not heard within stale_timeout_s. Returns count removed."""
        with self._lock:
            stale_ids = [
                vid
                for vid, state in self._table.items()
                if (current_time - state.timestamp) > self.stale_timeout_s
            ]
            for vid in stale_ids:
                del self._table[vid]
            if stale_ids:
                logger.debug("Pruned %d stale neighbors", len(stale_ids))
            return len(stale_ids)

    def snapshot(self) -> Dict[str, VehicleState]:
        """Return a shallow copy of the current neighbor table."""
        with self._lock:
            return dict(self._table)

    def __len__(self) -> int:
        with self._lock:
            return len(self._table)

    def __contains__(self, vehicle_id: str) -> bool:
        with self._lock:
            return vehicle_id in self._table


class BsmListener:
    """BSM parser and neighbor table manager.

    Converts raw BSM fields into VehicleState objects and maintains
    the neighbor table with stale entry pruning.

    Turn rate estimation uses finite differences between successive BSMs:
        θ̇_v = (θ_v[t] - θ_v[t - Δt]) / Δt

    T_arrive estimation:
    - Vehicles already inside coverage: T_arrive = 0
    - Approaching vehicles: estimated from current bearing and speed to coverage edge
    """

    def __init__(
        self,
        rsu_x: float = 0.0,
        rsu_y: float = 0.0,
        coverage_radius: float = 300.0,
        stale_timeout_s: float = 0.2,
        gps_noise_sigma: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.coverage_radius = coverage_radius
        self.gps_noise_sigma = gps_noise_sigma
        self._rng = rng if rng is not None else np.random.default_rng()

        self.neighbor_table = NeighborTable(
            stale_timeout_s=stale_timeout_s,
            rsu_x=rsu_x,
            rsu_y=rsu_y,
            coverage_radius=coverage_radius,
        )

        # Statistics counters
        self.bsm_received: int = 0
        self.bsm_dropped: int = 0

    def process_bsm(
        self,
        vehicle_id: str,
        x: float,
        y: float,
        speed: float,
        heading: float,
        timestamp: float,
        drop: bool = False,
    ) -> Optional[VehicleState]:
        """Process a single BSM message.

        Args:
            vehicle_id: Unique vehicle identifier
            x, y: Reported position (meters)
            speed: Speed (m/s)
            heading: Heading angle (radians)
            timestamp: BSM timestamp (simulation seconds)
            drop: If True, simulate packet loss (BSM is discarded)

        Returns:
            Updated VehicleState, or None if dropped.
        """
        if drop:
            self.bsm_dropped += 1
            return None

        self.bsm_received += 1

        # Apply GPS noise
        if self.gps_noise_sigma > 0.0:
            x += self._rng.normal(0.0, self.gps_noise_sigma)
            y += self._rng.normal(0.0, self.gps_noise_sigma)

        # Estimate turn rate from previous heading
        prev_state = self.neighbor_table.get(vehicle_id)
        if prev_state is not None and prev_state.timestamp < timestamp:
            dt = timestamp - prev_state.timestamp
            if dt > 0:
                # Wrap heading difference to [-π, π]
                dtheta = _wrap_angle(heading - prev_state.heading)
                turn_rate = dtheta / dt
            else:
                turn_rate = prev_state.turn_rate
        else:
            turn_rate = 0.0

        # Estimate T_arrive
        t_arrive = self._estimate_t_arrive(x, y, speed, heading)

        state = VehicleState(
            vehicle_id=vehicle_id,
            x=x,
            y=y,
            speed=speed,
            heading=heading,
            turn_rate=turn_rate,
            timestamp=timestamp,
            t_arrive=t_arrive,
        )
        self.neighbor_table.update(state)
        return state

    def _estimate_t_arrive(
        self, x: float, y: float, speed: float, heading: float
    ) -> float:
        """Estimate time (seconds) until vehicle enters RSU coverage.

        Returns 0 if vehicle is already inside coverage.
        Returns a large value if the vehicle is not heading toward the RSU.
        """
        dist = math.hypot(x - self.rsu_x, y - self.rsu_y)
        if dist <= self.coverage_radius:
            return 0.0

        if speed < 0.1:  # Nearly stationary
            return float("inf")

        # Bearing from vehicle to RSU center
        dx = self.rsu_x - x
        dy = self.rsu_y - y
        bearing_to_rsu = math.atan2(dy, dx)

        # Angle between vehicle heading and bearing to RSU
        angle_diff = abs(_wrap_angle(bearing_to_rsu - heading))

        if angle_diff > math.pi / 2:
            # Vehicle is heading away from RSU
            return float("inf")

        # Approximate: project remaining distance on approach vector
        d_remain = (dist - self.coverage_radius) / math.cos(angle_diff)
        if d_remain < 0:
            return 0.0
        return max(0.0, d_remain / speed)

    def tick(self, current_time: float) -> int:
        """Prune stale entries. Call once per eviction cycle."""
        return self.neighbor_table.prune_stale(current_time)

    @property
    def reception_rate(self) -> float:
        """Fraction of BSMs successfully received (not dropped)."""
        total = self.bsm_received + self.bsm_dropped
        return self.bsm_received / total if total > 0 else 1.0


def _wrap_angle(angle: float) -> float:
    """Wrap angle to the interval [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def parse_sae_j2735_bsm(raw_bytes: bytes) -> Optional[Tuple]:
    """Parse a raw SAE J2735 BSM binary payload.

    In production this would decode the ASN.1 UPER-encoded BSM.
    This stub provides the interface contract.

    Returns:
        (vehicle_id, x_cm, y_cm, speed_ms, heading_deg, timestamp_ms) or None
    """
    # NOTE: Full SAE J2735 ASN.1 decoding requires a licensed codec.
    # In simulation, BSMs are injected as structured Python objects via TraCI.
    # For hardware deployment, use the Cohda MK5 SDK BSM decoder.
    raise NotImplementedError(
        "SAE J2735 binary decoding requires hardware-specific BSM codec. "
        "Use process_bsm() with pre-decoded fields from TraCI or BSM SDK."
    )
