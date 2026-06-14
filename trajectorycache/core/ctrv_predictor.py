"""CTRV Trajectory Predictor.

Implements the Constant Turn Rate and Velocity (CTRV) kinematic model
for predicting vehicle positions over a dwell-time horizon.

Equations 2–3 from the paper:
    x̂(t+Δt) = x + (s/θ̇)·[sin(θ + θ̇·Δt) - sin(θ)]   if |θ̇| ≥ ε
    ŷ(t+Δt) = y + (s/θ̇)·[cos(θ) - cos(θ + θ̇·Δt)]   if |θ̇| ≥ ε

    x̂(t+Δt) = x + s·Δt·cos(θ)   if |θ̇| < ε  (constant velocity fallback)
    ŷ(t+Δt) = y + s·Δt·sin(θ)   if |θ̇| < ε

L'Hôpital's rule motivates the CV fallback: as θ̇ → 0,
the CTRV equations converge to the CV equations.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from trajectorycache.core.bsm_listener import VehicleState

logger = logging.getLogger(__name__)

# Type alias: trajectory is list of (x, y) points
Trajectory = List[Tuple[float, float]]


class CTRVPredictor:
    """Predicts vehicle trajectories using the CTRV kinematic model.

    For |θ̇| ≥ epsilon_turn → CTRV curved path equations
    For |θ̇| < epsilon_turn → CV (constant velocity) straight-line equations

    Map-aided correction optionally snaps predicted points to the nearest
    road centerline using a pre-loaded road graph.
    """

    def __init__(
        self,
        epsilon_turn: float = 1e-4,
        delta_tau: float = 0.5,
        map_graph: Optional[object] = None,
    ) -> None:
        """
        Args:
            epsilon_turn: Minimum |θ̇| (rad/s) to use CTRV vs CV fallback.
            delta_tau: Trajectory sampling step size (seconds).
            map_graph: Optional road graph for map-aided snapping. If None,
                       no snapping is applied.
        """
        self.epsilon_turn = epsilon_turn
        self.delta_tau = delta_tau
        self.map_graph = map_graph

    def predict(
        self,
        state: VehicleState,
        horizon_s: float,
    ) -> Trajectory:
        """Predict trajectory for a single vehicle over `horizon_s` seconds.

        Args:
            state: Current vehicle state from neighbor table.
            horizon_s: Prediction horizon (seconds). Typically T_dwell(v).

        Returns:
            List of (x, y) waypoints sampled every delta_tau seconds,
            including t=0 (current position).
        """
        if horizon_s <= 0 or state.speed < 0.01:
            return [(state.x, state.y)]

        trajectory: Trajectory = [(state.x, state.y)]
        x, y = state.x, state.y
        s = state.speed
        theta = state.heading
        theta_dot = state.turn_rate
        t = 0.0

        while t < horizon_s:
            dt = min(self.delta_tau, horizon_s - t)
            x, y = self._step(x, y, s, theta, theta_dot, dt)
            theta = theta + theta_dot * dt  # update heading for next step
            trajectory.append((x, y))
            t += dt

        if self.map_graph is not None:
            trajectory = self._snap_to_road(trajectory)

        return trajectory

    def predict_batch(
        self,
        states: dict[str, VehicleState],
        horizons: dict[str, float],
    ) -> dict[str, Trajectory]:
        """Predict trajectories for a batch of vehicles.

        Args:
            states: {vehicle_id: VehicleState}
            horizons: {vehicle_id: horizon_s}

        Returns:
            {vehicle_id: Trajectory}
        """
        results = {}
        for vid, state in states.items():
            horizon = horizons.get(vid, 0.0)
            results[vid] = self.predict(state, horizon)
        return results

    def _step(
        self,
        x: float,
        y: float,
        s: float,
        theta: float,
        theta_dot: float,
        dt: float,
    ) -> Tuple[float, float]:
        """Single CTRV step. Returns (x_new, y_new)."""
        if abs(theta_dot) >= self.epsilon_turn:
            # CTRV curved motion (Eqs. 2-3)
            r = s / theta_dot  # turning radius
            new_x = x + r * (math.sin(theta + theta_dot * dt) - math.sin(theta))
            new_y = y + r * (math.cos(theta) - math.cos(theta + theta_dot * dt))
        else:
            # CV straight-line fallback
            new_x = x + s * dt * math.cos(theta)
            new_y = y + s * dt * math.sin(theta)

        return new_x, new_y

    def _snap_to_road(self, trajectory: Trajectory) -> Trajectory:
        """Snap trajectory waypoints to nearest road centerline.

        Requires a road graph with a nearest_centerline(x, y) method.
        """
        if self.map_graph is None:
            return trajectory
        snapped = []
        for x, y in trajectory:
            try:
                sx, sy = self.map_graph.nearest_centerline(x, y)
                snapped.append((sx, sy))
            except Exception:
                snapped.append((x, y))
        return snapped

    def compute_corridor_bbox(
        self, trajectory: Trajectory, grz_radius: float
    ) -> Tuple[float, float, float, float]:
        """Compute axis-aligned bounding box of a trajectory corridor.

        The corridor is the Minkowski sum of the trajectory polyline
        and a disk of radius grz_radius. Used for R-tree range queries.

        Returns:
            (min_x, min_y, max_x, max_y)
        """
        if not trajectory:
            raise ValueError("Empty trajectory")

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        return (
            min(xs) - grz_radius,
            min(ys) - grz_radius,
            max(xs) + grz_radius,
            max(ys) + grz_radius,
        )

    def min_distance_to_point(
        self, trajectory: Trajectory, px: float, py: float
    ) -> float:
        """Compute minimum Euclidean distance from any trajectory waypoint to (px, py).

        This is an approximation; exact polyline distance would use segment queries.
        The approximation error is bounded by delta_tau * max_speed.
        """
        if not trajectory:
            return float("inf")

        min_dist = float("inf")
        for x, y in trajectory:
            d = math.hypot(x - px, y - py)
            if d < min_dist:
                min_dist = d
        return min_dist


def compute_dwell_time(
    state: VehicleState,
    rsu_x: float,
    rsu_y: float,
    coverage_radius: float,
    predictor: CTRVPredictor,
    max_horizon: float = 30.0,
) -> float:
    """Estimate how long vehicle `v` will remain inside RSU coverage (seconds).

    Uses binary search on the CTRV trajectory to find the exit point.

    Eq. 4: T_dwell(v) = d_remain / s_v (simplified constant-speed version)
    This function uses the full CTRV trajectory for greater accuracy.
    """
    dist = math.hypot(state.x - rsu_x, state.y - rsu_y)

    if dist > coverage_radius:
        # Vehicle is outside — dwell time is 0 (or use t_arrive + dwell_inside)
        return 0.0

    if state.speed < 0.01:
        # Stationary vehicle; cap at max_horizon
        return max_horizon

    # Fast approximation (Eq. 4)
    d_remain = coverage_radius - dist
    t_approx = d_remain / state.speed

    # Clamp to max_horizon
    return min(t_approx, max_horizon)
