"""Unit tests for the CTRV trajectory predictor."""

import math
import pytest

from trajectorycache.core.bsm_listener import VehicleState
from trajectorycache.core.ctrv_predictor import CTRVPredictor, compute_dwell_time


def make_state(
    x=0.0, y=0.0, speed=10.0, heading=0.0, turn_rate=0.0, t_arrive=0.0
) -> VehicleState:
    return VehicleState(
        vehicle_id="veh_0",
        x=x, y=y, speed=speed,
        heading=heading, turn_rate=turn_rate,
        timestamp=0.0, t_arrive=t_arrive,
    )


@pytest.fixture
def predictor():
    return CTRVPredictor(epsilon_turn=1e-4, delta_tau=0.5)


class TestCTRVStep:
    def test_straight_line_cv(self, predictor):
        """Zero turn rate → straight-line constant velocity."""
        state = make_state(x=0, y=0, speed=10.0, heading=0.0, turn_rate=0.0)
        traj = predictor.predict(state, horizon_s=1.0)

        assert len(traj) >= 2
        # Last point should be ~10 m along heading=0 (East)
        x_final, y_final = traj[-1]
        assert abs(x_final - 10.0) < 0.1, f"x_final={x_final}"
        assert abs(y_final) < 0.1, f"y_final={y_final}"

    def test_curved_ctrv(self, predictor):
        """Non-zero turn rate → curved arc."""
        # θ̇ = π/4 rad/s, speed = 10 m/s, heading = 0
        state = make_state(x=0, y=0, speed=10.0, heading=0.0, turn_rate=math.pi / 4)
        traj = predictor.predict(state, horizon_s=2.0)

        assert len(traj) >= 4
        # Vehicle should have turned; y should be non-zero
        x_final, y_final = traj[-1]
        assert abs(y_final) > 0.5, f"Expected curved trajectory, got y_final={y_final}"

    def test_zero_speed(self, predictor):
        """Stationary vehicle → single point."""
        state = make_state(x=5, y=5, speed=0.0)
        traj = predictor.predict(state, horizon_s=5.0)
        assert traj == [(5.0, 5.0)]

    def test_zero_horizon(self, predictor):
        """Zero horizon → single current position."""
        state = make_state(x=3, y=7)
        traj = predictor.predict(state, horizon_s=0.0)
        assert traj == [(3.0, 7.0)]

    def test_fallback_near_zero_turn_rate(self, predictor):
        """Turn rate just below epsilon → CV fallback."""
        state = make_state(x=0, y=0, speed=10.0, heading=0.0, turn_rate=5e-5)
        traj = predictor.predict(state, horizon_s=1.0)
        x_final, y_final = traj[-1]
        # Should still be approximately straight
        assert abs(x_final - 10.0) < 0.5

    def test_heading_north(self, predictor):
        """Heading π/2 (North) → movement in +y direction."""
        state = make_state(x=0, y=0, speed=10.0, heading=math.pi / 2, turn_rate=0.0)
        traj = predictor.predict(state, horizon_s=1.0)
        x_final, y_final = traj[-1]
        assert abs(y_final - 10.0) < 0.1
        assert abs(x_final) < 0.1

    def test_trajectory_length(self, predictor):
        """Trajectory has correct number of waypoints."""
        state = make_state(speed=10.0)
        horizon = 3.0
        traj = predictor.predict(state, horizon_s=horizon)
        # Should have ~(horizon / delta_tau) + 1 points
        expected_min = int(horizon / predictor.delta_tau)
        assert len(traj) >= expected_min


class TestBBox:
    def test_bbox_contains_trajectory(self, predictor):
        state = make_state(speed=10.0, heading=0.0)
        traj = predictor.predict(state, horizon_s=5.0)
        grz = 300.0
        bbox = predictor.compute_corridor_bbox(traj, grz)
        min_x, min_y, max_x, max_y = bbox

        for x, y in traj:
            assert min_x <= x <= max_x
            assert min_y <= y <= max_y

    def test_empty_trajectory_raises(self, predictor):
        with pytest.raises(ValueError):
            predictor.compute_corridor_bbox([], 100.0)


class TestMinDistance:
    def test_single_point(self, predictor):
        traj = [(0.0, 0.0)]
        d = predictor.min_distance_to_point(traj, 3.0, 4.0)
        assert abs(d - 5.0) < 1e-9

    def test_multiple_points(self, predictor):
        traj = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        d = predictor.min_distance_to_point(traj, 10.0, 3.0)
        assert abs(d - 3.0) < 1e-9

    def test_empty_trajectory(self, predictor):
        d = predictor.min_distance_to_point([], 5.0, 5.0)
        assert d == float("inf")


class TestDwellTime:
    def test_vehicle_inside_coverage(self, predictor):
        """Vehicle already inside RSU → dwell time > 0."""
        state = make_state(x=100, y=0, speed=20.0, heading=0.0)
        t = compute_dwell_time(state, rsu_x=0, rsu_y=0, coverage_radius=300, predictor=predictor)
        assert t > 0

    def test_vehicle_outside_coverage(self, predictor):
        """Vehicle far outside RSU → dwell time 0."""
        state = make_state(x=1000, y=0, speed=20.0)
        t = compute_dwell_time(state, rsu_x=0, rsu_y=0, coverage_radius=300, predictor=predictor)
        assert t == 0.0

    def test_stationary_vehicle(self, predictor):
        """Stationary vehicle inside coverage → capped at max_horizon."""
        state = make_state(x=50, y=0, speed=0.0)
        t = compute_dwell_time(state, rsu_x=0, rsu_y=0, coverage_radius=300,
                               predictor=predictor, max_horizon=30.0)
        assert t == 30.0
