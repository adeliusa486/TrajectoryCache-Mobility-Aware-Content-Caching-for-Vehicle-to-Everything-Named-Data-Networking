"""Unit tests for BSM Listener and Neighbor Table."""

import math
import time
import pytest

from trajectorycache.core.bsm_listener import BsmListener, NeighborTable, VehicleState


@pytest.fixture
def listener():
    return BsmListener(rsu_x=0.0, rsu_y=0.0, coverage_radius=300.0, stale_timeout_s=0.25)


class TestNeighborTable:
    def test_insert_and_get(self):
        nt = NeighborTable()
        state = VehicleState("v1", 10, 20, 5.0, 0.0, 0.0, 1.0, 0.0)
        nt.update(state)
        retrieved = nt.get("v1")
        assert retrieved is not None
        assert retrieved.x == 10
        assert retrieved.y == 20

    def test_prune_stale(self):
        nt = NeighborTable(stale_timeout_s=0.5)
        state = VehicleState("v1", 0, 0, 5.0, 0.0, 0.0, 0.0, 0.0)
        nt.update(state)
        assert len(nt) == 1
        removed = nt.prune_stale(current_time=1.0)
        assert removed == 1
        assert len(nt) == 0

    def test_snapshot_is_copy(self):
        nt = NeighborTable()
        state = VehicleState("v1", 0, 0, 5.0, 0.0, 0.0, 0.0, 0.0)
        nt.update(state)
        snap = nt.snapshot()
        assert "v1" in snap
        # Modifying snapshot does not affect table
        del snap["v1"]
        assert len(nt) == 1


class TestBsmListener:
    def test_basic_processing(self, listener):
        state = listener.process_bsm("v1", 10, 0, 20, 0, 0.5)
        assert state is not None
        assert state.vehicle_id == "v1"

    def test_turn_rate_estimation(self, listener):
        """Turn rate should be estimated from two successive BSMs."""
        listener.process_bsm("v1", 0, 0, 10, 0.0, 0.0)
        state2 = listener.process_bsm("v1", 1, 0, 10, 0.1, 0.1)
        assert state2 is not None
        # θ̇ = Δθ / Δt = 0.1 / 0.1 = 1.0 rad/s
        assert abs(state2.turn_rate - 1.0) < 0.01

    def test_dropped_bsm(self, listener):
        state = listener.process_bsm("v1", 0, 0, 10, 0, 0.5, drop=True)
        assert state is None
        assert listener.bsm_dropped == 1

    def test_reception_rate(self, listener):
        listener.process_bsm("v1", 0, 0, 10, 0, 0.1, drop=False)
        listener.process_bsm("v2", 5, 0, 10, 0, 0.1, drop=True)
        assert abs(listener.reception_rate - 0.5) < 0.01

    def test_t_arrive_inside_coverage(self, listener):
        """Vehicle inside coverage radius → T_arrive = 0."""
        state = listener.process_bsm("v1", 50, 0, 10, 0, 0.5)
        assert state.t_arrive == 0.0

    def test_t_arrive_outside_approaching(self, listener):
        """Approaching vehicle → finite positive T_arrive."""
        # Vehicle at x=500 heading West (toward RSU at 0,0), speed=20 m/s
        state = listener.process_bsm("v1", 500, 0, 20, math.pi, 0.5)
        assert state.t_arrive > 0
        assert state.t_arrive < float("inf")

    def test_t_arrive_receding(self, listener):
        """Receding vehicle → infinite T_arrive."""
        # Vehicle at x=500 heading East (away from RSU)
        state = listener.process_bsm("v1", 500, 0, 20, 0.0, 0.5)
        assert state.t_arrive == float("inf")

    def test_stale_pruning_via_tick(self, listener):
        listener.process_bsm("v1", 0, 0, 10, 0, 0.0)
        removed = listener.tick(current_time=1.0)
        assert removed == 1

    def test_gps_noise_applied(self):
        """GPS noise should perturb positions."""
        import numpy as np
        rng = np.random.default_rng(42)
        noisy_listener = BsmListener(gps_noise_sigma=5.0)
        positions = set()
        for i in range(10):
            s = noisy_listener.process_bsm("v1", 100.0, 100.0, 10, 0, float(i) * 0.1)
            positions.add((round(s.x, 1), round(s.y, 1)))
        # With noise, positions should vary
        assert len(positions) > 1

    def test_coverage_check(self, listener):
        state = VehicleState("v1", 100, 0, 10, 0, 0, 0, 0)
        assert state.is_inside_coverage(0, 0, 300)
        state2 = VehicleState("v2", 1000, 0, 10, 0, 0, 0, 0)
        assert not state2.is_inside_coverage(0, 0, 300)
