"""Content Affinity Estimator φ(v, c).

Maintains per-vehicle content affinity scores using an Exponential Moving
Average (EMA) over a W = 300 s window.

Update rule (inferred from paper description):
    φ_new(v, c) = (1 - β) · φ_old(v, c) + β · 𝟙[v requested c]
    where β = 1/W = 1/300

Cold-start prior: φ(v, c) = catalog_mean_prior for new vehicle/chunk pairs.

Privacy note: The affinity table maps (vehicle_id, chunk_name) → float.
In deployment this table should be held in volatile memory only and flushed
on pseudonym rotation (per ETSI ITS TS 102 941).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Type alias for (vehicle_id, chunk_name) key
AffinityKey = Tuple[str, str]


class AffinityEstimator:
    """EMA-based per-vehicle content affinity tracker.

    Thread-safety note: Not thread-safe by default. In production,
    wrap with an external lock if updated from a BSM listener thread.
    """

    def __init__(
        self,
        beta: float = 1.0 / 300.0,
        catalog_mean_prior: float = 0.1,
        max_entries: int = 500_000,
    ) -> None:
        """
        Args:
            beta: EMA learning rate. Default = 1/300 (300 s window).
            catalog_mean_prior: Initial φ value for new (vehicle, chunk) pairs.
            max_entries: Maximum table size; oldest entries evicted when exceeded.
        """
        self.beta = beta
        self.catalog_mean_prior = catalog_mean_prior
        self.max_entries = max_entries

        # (vehicle_id, chunk_name) → affinity score in [0, 1]
        self._table: Dict[AffinityKey, float] = {}

        # Stats
        self.update_count: int = 0
        self.evictions: int = 0

    def get(self, vehicle_id: str, chunk_name: str) -> float:
        """Return current affinity φ(v, c). Returns prior for unseen pairs."""
        return self._table.get((vehicle_id, chunk_name), self.catalog_mean_prior)

    def update(
        self,
        vehicle_id: str,
        chunk_name: str,
        requested: bool,
    ) -> float:
        """Update affinity via EMA and return the new value.

        Args:
            vehicle_id: Vehicle identifier.
            chunk_name: NDN name of the content chunk.
            requested: True if vehicle issued an Interest for this chunk.

        Returns:
            Updated φ(v, c).
        """
        key = (vehicle_id, chunk_name)
        phi_old = self._table.get(key, self.catalog_mean_prior)
        phi_new = (1.0 - self.beta) * phi_old + self.beta * float(requested)
        # Clamp to [0, 1]
        phi_new = max(0.0, min(1.0, phi_new))
        self._table[key] = phi_new
        self.update_count += 1

        if len(self._table) > self.max_entries:
            self._evict_oldest()

        return phi_new

    def decay_all(self, n_steps: int = 1) -> None:
        """Apply EMA decay to all entries without a positive signal.

        Call once per eviction cycle to implement the passive decay:
            φ_new = (1 - β)^n · φ_old
        """
        factor = (1.0 - self.beta) ** n_steps
        for key in list(self._table.keys()):
            self._table[key] *= factor
            # Prune near-zero entries to save memory
            if self._table[key] < 1e-6:
                del self._table[key]

    def flush_vehicle(self, vehicle_id: str) -> int:
        """Remove all affinity entries for a vehicle (e.g., on pseudonym rotation).

        Returns count of entries removed.
        """
        keys_to_remove = [k for k in self._table if k[0] == vehicle_id]
        for key in keys_to_remove:
            del self._table[key]
        logger.debug("Flushed %d affinity entries for vehicle %s", len(keys_to_remove), vehicle_id)
        return len(keys_to_remove)

    def get_vehicle_affinities(self, vehicle_id: str) -> Dict[str, float]:
        """Return all chunk affinities for a given vehicle."""
        return {
            chunk_name: phi
            for (vid, chunk_name), phi in self._table.items()
            if vid == vehicle_id
        }

    def _evict_oldest(self) -> None:
        """Evict 10% of entries with lowest affinity to stay under max_entries."""
        n_evict = len(self._table) // 10
        sorted_keys = sorted(self._table, key=lambda k: self._table[k])
        for key in sorted_keys[:n_evict]:
            del self._table[key]
        self.evictions += n_evict

    @property
    def table_size(self) -> int:
        return len(self._table)

    def __repr__(self) -> str:
        return (
            f"AffinityEstimator(beta={self.beta:.5f}, "
            f"entries={self.table_size}, updates={self.update_count})"
        )
