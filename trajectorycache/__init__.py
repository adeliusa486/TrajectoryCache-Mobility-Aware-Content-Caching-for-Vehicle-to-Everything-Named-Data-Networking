"""
TrajectoryCache: Mobility-Aware Content Caching for NDN-based V2X Networks.

This Python package provides:
- Pure-Python simulation of the TrajectoryCache algorithm
- BSM processing and neighbor table management
- CTRV trajectory prediction
- MRS scoring with spatial indexing
- Eviction engine (composite score)
- Content catalog generation
- Evaluation metrics and plotting
"""

__version__ = "1.0.0"
__author__ = "Ahmad Al-Rashidi, Layla Mansour, Omar Siddiqui, Fatima Al-Zahrawi"
__license__ = "MIT"

from trajectorycache.core.bsm_listener import BsmListener, VehicleState, NeighborTable
from trajectorycache.core.ctrv_predictor import CTRVPredictor
from trajectorycache.core.mrs_scorer import MrsScorer
from trajectorycache.core.affinity_estimator import AffinityEstimator
from trajectorycache.core.eviction_engine import EvictionEngine
from trajectorycache.core.content_store import ContentStore, ContentChunk
from trajectorycache.config import TrajectoryCacheConfig, RsuConfig, load_config

__all__ = [
    "BsmListener",
    "VehicleState",
    "NeighborTable",
    "CTRVPredictor",
    "MrsScorer",
    "AffinityEstimator",
    "EvictionEngine",
    "ContentStore",
    "ContentChunk",
    "TrajectoryCacheConfig",
    "RsuConfig",
    "load_config",
]
