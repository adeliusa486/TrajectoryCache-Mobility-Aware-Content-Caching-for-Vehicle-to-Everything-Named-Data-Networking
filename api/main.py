"""FastAPI monitoring and control interface for TrajectoryCache RSU.

Endpoints:
  GET  /health          — liveness probe
  GET  /metrics         — CS and eviction statistics
  GET  /neighbors       — current neighbor table
  GET  /config          — active configuration
  POST /config          — update configuration parameters
  POST /evict           — trigger manual eviction cycle
  GET  /experiments     — list saved experiment results
  POST /experiments/run — run a quick experiment

Used for:
  - Grafana/Prometheus scraping
  - Operator dashboards
  - CI/CD smoke testing
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trajectorycache.config import (
    ProjectConfig,
    TrajectoryCacheConfig,
    default_highway_config,
    load_config,
)
from trajectorycache.simulation.sim_loop import TrajectorySimulation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TrajectoryCache RSU API",
    description="Monitoring and control interface for the TrajectoryCache NDN cache replacement system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global simulation state (single RSU instance for demo)
# ---------------------------------------------------------------------------

_config: ProjectConfig = default_highway_config()
_sim: Optional[TrajectorySimulation] = None
_sim_running: bool = False
_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    version: str


class MetricsResponse(BaseModel):
    miss_rate: float
    hit_rate: float
    cs_occupancy: float
    total_interests: int
    cache_hits: int
    cache_misses: int
    eviction_cycles: int
    total_evicted: int
    mean_cycle_time_ms: float
    per_interest_overhead_us: float
    n_neighbors: int


class NeighborEntry(BaseModel):
    vehicle_id: str
    x: float
    y: float
    speed: float
    heading: float
    turn_rate: float
    t_arrive: float
    last_seen: float


class ConfigUpdateRequest(BaseModel):
    lambda_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    alpha: Optional[float] = Field(None, gt=0.0)
    eta_hw: Optional[float] = Field(None, gt=0.0, le=1.0)
    eta_lw: Optional[float] = Field(None, gt=0.0, le=1.0)
    r_grz: Optional[float] = Field(None, gt=0.0)


class ExperimentRequest(BaseModel):
    scenario: str = "highway"
    n_vehicles: int = Field(100, ge=10, le=500)
    duration_s: float = Field(60.0, ge=10.0, le=600.0)
    seed: int = 42
    policy: str = "tc"


class ExperimentResponse(BaseModel):
    policy: str
    scenario: str
    n_vehicles: int
    miss_rate: float
    mean_latency_ms: float
    isr: float
    per_interest_overhead_us: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        uptime_s=time.time() - _start_time,
        version="1.0.0",
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    global _sim
    if _sim is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    cs = _sim.cs
    engine = _sim.engine
    bsm = _sim.bsm_listener
    metrics = _sim.metrics

    return MetricsResponse(
        miss_rate=cs.stats.miss_rate,
        hit_rate=cs.stats.hit_rate,
        cs_occupancy=cs.occupancy,
        total_interests=cs.stats.total_interests,
        cache_hits=cs.stats.cache_hits,
        cache_misses=cs.stats.cache_misses,
        eviction_cycles=cs.stats.eviction_cycles,
        total_evicted=cs.stats.total_evicted,
        mean_cycle_time_ms=engine.average_cycle_time_ms,
        per_interest_overhead_us=metrics.per_interest_overhead_us,
        n_neighbors=len(bsm.neighbor_table),
    )


@app.get("/neighbors", response_model=List[NeighborEntry])
async def get_neighbors():
    global _sim
    if _sim is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")

    table = _sim.bsm_listener.neighbor_table.snapshot()
    return [
        NeighborEntry(
            vehicle_id=vid,
            x=state.x,
            y=state.y,
            speed=state.speed,
            heading=state.heading,
            turn_rate=state.turn_rate,
            t_arrive=state.t_arrive,
            last_seen=state.timestamp,
        )
        for vid, state in table.items()
    ]


@app.get("/config")
async def get_config():
    return {
        "trajectoryCache": {
            "lambda_weight": _config.tc.lambda_weight,
            "alpha": _config.tc.alpha,
            "eta_hw": _config.tc.eta_hw,
            "eta_lw": _config.tc.eta_lw,
            "r_grz": _config.tc.r_grz,
            "eviction_cycle_s": _config.tc.eviction_cycle_s,
        },
        "rsu": {
            "coverage_radius_m": _config.rsu.coverage_radius_m,
            "cs_capacity_mb": _config.rsu.cs_capacity_mb,
        },
        "simulation": {
            "scenario": _config.sim.scenario_name,
            "n_vehicles": _config.sim.num_vehicles,
        },
    }


@app.post("/config")
async def update_config(req: ConfigUpdateRequest):
    global _config, _sim
    if req.lambda_weight is not None:
        _config.tc.lambda_weight = req.lambda_weight
        if _sim:
            _sim.engine.lambda_weight = req.lambda_weight
    if req.alpha is not None:
        _config.tc.alpha = req.alpha
        if _sim:
            _sim.scorer.alpha = req.alpha
    if req.eta_hw is not None:
        _config.tc.eta_hw = req.eta_hw
        if _sim:
            _sim.engine.eta_hw = req.eta_hw
    if req.eta_lw is not None:
        _config.tc.eta_lw = req.eta_lw
        if _sim:
            _sim.engine.eta_lw = req.eta_lw
    if req.r_grz is not None:
        _config.tc.r_grz = req.r_grz
        if _sim:
            _sim.scorer.r_grz = req.r_grz
    return {"status": "updated", "config": await get_config()}


@app.post("/evict")
async def trigger_eviction():
    global _sim
    if _sim is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    states = _sim.bsm_listener.neighbor_table.snapshot()
    result = _sim.engine.run_cycle(states, current_time=time.time())
    return {
        "triggered": result.triggered,
        "n_evicted": result.n_evicted,
        "pre_occupancy": result.pre_occupancy,
        "post_occupancy": result.post_occupancy,
        "cycle_time_ms": result.cycle_time_ms,
    }


@app.post("/experiments/run", response_model=ExperimentResponse)
async def run_experiment(req: ExperimentRequest):
    """Run a quick simulation experiment and return results."""
    from trajectorycache.config import default_highway_config, default_urban_config
    from trajectorycache.evaluation.experiments import run_policy

    if req.scenario == "highway":
        cfg = default_highway_config()
    elif req.scenario == "urban":
        cfg = default_urban_config()
    else:
        raise HTTPException(status_code=400, detail="scenario must be 'highway' or 'urban'")

    cfg.sim.num_vehicles = req.n_vehicles
    cfg.sim.duration_s = req.duration_s
    cfg.sim.warmup_s = min(30.0, req.duration_s / 4)
    cfg.sim.random_seed = req.seed

    if req.policy not in ["tc", "lru", "lfu", "probcache", "wave"]:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {req.policy}")

    try:
        result = run_policy(req.policy, cfg, req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ExperimentResponse(
        policy=result.policy,
        scenario=result.scenario,
        n_vehicles=result.n_vehicles,
        miss_rate=result.miss_rate,
        mean_latency_ms=result.mean_latency_ms,
        isr=result.isr,
        per_interest_overhead_us=result.per_interest_overhead_us,
    )


@app.on_event("startup")
async def startup_event():
    global _sim, _config
    logger.info("TrajectoryCache API starting up")
    # Initialize a demo simulation for live metrics
    _config = default_highway_config()
    _config.sim.num_vehicles = 50
    _config.sim.duration_s = 300.0
    _sim = TrajectorySimulation(_config)
    logger.info("Demo simulation initialized")
