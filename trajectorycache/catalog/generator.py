"""Content Catalog Generation.

Generates the 12,500-item content catalog with:
- NDN name hierarchy: /v2x/{region}/{segment}/{type}/{chunk_id}
- Geographic tags (GRZ center) derived from road segment coordinates
- Zipf popularity distribution (α_z = 0.8)
- Content types: HD map tiles, traffic advisories, firmware chunks

Sizes per type:
  - HD map tiles:        ~80 KB  each, 10,000 items
  - Traffic advisories:  ~20 KB  each,  2,000 items
  - Firmware chunks:    ~400 KB  each,    500 items
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectorycache.core.content_store import ContentChunk

logger = logging.getLogger(__name__)


CONTENT_TYPES = {
    "hdmap": {"count": 10_000, "avg_size_kb": 80, "size_std_kb": 20},
    "traffic_advisory": {"count": 2_000, "avg_size_kb": 20, "size_std_kb": 5},
    "firmware": {"count": 500, "avg_size_kb": 400, "size_std_kb": 50},
}


@dataclass
class RoadSegment:
    """A road segment with a geographic center for GRZ tagging."""
    segment_id: str
    centroid_x: float  # Cartesian x (meters)
    centroid_y: float  # Cartesian y (meters)
    road_type: str = "highway"  # 'highway' or 'urban'
    length_m: float = 100.0


def generate_highway_segments(
    length_m: float = 5000.0,
    segment_length_m: float = 100.0,
    road_y: float = 0.0,
) -> List[RoadSegment]:
    """Generate road segments for a straight 5 km highway."""
    segments = []
    n = int(length_m / segment_length_m)
    for i in range(n):
        x = i * segment_length_m + segment_length_m / 2.0
        segments.append(RoadSegment(
            segment_id=f"hw_seg_{i:04d}",
            centroid_x=x,
            centroid_y=road_y,
            road_type="highway",
            length_m=segment_length_m,
        ))
    return segments


def generate_urban_segments(
    grid_size_m: float = 1000.0,
    block_size_m: float = 100.0,
) -> List[RoadSegment]:
    """Generate road segments for a 1×1 km Manhattan grid."""
    segments = []
    n_blocks = int(grid_size_m / block_size_m)
    seg_id = 0
    # Horizontal streets
    for row in range(n_blocks + 1):
        y = row * block_size_m
        for col in range(n_blocks):
            x = col * block_size_m + block_size_m / 2.0
            segments.append(RoadSegment(
                segment_id=f"urban_h_{seg_id:04d}",
                centroid_x=x,
                centroid_y=y,
                road_type="urban",
                length_m=block_size_m,
            ))
            seg_id += 1
    # Vertical streets
    for col in range(n_blocks + 1):
        x = col * block_size_m
        for row in range(n_blocks):
            y = row * block_size_m + block_size_m / 2.0
            segments.append(RoadSegment(
                segment_id=f"urban_v_{seg_id:04d}",
                centroid_x=x,
                centroid_y=y,
                road_type="urban",
                length_m=block_size_m,
            ))
            seg_id += 1
    return segments


def zipf_weights(n: int, alpha: float = 0.8) -> np.ndarray:
    """Generate normalised Zipf popularity weights for n items.

    P(rank k) ∝ 1/k^alpha
    """
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / (ranks ** alpha)
    return weights / weights.sum()


def generate_catalog(
    segments: List[RoadSegment],
    region: str = "riyadh",
    zipf_alpha: float = 0.8,
    rsu_x: float = 0.0,
    rsu_y: float = 0.0,
    rsu_radius: float = 300.0,
    random_seed: int = 42,
    r_grz: float = 300.0,
) -> List[ContentChunk]:
    """Generate the full content catalog.

    Returns:
        List of ContentChunk objects, one per catalog item.
    """
    rng = random.Random(random_seed)
    np_rng = np.random.default_rng(random_seed)

    chunks: List[ContentChunk] = []
    total_count = sum(v["count"] for v in CONTENT_TYPES.values())
    popularity_rank = 1

    for ctype, params in CONTENT_TYPES.items():
        count = params["count"]
        avg_size = int(params["avg_size_kb"] * 1024)
        std_size = int(params["size_std_kb"] * 1024)

        for i in range(count):
            # Assign a road segment (cycling through available segments)
            seg = segments[i % len(segments)]

            # NDN name hierarchy
            name = f"/v2x/{region}/{seg.segment_id}/{ctype}/chunk{i:05d}"

            # Chunk size with noise
            size_bytes = max(
                1024,
                int(np_rng.normal(avg_size, std_size))
            )

            chunk = ContentChunk(
                name=name,
                size_bytes=size_bytes,
                grz_x=seg.centroid_x,
                grz_y=seg.centroid_y,
                r_grz=r_grz,
                content_type=ctype,
                popularity_rank=popularity_rank,
            )
            chunks.append(chunk)
            popularity_rank += 1

    # Assign Zipf popularity ranks globally
    weights = zipf_weights(len(chunks), zipf_alpha)
    indices = np_rng.choice(len(chunks), size=len(chunks), replace=False, p=weights)
    for new_rank, idx in enumerate(indices, start=1):
        chunks[idx].popularity_rank = new_rank

    logger.info(
        "Generated catalog: %d chunks, %.1f MB total",
        len(chunks),
        sum(c.size_bytes for c in chunks) / (1024 * 1024),
    )
    return chunks


def save_catalog(chunks: List[ContentChunk], path: Path) -> None:
    """Serialize catalog to JSON."""
    data = [
        {
            "name": c.name,
            "size_bytes": c.size_bytes,
            "grz_x": c.grz_x,
            "grz_y": c.grz_y,
            "r_grz": c.r_grz,
            "content_type": c.content_type,
            "popularity_rank": c.popularity_rank,
        }
        for c in chunks
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d catalog entries to %s", len(chunks), path)


def load_catalog(path: Path) -> List[ContentChunk]:
    """Load catalog from JSON."""
    with open(path) as f:
        data = json.load(f)
    chunks = [
        ContentChunk(
            name=item["name"],
            size_bytes=item["size_bytes"],
            grz_x=item["grz_x"],
            grz_y=item["grz_y"],
            r_grz=item.get("r_grz", 300.0),
            content_type=item.get("content_type", "hdmap"),
            popularity_rank=item.get("popularity_rank", 1),
        )
        for item in data
    ]
    logger.info("Loaded %d catalog entries from %s", len(chunks), path)
    return chunks


def save_segment_table(segments: List[RoadSegment], path: Path) -> None:
    """Save OSM segment table as JSON."""
    data = {s.segment_id: {"x": s.centroid_x, "y": s.centroid_y, "type": s.road_type}
            for s in segments}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved segment table: %d entries → %s", len(segments), path)


def load_segment_table(path: Path) -> Dict[str, RoadSegment]:
    """Load segment table from JSON."""
    with open(path) as f:
        data = json.load(f)
    return {
        sid: RoadSegment(
            segment_id=sid,
            centroid_x=v["x"],
            centroid_y=v["y"],
            road_type=v.get("type", "highway"),
        )
        for sid, v in data.items()
    }
