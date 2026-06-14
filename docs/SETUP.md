# Full Setup Guide

## 1. Python Library (No SUMO/NS-3 Required)

```bash
git clone https://github.com/trajectorycache/trajectorycache.git
cd trajectorycache

# Install Python package
pip install -e ".[dev]"

# Optional: install rtree for 12-18x spatial indexing speedup
sudo apt-get install libspatialindex-dev
pip install rtree

# Verify
python scripts/smoke_test.py
make demo
```

---

## 2. Full Co-Simulation Stack (SUMO + ndnSIM)

### 2a. SUMO 1.14

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
sumo --version   # should show 1.14.x
export SUMO_HOME=/usr/share/sumo
```

### 2b. NS-3.36 + ndnSIM 2.8

Following the official ndnSIM installation guide:

```bash
# Install dependencies
sudo apt-get install -y build-essential libsqlite3-dev libboost-all-dev \
    libssl-dev git python3-dev pkg-config

# Get ndnSIM
mkdir ~/ndnSIM && cd ~/ndnSIM
git clone https://github.com/named-data-ndnSIM/ns-3 --branch ndnSIM-2.8 ns-3
git clone https://github.com/named-data-ndnSIM/ndnSIM --branch v0.23 ns-3/src/ndnSIM

cd ns-3
./waf configure --enable-examples --enable-tests
./waf build -j$(nproc)

# Set environment variable
export NS3_DIR=$HOME/ndnSIM/ns-3
```

### 2c. TrajectoryCache ndnSIM module

```bash
# The C++ core is in src/core/
cd ~/ndnSIM/ns-3
cp -r /path/to/trajectorycache/src/core src/trajectory-cache
./waf configure
./waf build -j$(nproc)
```

### 2d. Run a co-simulation scenario

```bash
cd ~/ndnSIM/ns-3
./waf --run "trajectory-cache-s1 --nVehicles=300 --simDuration=600 --seed=42"
```

---

## 3. Scenario Files

### S1 — 5 km Straight Highway

| Parameter | Value |
|---|---|
| Road length | 5 km |
| Lanes | 3 (bidirectional) |
| Speed limit | 120 km/h |
| RSU position | 2500 m midpoint |
| Vehicle density | 100–500 |

SUMO config: `scenarios/s1_highway/sumo/highway.sumocfg`

### S2 — 1×1 km Urban Grid

| Parameter | Value |
|---|---|
| Grid size | 1000×1000 m |
| Block size | 100×100 m |
| Speed limit | 50 km/h |
| RSU position | 500, 500 (center) |
| Turn probability | 0.3 per intersection |

SUMO config: `scenarios/s2_urban/sumo/urban.sumocfg`

---

## 4. Monitoring (Prometheus + Grafana)

```bash
docker-compose --profile monitoring up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

---

## 5. Troubleshooting

**`ImportError: No module named rtree`**
→ Install libspatialindex: `sudo apt-get install libspatialindex-dev && pip install rtree`

**`ModuleNotFoundError: No module named 'scipy'`**
→ `pip install scipy`

**`FileNotFoundError: configs/highway_default.json`**
→ Run scripts from the repository root: `cd /path/to/trajectorycache && make smoke`

**ndnSIM build fails with C++17 errors**
→ Ensure GCC ≥ 9: `sudo apt-get install gcc-11 g++-11` and re-run `./waf configure --cxx-standard=c++17`
