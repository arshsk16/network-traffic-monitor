# Network Traffic Monitoring & Intelligent Path Selection

> **Disclaimer**: This project is a **learning exercise** inspired by SD-WAN (Software-Defined Wide Area Network)
> path-selection concepts. It is **not** a production SD-WAN system, does not implement BGP/OSPF, MPLS, or
> any real routing protocol, and makes no claim to be a replacement for commercial SD-WAN products.
> It was built to develop and demonstrate networking and backend engineering fundamentals.

---

## What This Project Does

The system monitors multiple network paths (e.g., two internet uplinks), measures their quality, scores each path
using a transparent weighted formula, and selects the better available path. It exposes the results through a
REST API built with FastAPI.

**Metrics eventually measured:**
- Latency (round-trip time)
- Packet loss
- Jitter (variation in latency)
- Throughput

**Path scoring:**
Metrics are normalized and combined with configurable weights. Lower latency/loss/jitter → better score.
Higher throughput → better score. No ML — the formula is explicit and readable.

---

## Project Status

This project is built **one step at a time**, with each step explained and committed independently.

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup, FastAPI skeleton, health endpoint, tests | ✅ Done |
| 2 | Basic TCP/UDP network probing | 🔲 Planned |
| 3 | Latency measurement | 🔲 Planned |
| 4 | Packet loss & timeout detection | 🔲 Planned |
| 5 | Jitter measurement | 🔲 Planned |
| 6 | Throughput measurement | 🔲 Planned |
| 7 | Multiple paths support | 🔲 Planned |
| 8 | Configurable weighted path scoring | 🔲 Planned |
| 9 | Path selection & failure handling | 🔲 Planned |
| 10 | Full metrics/selection API endpoints | 🔲 Planned |
| 11 | Integration tests, Docker (if useful), docs | 🔲 Planned |

---

## Architecture

```
FastAPI (REST API)
     │
     ▼
Monitoring Engine (asyncio + Python sockets)
     │   measures: latency, loss, jitter, throughput
     ▼
Path Scoring Engine (weighted formula, configurable)
     │   selects: Path A or Path B (or N paths)
     ▼
Network Paths (TCP/UDP endpoints)
```

The API layer, monitoring engine, and scoring engine are kept strictly separated so each can be understood,
tested, and changed independently.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI + uvicorn | Fast, async-native, excellent OpenAPI docs |
| Monitoring | Python asyncio + sockets | Teaches the fundamentals directly, no library magic |
| Config | pydantic-settings | Typed, validated, env-variable driven |
| Testing | pytest + httpx TestClient | Simple, readable, standard |
| Runtime | Python 3.12 | Latest stable; asyncio improvements relevant here |

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Run the development server

```bash
# Option A — convenience script
python run.py

# Option B — uvicorn directly
uvicorn app.main:app --reload --port 8000
```

### 4. Check the API

- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs

### 5. Run tests

```bash
pytest -v
```

---

## Repository Structure

```
netmon/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app factory (create_app)
│   ├── config.py        # All settings via pydantic-settings
│   └── routers/
│       ├── __init__.py
│       └── health.py    # GET /health endpoint
├── core/                # Future: monitoring engine, scoring engine
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # Shared pytest fixtures (TestClient)
│   └── test_health.py   # Tests for /health
├── .env.example         # Document available env vars
├── .gitignore
├── pyproject.toml       # Dependencies, build config, pytest config
├── run.py               # Dev server entry point
└── README.md
```

---

## Design Decisions

**Why `create_app()` instead of a bare module-level `app`?**
Tests can call `create_app()` to get a fresh, isolated instance. This prevents test pollution and makes
it trivial to inject different settings per test.

**Why `pydantic-settings`?**
All configuration is typed and validated at startup. If you set `PORT=abc`, the app crashes immediately
with a clear error instead of failing mysteriously later.

**Why keep `core/` empty for now?**
The monitoring and scoring logic doesn't exist yet. Creating an empty `core/` directory makes the intended
separation visible from the start, without adding premature abstractions.

---

## What This Is NOT

- Not a BGP, OSPF, or MPLS implementation
- Not a production-grade SD-WAN system
- Not a replacement for commercial products like Cisco Viptela, VMware VeloCloud, or Palo Alto Prisma SD-WAN
- Not a Kubernetes deployment
- Not a machine-learning system

---

## Learning Goals

By the end of this project you should be able to explain:
- TCP vs UDP — when and why each is used for probing
- How latency, packet loss, jitter, and throughput are measured programmatically
- Why asyncio is useful for concurrent network probing
- How a simple transparent weighted scoring formula works
- How to detect path degradation and trigger failover
- The limitations of this approach vs real SD-WAN
