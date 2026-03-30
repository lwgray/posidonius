<h1 align="center">Posidonius</h1>

<p align="center">
  <strong>The experiment dashboard for multi-agent coordination at scale.</strong>
</p>

<p align="center">
  <a href="#get-started"><img src="https://img.shields.io/badge/Get_Started-5_min-blue?style=for-the-badge" alt="Get Started"></a>
  <a href="#see-it-work"><img src="https://img.shields.io/badge/See_It_Work-Demo-green?style=for-the-badge" alt="See It Work"></a>
  <a href="#ecosystem"><img src="https://img.shields.io/badge/Ecosystem-Marcus+Cato-purple?style=for-the-badge" alt="Ecosystem"></a>
</p>

<p align="center">
  <a href="https://github.com/lwgray/posidonius"><img src="https://img.shields.io/github/stars/lwgray/posidonius?style=social" alt="GitHub Stars"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/version-0.1.1-brightgreen" alt="Version 0.1.1">
</p>

---

## The Question That Started Everything

You can coordinate multiple AI agents with [Marcus](https://github.com/lwgray/marcus).
But how do you know if it actually works better with 3 agents or 10?

**You measure it.**

Posidonius is a web dashboard for running, monitoring, and comparing multi-agent
experiments. It launches independent agents via tmux, tracks metrics with MLflow,
and gives you a clean interface to answer the scaling question empirically.

Named after the Stoic philosopher known for empirical observation and measurement
-- fitting for a tool that measures whether multi-agent coordination actually
works at scale.

---

## See It Work

```
You: Configure an experiment — "Build a Snake game" with 3, 5, and 10 agents
```

What happens:

1. Posidonius creates a sequential pipeline: Run 1 (3 agents) -> Run 2 (5 agents) -> Run 3 (10 agents)
2. Each run generates a config, writes a project spec, and launches agents in tmux
3. The dashboard shows live agent output — who's working, who's waiting, who's done
4. MLflow tracks metrics for each run: completion time, task counts, blockers
5. When a run finishes, the next one starts automatically
6. You compare results: did 10 agents actually finish faster than 3?

**One form. One button. Walk away and come back to data.**

---

## What You Get

- **Sequential experiment pipelines** — run scaling tests (3 agents -> 5 -> 10) in sequence, automatically
- **Live agent monitoring** — real-time tmux pane capture showing what each agent is doing
- **Agent status detection** — working / waiting / complete / error / idle
- **Interactive terminals** — WebSocket-based shell access to individual agent panes
- **MLflow experiment tracking** — parent/child runs with metrics for each scaling level
- **Optimal agent estimation** — pre-flight CPM analysis via Marcus MCP to recommend team size
- **Task progress strip** — completed tasks, percentage, active agents, blockers at a glance
- **History tab** — browse past experiments from MLflow with metrics comparison
- **Export** — download all agent output as a ZIP file
- **Auto-advance** — optionally auto-complete runs and start the next one

---

## Get Started

**Prerequisites:**
- Python 3.11+
- tmux (`brew install tmux` on macOS)
- [Marcus](https://github.com/lwgray/marcus) MCP server running (for experiments)
- `run_experiment.py` available from Marcus (`dev-tools/experiments/runners/`)

### Step 1: Install

```bash
git clone https://github.com/lwgray/posidonius.git
cd posidonius
pip install -e ".[dev]"
```

### Step 2: Start the Dashboard

```bash
# Default: http://localhost:8420
python -m posidonius

# Or with custom options
python -m posidonius --host 0.0.0.0 --port 9000 \
  --experiments-dir ~/experiments \
  --templates-dir ~/marcus/templates
```

### Step 3: Using the CLI Wrapper

```bash
./pos start              # Start as daemon
./pos start -f           # Run in foreground
./pos stop               # Stop daemon
./pos restart            # Restart daemon
./pos logs               # View logs
./pos status             # Check status
```

### Step 4: Run Your First Experiment

1. Open http://localhost:8420
2. Fill in the form: project name, specification, complexity, agent counts
3. Click "Run Experiment"
4. Watch agents work in real time

---

## Architecture

Posidonius is a standalone project that communicates with Marcus across process
boundaries. No shared memory. No direct Python imports. If Marcus runs on a
remote server and Posidonius runs on your laptop, it still works.

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (index.html)                      │
│                  Single-Page Application                     │
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP REST + WebSocket
                            v
┌─────────────────────────────────────────────────────────────┐
│              Posidonius FastAPI Server (app.py)              │
│                                                             │
│  POST /api/experiments           Create pipeline            │
│  POST /api/experiments/{n}/start Launch a run               │
│  GET  /api/experiments/{n}/output  Live agent output        │
│  POST /api/experiments/optimize  CPM analysis via MCP       │
│  GET  /api/experiments/history   Past experiments           │
│  WS   /terminal/{pane}          Interactive shell           │
└───────┬────────────┬────────────┬──────────────┬────────────┘
        │            │            │              │
        v            v            v              v
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
  │ Pipeline │ │  Tmux    │ │ MLflow   │ │  Marcus    │
  │ Engine   │ │ Manager  │ │ Tracker  │ │ MCP (HTTP) │
  └──────────┘ └──────────┘ └──────────┘ └────────────┘
```

### Process Boundaries

| Boundary | Mechanism |
|----------|-----------|
| Posidonius -> Marcus agents | subprocess -> `run_experiment.py` |
| Posidonius -> Marcus MCP | httpx -> `localhost:4298/mcp` (optimizer only) |
| Posidonius -> tmux | subprocess -> `tmux capture-pane / send-keys` |
| Posidonius -> MLflow | mlflow Python SDK -> local `mlruns/` directory |

### Component Overview

```
posidonius/
├── __main__.py              # CLI entry point (uvicorn server)
├── app.py                   # FastAPI application + REST API
├── models.py                # Pydantic models for configs and status
├── engine/
│   ├── runner.py            # Config generation + directory setup
│   ├── pipeline.py          # Sequential experiment orchestration
│   ├── tmux.py              # Tmux capture/list/kill operations
│   ├── terminal.py          # Terminal session management (WebSocket)
│   └── optimizer.py         # Optimal agent count via Marcus MCP
├── tracking/
│   └── mlflow_tracker.py    # Parent/child MLflow run tracking
├── static/
│   └── index.html           # Frontend SPA (vanilla JS + xterm.js)
└── tests/                   # Unit + integration tests
```

---

## Configuration

Posidonius generates experiment configs identical to what Marcus expects:

```yaml
project_name: "Snake-Game-Demo-run_0-5_agents"
project_spec_file: "project_spec.md"
project_options:
  complexity: "prototype"
  provider: "planka"
  mode: "new_project"
  project_root: "/path/to/implementation"
agents:
  - id: "agent_unicorn_1"
    name: "Unicorn Developer 1"
    role: "full-stack"
    skills: [python, javascript]
    subagents: 0
timeouts:
  project_creation: 300
  agent_startup: 60
```

---

## Ecosystem

Posidonius is part of a three-project platform for multi-agent coordination:

```
┌──────────────────────────────────────────────────────────┐
│                     The Platform                         │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Marcus     │  │ Posidonius  │  │    Cato     │      │
│  │ Coordinator  │  │  Experiment │  │ Observability│     │
│  │             │  │  Dashboard  │  │  Dashboard  │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │              │
│         └────────────────┴────────────────┘              │
│                     The Board                            │
└──────────────────────────────────────────────────────────┘
```

| Project | Role | What It Does |
|---------|------|--------------|
| [**Marcus**](https://github.com/lwgray/marcus) | The Stoic Coordinator | Breaks work into tasks, assigns to agents via shared board state, enforces dependencies |
| **Posidonius** | The Empiricist | Runs scaling experiments, tracks metrics, answers "does adding agents actually help?" |
| [**Cato**](https://github.com/lwgray/cato) | The Transparent Lens | Real-time observability dashboard with playable audit trails |

> *Marcus coordinates. Posidonius measures. Cato observes.*

---

## Testing

```bash
pytest tests/ -v           # All tests
pytest tests/ -m unit       # Fast, isolated tests
pytest tests/ -m asyncio    # Async tests
```

### Dev Tools

```bash
black posidonius/           # Format code
isort posidonius/           # Sort imports
mypy posidonius/            # Type checking
```

---

## Milestones

| Version | Highlights |
|---------|------------|
| **v0.1.1** | Version bump, stabilization |
| **v0.1.0** | UX overhaul, board panel, interactive terminals (xterm.js), auto-confirm for Claude trust prompts in tmux, agent status detection, task progress strip |
| **v0.0.1** | Initial commit — FastAPI dashboard, sequential pipelines, MLflow tracking, tmux agent monitoring, export, optimizer |

---

## License

MIT License -- see [LICENSE](LICENSE) for details.
