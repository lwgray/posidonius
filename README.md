# Posidonius

**Experiment dashboard for multi-agent coordination systems.**

Posidonius is a web UI for running, monitoring, and comparing multi-agent experiments. It launches agents via tmux, tracks metrics with MLflow, and provides a clean interface for scaling tests (e.g., 2 agents vs 5 vs 10).

Named after the Stoic philosopher known for empirical observation and measurement — fitting for a tool that measures whether multi-agent coordination actually works at scale.

## Architecture

Posidonius is a standalone project that communicates with [Marcus](https://github.com/lwgray/marcus) across process boundaries:

```
Posidonius → (HTTP)  → Marcus MCP Server (localhost:4298/mcp)
Posidonius → (shell) → run_experiment.py <dir>
Posidonius → (file)  → config.yaml + project_spec.md
```

No shared memory. No direct Python imports from Marcus. If Marcus runs on a remote server and Posidonius runs on your laptop, it still works.

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
│   ├── terminal.py          # Terminal session management
│   └── optimizer.py         # Optimal agent count via Marcus MCP (httpx)
├── tracking/
│   └── mlflow_tracker.py    # Parent/child MLflow run tracking
├── static/
│   └── index.html           # Frontend SPA
└── tests/                   # Unit tests
```

### How It Works

```
Browser (index.html)
  │
  ├─ POST /api/experiments        → Creates ExperimentPipeline in memory
  ├─ POST /api/experiments/{name}/start
  │     │
  │     ├─ ExperimentRunner.prepare_run()
  │     │     └─ Creates run directory with config.yaml + project_spec.md
  │     │
  │     ├─ MLflowTracker.start_child_run()
  │     │     └─ Logs params (agent count, complexity) to MLflow
  │     │
  │     └─ subprocess.Popen("python run_experiment.py <dir>")
  │           └─ Marcus CLI launcher takes over:
  │                 1. Creates tmux session
  │                 2. Spawns project creator (claude --print)
  │                 3. Waits for project_info.json
  │                 4. Spawns workers (claude --dangerously-skip-permissions)
  │                 5. Spawns monitor
  │
  ├─ GET /api/experiments/{name}/output  (polled every 15s)
  │     └─ TmuxManager.capture_all_panes()
  │           └─ `tmux capture-pane -t <target> -p -S -50` per pane
  │           └─ detect_agent_status() → working/waiting/complete/error/idle
  │
  ├─ DELETE /api/experiments/{name}     (stop button)
  │     └─ TmuxManager.kill_session() + MLflow end_run(KILLED)
  │
  └─ GET /api/experiments/history
        └─ mlflow.search_experiments() + search_runs()
```

### Key Boundaries

| Boundary | Mechanism |
|----------|-----------|
| Posidonius → Marcus agents | subprocess → run_experiment.py |
| Posidonius → Marcus MCP | httpx → localhost:4298/mcp (optimizer only) |
| Posidonius → tmux | subprocess → tmux capture-pane / send-keys |
| Posidonius → MLflow | mlflow Python SDK → local mlruns/ directory |

## v1 MVP

Single-page dashboard with:

- **One form, one button** — project name, prompt, complexity, agent count, subagents, then "Run Experiment"
- **Subprocess execution** — shells out to `run_experiment.py` (proven, working CLI launcher)
- **Read-only status** — polls `tmux capture-pane` for agent output (no WebSocket, no interactive terminals)
- **Agent status indicators** — waiting / working / complete / error
- **MLflow tracking** — parent experiment + child runs with metrics
- **History tab** — past experiments from MLflow with metrics comparison
- **Export** — zip download of all terminal output
- **Stop** — kills the tmux session

### What v1 Does NOT Include

- Interactive browser terminals (xterm.js) — deferred to v2
- Sequential run auto-advancement — deferred to v2
- Real-time log streaming via WebSocket — deferred to v2
- Optimal agent pre-flight via MCP — deferred to v2

## Prerequisites

- Python 3.11+
- tmux
- MLflow
- Marcus MCP server running at `localhost:4298/mcp` (for experiments, not for Posidonius itself)
- `run_experiment.py` available from Marcus (`dev-tools/experiments/runners/`)

## Installation

```bash
cd ~/dev/posidonius
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Start the dashboard (default: http://localhost:8420)
python -m posidonius

# Or with custom options
python -m posidonius --host 0.0.0.0 --port 9000 --experiments-dir ~/experiments --templates-dir ~/marcus/templates
```

## Configuration

Posidonius generates configs identical to what `run_experiment.py` expects:

```yaml
project_name: "Snake-Game-Demo"
project_spec_file: "project_spec.md"
project_options:
  complexity: "prototype"
  provider: "planka"
  mode: "new_project"
agents:
  - id: "agent_unicorn_1"
    name: "Unicorn Developer 1"
    role: "full-stack"
    skills: ["python", "javascript"]
    subagents: 0
timeouts:
  project_creation: 300
  agent_startup: 60
```

## Testing

```bash
pytest tests/ -v
```

## Ecosystem

| Project | Purpose |
|---------|---------|
| **Marcus** | Multi-agent coordination platform (MCP server, task management) |
| **Posidonius** | Experiment runner and measurement dashboard |
| **Cato** | Observability and audit trail viewer |

## License

MIT
