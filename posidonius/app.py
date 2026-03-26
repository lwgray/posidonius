"""FastAPI application for the Posidonius experiment dashboard.

Provides REST API endpoints for creating, starting, monitoring, and stopping
experiment pipelines. Includes live tmux output capture and optimal agent
pre-flight estimation via Marcus MCP.
"""

import asyncio
import io
import zipfile
from pathlib import Path
from typing import Any, Optional

import mlflow
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from mlflow.exceptions import MlflowException
from posidonius.engine.optimizer import OptimalAgentOptimizer
from posidonius.engine.pipeline import ExperimentPipeline
from posidonius.engine.terminal import TmuxTerminalSession
from posidonius.models import (
    ExperimentStatus,
    OptimalAgentRequest,
    PipelineConfig,
)


def create_app(
    templates_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    templates_dir : Path | None
        Path to experiment templates. Defaults to Marcus templates dir.
    experiments_dir : Path | None
        Base directory for experiment output. Defaults to ~/experiments.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    app = FastAPI(
        title="Posidonius Experiment Dashboard",
        description=(
            "Web UI for running multi-agent experiments"
        ),
        version="0.1.0",
    )

    if templates_dir is None:
        templates_dir = (
            Path.home()
            / "dev"
            / "marcus"
            / "dev-tools"
            / "experiments"
            / "templates"
        )
    if experiments_dir is None:
        experiments_dir = Path.home() / "experiments"

    static_dir = Path(__file__).parent / "static"

    # Set MLflow tracking URI to SQLite database
    mlflow_db = experiments_dir / "mlflow.db"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")

    # Per-app instance store of active pipelines
    pipelines: dict[str, ExperimentPipeline] = {}
    app.state.pipelines = pipelines

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        """List all experiment pipelines."""
        return [
            p.get_status().model_dump() for p in pipelines.values()
        ]

    @app.get("/api/experiments/history")
    def get_experiment_history() -> list[dict[str, Any]]:
        """Get past experiment history from MLflow.

        Returns
        -------
        list[dict[str, Any]]
            List of past experiments with their runs and metrics.
        """
        try:
            experiments = mlflow.search_experiments()
        except MlflowException:
            return []
        history: list[dict[str, Any]] = []

        for exp in experiments:
            if exp.name == "Default":
                continue
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                output_format="list",
            )
            run_data: list[dict[str, Any]] = []
            for run in runs:
                run_data.append(
                    {
                        "run_id": run.info.run_id,
                        "run_name": run.info.run_name,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "params": dict(run.data.params),
                        "metrics": dict(run.data.metrics),
                    }
                )
            history.append(
                {
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.name,
                    "runs": run_data,
                }
            )

        return history

    @app.post("/api/experiments", status_code=201)
    def create_experiment(
        config: PipelineConfig,
    ) -> dict[str, Any]:
        """Create a new experiment pipeline (does not start it).

        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration.

        Returns
        -------
        dict[str, Any]
            Pipeline status after creation.
        """
        if config.name in pipelines:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Pipeline '{config.name}' already exists"
                ),
            )

        pipeline = ExperimentPipeline(
            config=config,
            templates_dir=templates_dir,
            base_dir=experiments_dir / config.name,
        )
        pipelines[config.name] = pipeline
        return pipeline.get_status().model_dump()

    @app.post("/api/experiments/{name}/start")
    def start_experiment(
        name: str, run_index: int = 0
    ) -> dict[str, Any]:
        """Start an experiment run (launches agents via subprocess).

        Parameters
        ----------
        name : str
            Pipeline name.
        run_index : int
            Which run in the pipeline to start (default 0).

        Returns
        -------
        dict[str, Any]
            Pipeline status with tmux session info.
        """
        if name not in pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{name}' not found",
            )
        pipeline = pipelines[name]
        if pipeline.status == ExperimentStatus.RUNNING:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Pipeline is already running. Stop it first."
                ),
            )
        if run_index >= len(pipeline.config.runs):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Run index {run_index} out of range "
                    f"(pipeline has {len(pipeline.config.runs)} runs)"
                ),
            )
        try:
            tmux_session = pipeline.start_run(run_index)
            return {
                **pipeline.get_status().model_dump(),
                "tmux_session": tmux_session,
                "message": (
                    f"Started run {run_index} with "
                    f"{pipeline.config.runs[run_index].num_agents}"
                    " agents"
                ),
            }
        except Exception as e:
            pipeline.status = ExperimentStatus.FAILED
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start experiment: {e}",
            )

    @app.post("/api/experiments/{name}/complete-run")
    def complete_run(
        name: str, run_index: int
    ) -> dict[str, Any]:
        """Mark the current run as complete and tear it down.

        Parameters
        ----------
        name : str
            Pipeline name.
        run_index : int
            Which run to complete.

        Returns
        -------
        dict[str, Any]
            Updated pipeline status.
        """
        if name not in pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{name}' not found",
            )
        pipeline = pipelines[name]
        if run_index not in pipeline.run_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_index} not found",
            )
        tmux_session = pipeline.run_statuses[run_index].get(
            "tmux_session"
        )
        if tmux_session:
            pipeline.teardown_run(run_index, tmux_session)

        # If all runs are done, complete the pipeline
        if run_index + 1 >= len(pipeline.config.runs):
            pipeline.complete_pipeline()
        else:
            # Ready for next run
            pipeline.status = ExperimentStatus.PENDING

        return pipeline.get_status().model_dump()

    @app.get("/api/experiments/{name}")
    def get_experiment_status(name: str) -> dict[str, Any]:
        """Get status of an experiment pipeline.

        Parameters
        ----------
        name : str
            Pipeline name.

        Returns
        -------
        dict[str, Any]
            Current pipeline status.
        """
        if name not in pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{name}' not found",
            )
        return pipelines[name].get_status().model_dump()

    @app.get("/api/experiments/{name}/output")
    def get_experiment_output(
        name: str,
    ) -> list[dict[str, Any]]:
        """Get live output from all tmux panes of the active run.

        Parameters
        ----------
        name : str
            Pipeline name.

        Returns
        -------
        list[dict[str, Any]]
            List of pane outputs with title, output text, and
            agent status.
        """
        if name not in pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{name}' not found",
            )
        return pipelines[name].get_run_output()

    @app.delete("/api/experiments/{name}")
    def stop_experiment(name: str) -> dict[str, Any]:
        """Stop an experiment pipeline and kill tmux sessions.

        Parameters
        ----------
        name : str
            Pipeline name.

        Returns
        -------
        dict[str, Any]
            Updated pipeline status after stopping.
        """
        if name not in pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{name}' not found",
            )
        pipeline = pipelines[name]
        pipeline.stop()
        return pipeline.get_status().model_dump()

    @app.post("/api/experiments/optimize")
    def optimize_agents(
        request: OptimalAgentRequest,
    ) -> dict[str, Any]:
        """Run optimal agent pre-flight via Marcus MCP CPM analysis.

        Connects to Marcus, creates the project, and runs real
        dependency graph analysis. Requires Marcus server to be running.

        Parameters
        ----------
        request : OptimalAgentRequest
            Project spec and complexity for estimation.

        Returns
        -------
        dict[str, Any]
            Optimal agent recommendation with suggested runs.
        """
        optimizer = OptimalAgentOptimizer()
        try:
            response = optimizer.analyze_sync(
                project_name="optimization-preflight",
                project_spec=request.project_spec,
                complexity=request.complexity,
            )
            return response.model_dump()
        except (ConnectionError, OSError) as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Marcus server unavailable. Start it first: "
                    "python -m src.marcus_mcp.server --http. "
                    f"Error: {e}"
                ),
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {e}",
            )

    # Track active terminal sessions for cleanup
    terminal_sessions: dict[str, TmuxTerminalSession] = {}

    @app.websocket(
        "/api/experiments/{name}/terminal/{pane_index}"
    )
    async def terminal_websocket(
        websocket: WebSocket, name: str, pane_index: int
    ) -> None:
        """Interactive terminal via WebSocket.

        Attaches to a tmux pane and bridges I/O to the browser
        via xterm.js.

        Parameters
        ----------
        websocket : WebSocket
            WebSocket connection from xterm.js.
        name : str
            Pipeline name.
        pane_index : int
            Index of the pane to attach to.
        """
        await websocket.accept()

        if name not in pipelines:
            await websocket.close(
                code=4004, reason="Pipeline not found"
            )
            return

        pipeline = pipelines[name]
        if not pipeline.active_tmux_session:
            await websocket.close(
                code=4004, reason="No active session"
            )
            return

        # Build tmux target for the specific pane
        panes = pipeline.tmux.list_panes(
            pipeline.active_tmux_session
        )
        if pane_index >= len(panes):
            await websocket.close(
                code=4004, reason="Pane not found"
            )
            return

        pane_target = panes[pane_index]["target"]
        session_key = f"{name}:{pane_index}"

        # Per-pane terminal using capture-pane + send-keys
        term = TmuxTerminalSession(pane_target=pane_target)
        terminal_sessions[session_key] = term

        try:
            term.start(rows=24, cols=80)
            last_content = b""

            # Read loop: poll capture-pane, send only on change
            async def read_loop() -> None:
                nonlocal last_content
                while term.is_alive:
                    data = await term.read_async()
                    if data and data != last_content:
                        last_content = data
                        await websocket.send_bytes(
                            b"\x1b[2J\x1b[H" + data
                        )
                    await asyncio.sleep(0.5)

            read_task = asyncio.create_task(read_loop())

            # Write loop: send browser input to pane
            key_map = {
                "\r": "Enter",
                "\x1b": "Escape",
                "\x1b[A": "Up",
                "\x1b[B": "Down",
                "\x1b[C": "Right",
                "\x1b[D": "Left",
                "\x7f": "BSpace",
                "\x03": "C-c",
                "\x04": "C-d",
            }

            try:
                while True:
                    message = await websocket.receive()
                    if "bytes" in message:
                        raw = message["bytes"]
                        text = raw.decode(
                            "utf-8", errors="replace"
                        )
                    elif "text" in message:
                        text = message["text"]
                        if text.startswith("resize:"):
                            parts = text.split(":")
                            if len(parts) == 3:
                                term.resize(
                                    int(parts[1]),
                                    int(parts[2]),
                                )
                            continue
                    else:
                        continue

                    # Check for special key sequences
                    if text in key_map:
                        term.send_key(key_map[text])
                    else:
                        term.write(text.encode("utf-8"))
            except WebSocketDisconnect:
                pass
            finally:
                read_task.cancel()
        finally:
            term.stop()
            terminal_sessions.pop(session_key, None)

    @app.get("/api/experiments/{name}/export")
    def export_terminal_output(
        name: str,
    ) -> StreamingResponse:
        """Export all agent terminal output as a zip file.

        Parameters
        ----------
        name : str
            Pipeline name.

        Returns
        -------
        StreamingResponse
            Zip file containing one text file per agent pane.
        """
        if name not in pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{name}' not found",
            )

        pipeline = pipelines[name]
        if not pipeline.active_tmux_session:
            raise HTTPException(
                status_code=400,
                detail="No active tmux session to export",
            )

        panes = pipeline.tmux.list_panes(
            pipeline.active_tmux_session
        )
        if not panes:
            raise HTTPException(
                status_code=400,
                detail="No panes found in tmux session",
            )

        # Capture full scrollback (large buffer)
        buf = io.BytesIO()
        with zipfile.ZipFile(
            buf, "w", zipfile.ZIP_DEFLATED
        ) as zf:
            for i, pane in enumerate(panes):
                output = pipeline.tmux.capture_pane(
                    pane["target"], lines=10000
                )
                title = (
                    pane["title"]
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                filename = f"{i:02d}_{title}.txt"
                zf.writestr(filename, output)

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{name}_output.zip"'
                )
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the main dashboard page."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return index_file.read_text()
        return _minimal_html()

    return app


def _minimal_html() -> str:
    """Return minimal HTML when static files aren't built yet."""
    return """<!DOCTYPE html>
<html>
<head><title>Posidonius Experiment Dashboard</title></head>
<body>
<h1>Posidonius Experiment Dashboard</h1>
<p>Static files not found. Build the frontend first.</p>
</body>
</html>"""
