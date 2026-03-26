"""Sequential experiment pipeline engine.

Orchestrates multiple experiment runs in sequence (e.g., 5 -> 10 -> 20 agents),
handling teardown between runs, MLflow tracking, and status management.
Launches experiments via subprocess to run_experiment.py, communicating
across process boundaries rather than importing Marcus internals.
"""

import json
import subprocess  # nosec B404
import threading
import time
from pathlib import Path
from typing import Any, Optional

from posidonius.engine.runner import ExperimentRunner
from posidonius.engine.tmux import TmuxManager
from posidonius.models import (
    ExperimentStatus,
    PipelineConfig,
    PipelineStatus,
    RunStatus,
)
from posidonius.tracking.mlflow_tracker import MLflowTracker


class ExperimentPipeline:
    """Orchestrates sequential experiment runs with MLflow tracking.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with sequential runs.
    templates_dir : Path
        Path to experiment templates directory.
    base_dir : Path
        Base directory for experiment output.
    run_experiment_script : Path | None
        Path to Marcus's run_experiment.py script.
        Defaults to ~/dev/marcus/dev-tools/experiments/runners/run_experiment.py.
    """

    def __init__(
        self,
        config: PipelineConfig,
        templates_dir: Path,
        base_dir: Path,
        run_experiment_script: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.runner = ExperimentRunner(
            pipeline=config,
            templates_dir=templates_dir,
            base_dir=base_dir,
        )
        self.tmux = TmuxManager()
        self.current_run_index: int = -1
        self.status: ExperimentStatus = ExperimentStatus.PENDING
        self.run_statuses: dict[int, dict[str, Any]] = {}
        self.active_tmux_session: Optional[str] = None
        self._run_dirs: dict[int, Path] = {}
        self._run_start_times: dict[int, float] = {}
        self.tracker: Optional[MLflowTracker] = None
        self._run_experiment_script = run_experiment_script or (
            Path.home()
            / "dev"
            / "marcus"
            / "dev-tools"
            / "experiments"
            / "runners"
            / "run_experiment.py"
        )

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status.

        Returns
        -------
        PipelineStatus
            Current status of the pipeline and all runs.
        """
        runs: list[RunStatus] = []
        for idx, run_info in self.run_statuses.items():
            runs.append(
                RunStatus(
                    run_index=idx,
                    num_agents=run_info["num_agents"],
                    status=run_info["status"],
                    tmux_session=run_info.get("tmux_session"),
                    tasks_completed=run_info.get("tasks_completed"),
                    tasks_total=run_info.get("tasks_total"),
                    mlflow_run_id=run_info.get("mlflow_run_id"),
                    error=run_info.get("error"),
                )
            )

        mlflow_experiment_id = None
        if self.tracker is not None:
            mlflow_experiment_id = self.tracker.experiment_id

        return PipelineStatus(
            pipeline_name=self.config.name,
            total_runs=len(self.config.runs),
            current_run=self.current_run_index,
            status=self.status,
            runs=runs,
            mlflow_experiment_id=mlflow_experiment_id,
        )

    def is_run_complete(self, run_dir: Path) -> bool:
        """Check if an experiment run has completed.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        bool
            True if the run has completed.
        """
        completion_file = run_dir / "experiment_complete.json"
        return completion_file.exists()

    def start_run(self, run_index: int) -> str:
        """Start an experiment run via subprocess.

        Shells out to run_experiment.py with the prepared run directory,
        matching the same behavior as the CLI launcher.

        Parameters
        ----------
        run_index : int
            Index of the run in the pipeline.

        Returns
        -------
        str
            Tmux session name for the launched run.
        """
        # Initialize MLflow tracker on first run
        if self.tracker is None:
            self.tracker = MLflowTracker(self.config)
            self.tracker.start_pipeline_run()

        # Prepare directory with config.yaml and project_spec.md
        run_dir = self.runner.prepare_run(run_index)
        self._run_dirs[run_index] = run_dir

        tmux_session = self.runner.get_tmux_session_name(run_index)

        self.current_run_index = run_index
        self.active_tmux_session = tmux_session
        self.status = ExperimentStatus.RUNNING

        run_config = self.config.runs[run_index]

        # Start MLflow child run
        child_run_id = self.tracker.start_child_run(
            run_index=run_index,
            num_agents=run_config.num_agents,
            subagents_per_agent=run_config.subagents_per_agent,
        )

        self._run_start_times[run_index] = time.time()

        self.run_statuses[run_index] = {
            "status": ExperimentStatus.RUNNING,
            "num_agents": run_config.num_agents,
            "tmux_session": tmux_session,
            "mlflow_run_id": child_run_id,
        }

        # Launch via subprocess to run_experiment.py
        def _launch_experiment() -> None:
            try:
                subprocess.Popen(  # nosec B603
                    [
                        "python",
                        str(self._run_experiment_script),
                        str(run_dir),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as e:
                self.run_statuses[run_index][
                    "status"
                ] = ExperimentStatus.FAILED
                self.run_statuses[run_index]["error"] = str(e)
                self.status = ExperimentStatus.FAILED

        thread = threading.Thread(
            target=_launch_experiment, daemon=True
        )
        thread.start()

        return tmux_session

    def get_run_output(self) -> list[dict[str, Any]]:
        """Get live output from all panes of the active tmux session.

        Returns
        -------
        list[dict[str, Any]]
            List of pane outputs with title, output text, and status.
        """
        if not self.active_tmux_session:
            return []
        if not self.tmux.session_exists(self.active_tmux_session):
            return []
        return self.tmux.capture_all_panes(self.active_tmux_session)

    def teardown_run(
        self, run_index: int, tmux_session: str
    ) -> None:
        """Clean up after a completed run, log metrics to MLflow.

        Parameters
        ----------
        run_index : int
            Index of the run to tear down.
        tmux_session : str
            Tmux session name to kill.
        """
        if self.tracker is not None:
            elapsed = time.time() - self._run_start_times.get(
                run_index, time.time()
            )
            run_info = self.run_statuses.get(run_index, {})
            self.tracker.log_run_metrics(
                completion_time_seconds=elapsed,
                tasks_completed=run_info.get("tasks_completed", 0),
                tasks_total=run_info.get("tasks_total", 0),
                blockers=run_info.get("blockers", 0),
            )
            self.tracker.end_child_run(status="FINISHED")

        self.tmux.kill_session(tmux_session)
        self.active_tmux_session = None
        if run_index in self.run_statuses:
            self.run_statuses[run_index][
                "status"
            ] = ExperimentStatus.COMPLETED

    def stop(self) -> None:
        """Stop the pipeline, kill tmux, and end MLflow runs."""
        if self.active_tmux_session:
            self.tmux.kill_session(self.active_tmux_session)
            self.active_tmux_session = None

        if self.tracker is not None:
            self.tracker.end_child_run(status="KILLED")
            self.tracker.end_pipeline_run(status="KILLED")

        self.status = ExperimentStatus.STOPPED

    def complete_pipeline(self) -> None:
        """Mark pipeline as complete and end the parent MLflow run."""
        if self.tracker is not None:
            self.tracker.end_pipeline_run(status="FINISHED")
        self.status = ExperimentStatus.COMPLETED

    def prepare_next_run(self) -> Optional[Path]:
        """Prepare the next run in the pipeline (without launching).

        Returns
        -------
        Path | None
            Path to the prepared run directory, or None if all runs
            are exhausted.
        """
        next_index = self.current_run_index + 1
        if next_index >= len(self.config.runs):
            return None

        self.current_run_index = next_index
        run_dir = self.runner.prepare_run(next_index)
        self._run_dirs[next_index] = run_dir

        run_config = self.config.runs[next_index]
        tmux_session = self.runner.get_tmux_session_name(next_index)
        self.active_tmux_session = tmux_session

        self.run_statuses[next_index] = {
            "status": ExperimentStatus.PENDING,
            "num_agents": run_config.num_agents,
            "tmux_session": tmux_session,
        }

        return run_dir
