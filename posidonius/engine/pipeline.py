"""Sequential experiment pipeline engine.

Orchestrates multiple experiment runs in sequence (e.g., 5 -> 10 -> 20 agents),
handling teardown between runs, MLflow tracking, and status management.
Launches experiments via subprocess to run_experiment.py, communicating
across process boundaries rather than importing Marcus internals.
"""

import os
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from posidonius.engine.event_log import PipelineEventLog
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
        marcus_instance: Optional[dict[str, Any]] = None,
        marcus_python: Optional[str] = None,
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
        self._auto_advance_active: bool = False
        self._auto_advance_paused: bool = False
        self.events = PipelineEventLog(base_dir, config.name)
        self._run_experiment_script = run_experiment_script or (
            Path.home()
            / "dev"
            / "marcus"
            / "dev-tools"
            / "experiments"
            / "runners"
            / "run_experiment.py"
        )
        # Optional: per-instance Marcus URL and DB path for parallel experiments
        self._marcus_instance: Optional[dict[str, Any]] = marcus_instance
        # Python interpreter to use when spawning run_experiment.py.
        # Must match the env where Marcus is installed.
        self._python_executable: str = marcus_python or sys.executable

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
                    run_dir=str(self._run_dirs.get(idx, "")) or None,
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

        self.events.log(
            "RUN_STARTING",
            f"Starting run {run_index} with "
            f"{self.config.runs[run_index].num_agents} agents",
            run_index=run_index,
            tmux_session=tmux_session,
            run_dir=str(run_dir),
        )

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

        # Build env for subprocess: inherit current env, then inject
        # per-instance Marcus URL and DB path when running in parallel mode.
        run_env: Optional[dict[str, str]] = None
        if self._marcus_instance:
            run_env = dict(os.environ)
            run_env["MARCUS_URL"] = self._marcus_instance["url"]
            if "db_path" in self._marcus_instance:
                run_env["SQLITE_KANBAN_DB_PATH"] = self._marcus_instance["db_path"]

        # Redirect run_experiment.py output to a per-run log so any
        # crash in spawn_agents.py (Phase 2 worker spawn, git worktree
        # creation, etc.) is visible rather than silently swallowed.
        spawn_log_path = run_dir / "logs" / "spawn.log"
        spawn_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Launch via subprocess to run_experiment.py, then auto-confirm
        # any trust prompts that appear in agent panes.
        def _launch_and_confirm_trust() -> None:
            try:
                spawn_log = open(spawn_log_path, "w")  # noqa: SIM115
                subprocess.Popen(  # nosec B603
                    [
                        self._python_executable,
                        str(self._run_experiment_script),
                        str(run_dir),
                    ],
                    stdout=spawn_log,
                    stderr=spawn_log,
                    env=run_env,
                )
            except OSError as e:
                self.events.log(
                    "RUN_FAILED",
                    f"Failed to launch subprocess: {e}",
                    run_index=run_index,
                )
                self.run_statuses[run_index]["status"] = ExperimentStatus.FAILED
                self.run_statuses[run_index]["error"] = str(e)
                self.status = ExperimentStatus.FAILED
                return

            # Wait for the tmux session to come up, then poll for
            # trust prompts repeatedly. Agents spawn at different times
            # so trust prompts can appear after the session is already up.
            session_found = False
            for attempt in range(30):
                time.sleep(2)
                if self.tmux.session_exists(tmux_session):
                    session_found = True
                    self.events.log(
                        "TMUX_SESSION_FOUND",
                        f"Session {tmux_session} detected after "
                        f"{(attempt + 1) * 2}s",
                        run_index=run_index,
                    )
                    confirmed = self.tmux.auto_confirm_trust(tmux_session)
                    if confirmed:
                        self.events.log(
                            "TRUST_CONFIRMED",
                            f"Confirmed {confirmed} trust prompt(s)",
                            run_index=run_index,
                            count=confirmed,
                        )
                    break

            if not session_found:
                self.events.log(
                    "TMUX_SESSION_TIMEOUT",
                    f"Session {tmux_session} not found after 60s",
                    run_index=run_index,
                )

            # Keep polling for late trust prompts for 60 more seconds
            if session_found:
                for poll in range(12):
                    time.sleep(5)
                    self.tmux.auto_confirm_trust(tmux_session)

        thread = threading.Thread(target=_launch_and_confirm_trust, daemon=True)
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

    def _run_epictetus(self, run_index: int, tmux_session: Optional[str]) -> None:
        """Run Epictetus audit in a visible tmux pane while the session is alive.

        Spawns a dedicated 'Epictetus' pane in the existing tmux session so
        progress is visible (attach with ``tmux attach -t <session>`` or via
        the Posidonius terminal viewer). Output is also written to
        ``{run_dir}/epictetus.log``.

        Must be called BEFORE teardown_run() — Epictetus uses --session to
        interrogate live agents and read terminal output. Blocks until the
        audit pane exits so quality data is written before teardown kills
        the session.

        Parameters
        ----------
        run_index : int
            Index of the run to audit.
        tmux_session : str | None
            Active tmux session name. Required for pane creation; falls back
            to a plain subprocess if None.
        """
        run_dir = self._run_dirs.get(run_index)
        if run_dir is None:
            self.events.log(
                "EPICTETUS_SKIPPED",
                f"Run {run_index}: no run_dir tracked",
                run_index=run_index,
            )
            return

        impl_dir = run_dir / "implementation"
        if not impl_dir.exists():
            self.events.log(
                "EPICTETUS_SKIPPED",
                f"Run {run_index}: implementation/ not found at {impl_dir}",
                run_index=run_index,
            )
            return

        skill_input = f"/epictetus {impl_dir}"
        if tmux_session:
            skill_input += f" --session {tmux_session}"

        log_file = run_dir / "epictetus.log"

        self.events.log(
            "EPICTETUS_STARTED",
            f"Run {run_index}: auditing {impl_dir}",
            run_index=run_index,
            session=tmux_session,
            log=str(log_file),
        )

        if tmux_session and self.tmux.session_exists(tmux_session):
            self._run_epictetus_in_tmux(
                run_index, tmux_session, impl_dir, skill_input, log_file
            )
        else:
            # Fallback: plain subprocess when no tmux session available
            self._run_epictetus_subprocess(run_index, impl_dir, skill_input, log_file)

    def _run_epictetus_in_tmux(
        self,
        run_index: int,
        tmux_session: str,
        impl_dir: Path,
        skill_input: str,
        log_file: Path,
    ) -> None:
        """Run Epictetus in a new tmux window within the existing session.

        Parameters
        ----------
        run_index : int
            Index of the run being audited.
        tmux_session : str
            Existing tmux session name.
        impl_dir : Path
            Implementation directory to audit.
        skill_input : str
            Skill invocation string piped to claude.
        log_file : Path
            Path for Epictetus log output.
        """
        # Write skill input to a temp file so the shell can read it cleanly
        prompt_file = impl_dir.parent / "prompts" / "epictetus_input.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(skill_input)

        script = (
            "#!/bin/bash\n"
            "[ -f ~/.zshrc ] && source ~/.zshrc\n"
            "[ -f ~/.bashrc ] && source ~/.bashrc\n"
            "unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_SESSION\n"
            f'cd "{impl_dir}" || exit 1\n'
            'echo "=========================================="\n'
            'echo "EPICTETUS AUDIT"\n'
            f'echo "Auditing: {impl_dir}"\n'
            'echo "=========================================="\n'
            f'claude --dangerously-skip-permissions --print < "{prompt_file}" '
            f'| tee "{log_file}"\n'
            'echo "=========================================="\n'
            'echo "Epictetus Complete"\n'
            'echo "=========================================="\n'
        )
        script_file = impl_dir.parent / "prompts" / "epictetus.sh"
        script_file.write_text(script)
        script_file.chmod(0o755)

        # Spawn a new window in the session
        result = subprocess.run(  # nosec B603
            [
                "tmux",
                "new-window",
                "-t",
                tmux_session,
                "-n",
                "epictetus",
                "-P",
                "-F",
                "#{pane_id}",
            ],
            capture_output=True,
            text=True,
        )
        pane_id = result.stdout.strip()

        if not pane_id:
            self.events.log(
                "EPICTETUS_FAILED",
                f"Run {run_index}: could not create tmux window",
                run_index=run_index,
            )
            self._run_epictetus_subprocess(run_index, impl_dir, skill_input, log_file)
            return

        subprocess.run(  # nosec B603
            ["tmux", "select-pane", "-t", pane_id, "-T", "Epictetus"],
            capture_output=True,
        )
        subprocess.run(  # nosec B603
            ["tmux", "send-keys", "-t", pane_id, f"bash {script_file}", "Enter"],
            capture_output=True,
        )

        # Poll until pane exits or timeout (30 min)
        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(10)
            dead = subprocess.run(  # nosec B603
                ["tmux", "display-message", "-p", "-t", pane_id, "#{pane_dead}"],
                capture_output=True,
                text=True,
            )
            if dead.stdout.strip() == "1":
                break
        else:
            self.events.log(
                "EPICTETUS_FAILED",
                f"Run {run_index}: audit timed out after 1800s in tmux pane {pane_id}",
                run_index=run_index,
            )
            return

        # Check log for success signal
        if log_file.exists() and "Epictetus Complete" in log_file.read_text():
            self.events.log(
                "EPICTETUS_COMPLETE",
                f"Run {run_index}: audit done — see {log_file}",
                run_index=run_index,
            )
        else:
            self.events.log(
                "EPICTETUS_FAILED",
                f"Run {run_index}: audit exited without completion signal",
                run_index=run_index,
            )

    def _run_epictetus_subprocess(
        self,
        run_index: int,
        impl_dir: Path,
        skill_input: str,
        log_file: Path,
    ) -> None:
        """Fallback: run Epictetus as a plain subprocess, output to log_file.

        Parameters
        ----------
        run_index : int
            Index of the run being audited.
        impl_dir : Path
            Implementation directory to audit.
        skill_input : str
            Skill invocation string piped to claude.
        log_file : Path
            Path to write stdout/stderr.
        """
        try:
            with open(log_file, "wb") as lf:
                proc = subprocess.Popen(  # nosec B603
                    ["claude", "--dangerously-skip-permissions", "--print"],
                    stdin=subprocess.PIPE,
                    stdout=lf,
                    stderr=lf,
                    cwd=str(impl_dir),
                )
                try:
                    proc.communicate(input=skill_input.encode(), timeout=1800)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    self.events.log(
                        "EPICTETUS_FAILED",
                        f"Run {run_index}: subprocess audit timed out — partial log: {log_file}",
                        run_index=run_index,
                    )
                    return

            if proc.returncode == 0:
                self.events.log(
                    "EPICTETUS_COMPLETE",
                    f"Run {run_index}: audit done — see {log_file}",
                    run_index=run_index,
                )
            else:
                self.events.log(
                    "EPICTETUS_FAILED",
                    f"Run {run_index}: audit failed (rc={proc.returncode}) — see {log_file}",
                    run_index=run_index,
                )
        except Exception as exc:
            self.events.log(
                "EPICTETUS_FAILED",
                f"Run {run_index}: unexpected error — {exc}",
                run_index=run_index,
            )

    def teardown_run(self, run_index: int, tmux_session: str) -> None:
        """Clean up after a completed run, log metrics to MLflow.

        Parameters
        ----------
        run_index : int
            Index of the run to tear down.
        tmux_session : str
            Tmux session name to kill.
        """
        elapsed = time.time() - self._run_start_times.get(run_index, time.time())

        self.events.log(
            "RUN_TEARDOWN",
            f"Tearing down run {run_index} after {elapsed:.0f}s",
            run_index=run_index,
            tmux_session=tmux_session,
            elapsed_seconds=round(elapsed, 1),
        )

        if self.tracker is not None:
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
            self.run_statuses[run_index]["status"] = ExperimentStatus.COMPLETED

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
        self.events.log(
            "PIPELINE_FINISHING",
            f"Completing pipeline with " f"{len(self.run_statuses)} runs",
        )
        if self.tracker is not None:
            self.tracker.end_pipeline_run(status="FINISHED")
        self.status = ExperimentStatus.COMPLETED

    def auto_advance(self, poll_interval: int = 30) -> None:
        """Poll for completion and advance to next run automatically.

        Runs in a background thread. Checks for experiment_complete.json
        every ``poll_interval`` seconds. When found, tears down the current
        run (killing tmux and zombie agents), then starts the next run.
        After all runs complete, marks the pipeline finished.

        Parameters
        ----------
        poll_interval : int
            Seconds between completion checks. Default 30.
        """
        if self._auto_advance_active:
            return

        def _loop() -> None:
            self._auto_advance_active = True
            poll_count = 0
            self.events.log(
                "AUTO_ADVANCE_STARTED",
                f"Polling every {poll_interval}s for completion",
                poll_interval=poll_interval,
                total_runs=len(self.config.runs),
            )
            try:
                while self.status == ExperimentStatus.RUNNING:
                    run_index = self.current_run_index
                    run_dir = self._run_dirs.get(run_index)
                    poll_count += 1

                    if run_dir and self.is_run_complete(run_dir):
                        # If paused, wait for resume before teardown
                        if self._auto_advance_paused:
                            time.sleep(poll_interval)
                            continue

                        elapsed = time.time() - self._run_start_times.get(
                            run_index, time.time()
                        )
                        self.events.log(
                            "COMPLETION_DETECTED",
                            f"Run {run_index} complete after "
                            f"{elapsed:.0f}s ({poll_count} polls)",
                            run_index=run_index,
                            elapsed_seconds=round(elapsed, 1),
                            poll_count=poll_count,
                        )

                        # Audit while tmux is still alive, then tear down.
                        # Order is critical: Epictetus needs --session to
                        # interrogate agents; teardown kills the session.
                        tmux_session = self.run_statuses.get(run_index, {}).get(
                            "tmux_session"
                        )
                        self._run_epictetus(run_index, tmux_session)
                        if tmux_session:
                            self.teardown_run(run_index, tmux_session)

                        # Start next run or finish
                        next_index = run_index + 1
                        if next_index < len(self.config.runs):
                            self.events.log(
                                "ADVANCING",
                                f"Starting run {next_index}",
                                from_run=run_index,
                                to_run=next_index,
                            )
                            self.start_run(next_index)
                            poll_count = 0
                        else:
                            self.events.log(
                                "PIPELINE_COMPLETE",
                                f"All {len(self.config.runs)} runs finished",
                            )
                            self.complete_pipeline()
                            break
                    else:
                        if poll_count % 10 == 0:
                            # Log every 10th poll so we know it's alive
                            self.events.log(
                                "AUTO_ADVANCE_POLLING",
                                f"Run {run_index}: poll #{poll_count}, "
                                f"not complete yet",
                                run_index=run_index,
                                poll_count=poll_count,
                                run_dir=str(run_dir) if run_dir else None,
                            )

                    time.sleep(poll_interval)
            except Exception as e:
                self.events.log(
                    "AUTO_ADVANCE_ERROR",
                    f"Loop crashed: {e}",
                    error=str(e),
                )
            finally:
                self._auto_advance_active = False
                self.events.log(
                    "AUTO_ADVANCE_STOPPED",
                    f"Loop ended (status={self.status.value})",
                )

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()

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
