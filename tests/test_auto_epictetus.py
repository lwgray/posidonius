"""Unit tests for auto-Epictetus audit integration.

Verifies that _run_epictetus() fires BEFORE teardown_run() so the tmux
session is still alive when Epictetus needs to interrogate agents.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from posidonius.engine.pipeline import ExperimentPipeline
from posidonius.models import (
    ExperimentRunConfig,
    ExperimentStatus,
    PipelineConfig,
)


@pytest.fixture
def sample_pipeline() -> PipelineConfig:
    """Create a minimal pipeline config for Epictetus tests."""
    return PipelineConfig(
        name="epictetus-test",
        project_name="Test Project",
        project_spec="Build a test app",
        complexity="prototype",
        runs=[ExperimentRunConfig(num_agents=2)],
    )


@pytest.fixture
def pipeline(sample_pipeline: PipelineConfig, tmp_path: Path) -> ExperimentPipeline:
    """Create an ExperimentPipeline with a real implementation directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "config.yaml.template").write_text("template")
    (templates_dir / "agent_prompt.md").write_text("prompt")
    return ExperimentPipeline(
        config=sample_pipeline,
        templates_dir=templates_dir,
        base_dir=tmp_path / "experiments",
    )


class TestRunEpictetus:
    """Tests for ExperimentPipeline._run_epictetus()."""

    @patch("subprocess.run")
    def test_epictetus_invokes_claude_with_skill(
        self, mock_run: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """_run_epictetus must call claude CLI with /epictetus skill."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"

        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        pipeline._run_epictetus(0, tmux_session=None)

        assert mock_run.called
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "claude"
        assert "--print" in cmd_args or "--dangerously-skip-permissions" in cmd_args

    @patch("subprocess.run")
    def test_epictetus_passes_session_when_provided(
        self, mock_run: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """_run_epictetus must include --session in the Epictetus invocation."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"

        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")

        assert mock_run.called
        call_kwargs = mock_run.call_args
        # Session name should appear somewhere in the invocation
        # (either as arg or stdin input)
        all_text = str(call_kwargs)
        assert "marcus_test_run_0" in all_text

    @patch("subprocess.run")
    def test_epictetus_skipped_when_no_impl_dir(
        self, mock_run: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """_run_epictetus must skip (log EPICTETUS_SKIPPED) if no implementation/ dir."""
        # run_dir exists but no implementation/ subdirectory
        run_dir = tmp_path / "run_0"
        run_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = run_dir

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")

        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_epictetus_skipped_when_no_run_dir(
        self, mock_run: Mock, pipeline: ExperimentPipeline
    ) -> None:
        """_run_epictetus must skip gracefully if run dir not tracked."""
        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")

        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_epictetus_logs_complete_on_success(
        self, mock_run: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """_run_epictetus must log EPICTETUS_COMPLETE when returncode == 0."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"

        mock_run.return_value = Mock(returncode=0, stdout="done", stderr="")

        with patch.object(pipeline.events, "log") as mock_log:
            pipeline._run_epictetus(0, tmux_session=None)
            log_calls = [c[0][0] for c in mock_log.call_args_list]
            assert "EPICTETUS_COMPLETE" in log_calls

    @patch("subprocess.run")
    def test_epictetus_logs_failed_on_nonzero_returncode(
        self, mock_run: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """_run_epictetus must log EPICTETUS_FAILED when subprocess fails."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"

        mock_run.return_value = Mock(returncode=1, stdout="", stderr="audit failed")

        with patch.object(pipeline.events, "log") as mock_log:
            pipeline._run_epictetus(0, tmux_session=None)
            log_calls = [c[0][0] for c in mock_log.call_args_list]
            assert "EPICTETUS_FAILED" in log_calls


class TestEpictetusRunsBeforeTeardown:
    """Epictetus must fire before teardown so tmux session is still alive."""

    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    @patch("subprocess.run")
    def test_epictetus_called_before_teardown_kill(
        self,
        mock_subprocess: Mock,
        mock_kill: Mock,
        pipeline: ExperimentPipeline,
        tmp_path: Path,
    ) -> None:
        """Verify _run_epictetus() is called BEFORE kill_session() in teardown sequence."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"
        pipeline._run_start_times[0] = 0.0
        pipeline.run_statuses[0] = {"status": ExperimentStatus.RUNNING, "num_agents": 2}

        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        call_order: list[str] = []

        def record_subprocess(*args: object, **kwargs: object) -> Mock:
            call_order.append("epictetus")
            return Mock(returncode=0, stdout="", stderr="")

        def record_kill(session: str) -> None:
            call_order.append("kill")

        mock_subprocess.side_effect = record_subprocess
        mock_kill.side_effect = record_kill

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")
        pipeline.teardown_run(0, "marcus_test_run_0")

        assert call_order.index("epictetus") < call_order.index(
            "kill"
        ), "Epictetus must run before tmux kill — it needs the live session"

    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    @patch("subprocess.run")
    def test_teardown_still_runs_if_epictetus_fails(
        self,
        mock_subprocess: Mock,
        mock_kill: Mock,
        pipeline: ExperimentPipeline,
        tmp_path: Path,
    ) -> None:
        """teardown_run() must still execute even when Epictetus fails."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"
        pipeline._run_start_times[0] = 0.0
        pipeline.run_statuses[0] = {"status": ExperimentStatus.RUNNING, "num_agents": 2}

        # Epictetus fails with non-zero returncode
        mock_subprocess.return_value = Mock(returncode=1, stdout="", stderr="error")

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")
        pipeline.teardown_run(0, "marcus_test_run_0")

        mock_kill.assert_called_once_with("marcus_test_run_0")
