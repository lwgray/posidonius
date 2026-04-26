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


def _mock_popen(returncode: int = 0) -> MagicMock:
    """Build a mock Popen object with the given returncode."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate.return_value = (b"", b"")
    return proc


class TestRunEpictetusSubprocessFallback:
    """Tests for _run_epictetus_subprocess (no tmux session available)."""

    @patch("subprocess.Popen")
    def test_invokes_claude_with_print_flag(
        self, mock_popen: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """Fallback path must call claude --print with /epictetus skill input."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"
        mock_popen.return_value = _mock_popen(returncode=0)

        pipeline._run_epictetus(0, tmux_session=None)

        assert mock_popen.called
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "claude"
        assert "--print" in cmd
        assert "--dangerously-skip-permissions" in cmd

    @patch("subprocess.Popen")
    def test_skill_input_contains_impl_dir(
        self, mock_popen: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """Skill input piped to claude must reference the implementation dir."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"
        mock_popen.return_value = _mock_popen(returncode=0)

        pipeline._run_epictetus(0, tmux_session=None)

        stdin_bytes = mock_popen.return_value.communicate.call_args[1]["input"]
        assert str(impl_dir).encode() in stdin_bytes

    @patch("subprocess.Popen")
    def test_logs_complete_on_success(
        self, mock_popen: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """Must log EPICTETUS_COMPLETE when subprocess exits 0."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"
        mock_popen.return_value = _mock_popen(returncode=0)

        with patch.object(pipeline.events, "log") as mock_log:
            pipeline._run_epictetus(0, tmux_session=None)
            log_calls = [c[0][0] for c in mock_log.call_args_list]
            assert "EPICTETUS_COMPLETE" in log_calls

    @patch("subprocess.Popen")
    def test_logs_failed_on_nonzero_returncode(
        self, mock_popen: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """Must log EPICTETUS_FAILED when subprocess exits non-zero."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"
        mock_popen.return_value = _mock_popen(returncode=1)

        with patch.object(pipeline.events, "log") as mock_log:
            pipeline._run_epictetus(0, tmux_session=None)
            log_calls = [c[0][0] for c in mock_log.call_args_list]
            assert "EPICTETUS_FAILED" in log_calls

    @patch("subprocess.Popen")
    def test_skipped_when_no_impl_dir(
        self, mock_popen: Mock, pipeline: ExperimentPipeline, tmp_path: Path
    ) -> None:
        """Must skip without any subprocess call if implementation/ missing."""
        run_dir = tmp_path / "run_0"
        run_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = run_dir

        pipeline._run_epictetus(0, tmux_session=None)

        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    def test_skipped_when_no_run_dir(
        self, mock_popen: Mock, pipeline: ExperimentPipeline
    ) -> None:
        """Must skip gracefully if run dir not tracked at all."""
        pipeline._run_epictetus(0, tmux_session=None)

        mock_popen.assert_not_called()


class TestRunEpictetusPassesSession:
    """Session name must reach the skill input in the tmux path."""

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_session_name_in_skill_input(
        self,
        mock_run: Mock,
        mock_popen: Mock,
        pipeline: ExperimentPipeline,
        tmp_path: Path,
    ) -> None:
        """--session flag must appear in the /epictetus skill input string."""
        impl_dir = tmp_path / "run_0" / "implementation"
        impl_dir.mkdir(parents=True)
        pipeline._run_dirs[0] = tmp_path / "run_0"

        # session_exists → False so we fall straight to subprocess fallback
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")
        mock_popen.return_value = _mock_popen(returncode=0)

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")

        stdin_bytes = mock_popen.return_value.communicate.call_args[1]["input"]
        assert b"marcus_test_run_0" in stdin_bytes


class TestEpictetusRunsBeforeTeardown:
    """Epictetus must fire before teardown so tmux session is still alive."""

    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_epictetus_called_before_teardown_kill(
        self,
        mock_run: Mock,
        mock_popen: Mock,
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

        # session_exists returns False → falls to Popen subprocess path
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")

        call_order: list[str] = []

        def record_popen(*args: object, **kwargs: object) -> MagicMock:
            call_order.append("epictetus")
            return _mock_popen(returncode=0)

        def record_kill(session: str) -> None:
            call_order.append("kill")

        mock_popen.side_effect = record_popen
        mock_kill.side_effect = record_kill

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")
        pipeline.teardown_run(0, "marcus_test_run_0")

        assert call_order.index("epictetus") < call_order.index(
            "kill"
        ), "Epictetus must run before tmux kill — it needs the live session"

    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_teardown_still_runs_if_epictetus_fails(
        self,
        mock_run: Mock,
        mock_popen: Mock,
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

        # session_exists returns False → Popen path; Popen exits non-zero
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")
        mock_popen.return_value = _mock_popen(returncode=1)

        pipeline._run_epictetus(0, tmux_session="marcus_test_run_0")
        pipeline.teardown_run(0, "marcus_test_run_0")

        mock_kill.assert_called_once_with("marcus_test_run_0")
