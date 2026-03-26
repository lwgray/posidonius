"""Unit tests for the sequential experiment pipeline engine."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from posidonius.engine.pipeline import ExperimentPipeline
from posidonius.models import (
    ExperimentRunConfig,
    ExperimentStatus,
    PipelineConfig,
)


@pytest.fixture
def sample_pipeline() -> PipelineConfig:
    """Create a sample pipeline config."""
    return PipelineConfig(
        name="scaling-test",
        project_name="Test Project",
        project_spec="Build a test app",
        complexity="prototype",
        runs=[
            ExperimentRunConfig(num_agents=2),
            ExperimentRunConfig(num_agents=5),
            ExperimentRunConfig(num_agents=10),
        ],
    )


@pytest.fixture
def pipeline(
    sample_pipeline: PipelineConfig, tmp_path: Path
) -> ExperimentPipeline:
    """Create an ExperimentPipeline for testing."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "config.yaml.template").write_text(
        "template"
    )
    (templates_dir / "agent_prompt.md").write_text("prompt")
    return ExperimentPipeline(
        config=sample_pipeline,
        templates_dir=templates_dir,
        base_dir=tmp_path / "experiments",
    )


class TestExperimentPipeline:
    """Test suite for ExperimentPipeline."""

    def test_init_state(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test pipeline initializes with correct state."""
        assert pipeline.current_run_index == -1
        assert pipeline.status == ExperimentStatus.PENDING

    def test_get_status_pending(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test status when pipeline hasn't started."""
        status = pipeline.get_status()
        assert status.pipeline_name == "scaling-test"
        assert status.total_runs == 3
        assert status.current_run == -1
        assert status.status == ExperimentStatus.PENDING

    def test_get_status_with_runs(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test status reflects completed runs."""
        pipeline.current_run_index = 1
        pipeline.status = ExperimentStatus.RUNNING
        pipeline.run_statuses[0] = {
            "status": ExperimentStatus.COMPLETED,
            "num_agents": 2,
            "tmux_session": "marcus_scaling_test_run_0",
        }
        status = pipeline.get_status()
        assert status.current_run == 1
        assert len(status.runs) == 1
        assert (
            status.runs[0].status == ExperimentStatus.COMPLETED
        )

    def test_is_run_complete_checks_completion_file(
        self,
        pipeline: ExperimentPipeline,
        tmp_path: Path,
    ) -> None:
        """Test completion check looks for monitor signal."""
        run_dir = (
            tmp_path / "experiments" / "scaling-test_run_0"
        )
        run_dir.mkdir(parents=True)
        assert pipeline.is_run_complete(run_dir) is False

        (run_dir / "experiment_complete.json").write_text(
            json.dumps({"status": "complete"})
        )
        assert pipeline.is_run_complete(run_dir) is True

    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    def test_teardown_run(
        self,
        mock_kill: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test teardown kills tmux session."""
        pipeline.run_statuses[0] = {
            "status": ExperimentStatus.RUNNING,
            "num_agents": 2,
        }
        pipeline.teardown_run(0, "marcus_scaling_test_run_0")
        mock_kill.assert_called_once_with(
            "marcus_scaling_test_run_0"
        )
        assert (
            pipeline.run_statuses[0]["status"]
            == ExperimentStatus.COMPLETED
        )

    def test_stop_sets_status(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test stopping pipeline updates status."""
        pipeline.status = ExperimentStatus.RUNNING
        pipeline.stop()
        assert pipeline.status == ExperimentStatus.STOPPED

    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    def test_stop_kills_active_session(
        self,
        mock_kill: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test stopping kills active tmux session."""
        pipeline.status = ExperimentStatus.RUNNING
        pipeline.current_run_index = 1
        pipeline.active_tmux_session = (
            "marcus_scaling_test_run_1"
        )
        pipeline.stop()
        mock_kill.assert_called_once_with(
            "marcus_scaling_test_run_1"
        )

    def test_prepare_next_run(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test preparing the next run in sequence."""
        run_dir = pipeline.prepare_next_run()
        assert run_dir is not None
        assert pipeline.current_run_index == 0
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "project_spec.md").exists()

    def test_prepare_next_run_increments(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test preparing runs increments the index."""
        pipeline.prepare_next_run()
        assert pipeline.current_run_index == 0
        pipeline.prepare_next_run()
        assert pipeline.current_run_index == 1

    def test_prepare_next_run_returns_none_when_done(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test returns None when all runs exhausted."""
        pipeline.current_run_index = 2
        result = pipeline.prepare_next_run()
        assert result is None

    @patch(
        "posidonius.engine.tmux.TmuxManager.session_exists"
    )
    @patch(
        "posidonius.engine.tmux.TmuxManager.capture_all_panes"
    )
    def test_get_run_output(
        self,
        mock_capture: Mock,
        mock_exists: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test getting live output from active tmux session."""
        pipeline.active_tmux_session = "marcus_test_run_0"
        mock_exists.return_value = True
        mock_capture.return_value = [
            {
                "target": "s:0.0",
                "title": "Creator",
                "output": "Working",
                "status": "working",
            },
        ]
        output = pipeline.get_run_output()
        assert len(output) == 1
        assert output[0]["status"] == "working"

    def test_get_run_output_no_session(
        self, pipeline: ExperimentPipeline
    ) -> None:
        """Test get_run_output returns empty when no session."""
        assert pipeline.get_run_output() == []


class TestPipelineMLflowIntegration:
    """Test suite for MLflow tracking wired into the pipeline."""

    @patch("posidonius.engine.pipeline.threading.Thread")
    @patch("posidonius.engine.pipeline.MLflowTracker")
    def test_start_run_creates_mlflow_parent_and_child(
        self,
        mock_tracker_cls: Mock,
        mock_thread_cls: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test that start_run creates MLflow parent + child runs."""
        mock_tracker = MagicMock()
        mock_tracker.start_pipeline_run.return_value = (
            "parent_123"
        )
        mock_tracker.start_child_run.return_value = "child_456"
        mock_tracker.experiment_id = "exp_789"
        mock_tracker_cls.return_value = mock_tracker

        # Mock thread so background worker doesn't run
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        pipeline.start_run(0)

        mock_tracker_cls.assert_called_once_with(pipeline.config)
        mock_tracker.start_pipeline_run.assert_called_once()
        mock_tracker.start_child_run.assert_called_once_with(
            run_index=0, num_agents=2, subagents_per_agent=0
        )
        assert (
            pipeline.run_statuses[0]["mlflow_run_id"]
            == "child_456"
        )
        mock_thread.start.assert_called_once()

    @patch("posidonius.engine.pipeline.MLflowTracker")
    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    def test_teardown_run_logs_metrics_and_ends_child(
        self,
        mock_kill: Mock,
        mock_tracker_cls: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test that teardown logs metrics and ends MLflow child."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        pipeline.tracker = mock_tracker
        pipeline._run_start_times[0] = time.time() - 120.0
        pipeline.run_statuses[0] = {
            "status": ExperimentStatus.RUNNING,
            "num_agents": 2,
            "tmux_session": "marcus_test_run_0",
        }

        pipeline.teardown_run(0, "marcus_test_run_0")

        mock_tracker.log_run_metrics.assert_called_once()
        call_kwargs = (
            mock_tracker.log_run_metrics.call_args[1]
        )
        assert call_kwargs["completion_time_seconds"] >= 120.0
        mock_tracker.end_child_run.assert_called_once_with(
            status="FINISHED"
        )

    @patch("posidonius.engine.pipeline.MLflowTracker")
    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    def test_stop_ends_mlflow_runs_as_killed(
        self,
        mock_kill: Mock,
        mock_tracker_cls: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test that stop() ends MLflow runs with KILLED status."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        pipeline.tracker = mock_tracker
        pipeline.status = ExperimentStatus.RUNNING
        pipeline.active_tmux_session = "marcus_test_run_0"

        pipeline.stop()

        mock_tracker.end_child_run.assert_called_once_with(
            status="KILLED"
        )
        mock_tracker.end_pipeline_run.assert_called_once_with(
            status="KILLED"
        )

    @patch("posidonius.engine.pipeline.MLflowTracker")
    @patch("posidonius.engine.tmux.TmuxManager.kill_session")
    def test_stop_without_tracker_does_not_error(
        self,
        mock_kill: Mock,
        mock_tracker_cls: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test stop works even if tracker was never initialized."""
        pipeline.status = ExperimentStatus.RUNNING
        pipeline.active_tmux_session = "marcus_test_run_0"
        # tracker is None (never started)
        pipeline.stop()
        assert pipeline.status == ExperimentStatus.STOPPED

    @patch("posidonius.engine.pipeline.MLflowTracker")
    def test_get_status_includes_mlflow_experiment_id(
        self,
        mock_tracker_cls: Mock,
        pipeline: ExperimentPipeline,
    ) -> None:
        """Test that status includes MLflow experiment ID."""
        mock_tracker = MagicMock()
        mock_tracker.experiment_id = "exp_789"
        pipeline.tracker = mock_tracker
        status = pipeline.get_status()
        assert status.mlflow_experiment_id == "exp_789"
