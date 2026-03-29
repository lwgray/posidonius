"""Unit tests for MLflow tracking integration."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from posidonius.models import ExperimentRunConfig, PipelineConfig
from posidonius.tracking.mlflow_tracker import MLflowTracker


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
        ],
    )


class TestMLflowTracker:
    """Test suite for MLflowTracker."""

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_init_creates_experiment(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test tracker creates MLflow experiment on init."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        tracker = MLflowTracker(sample_pipeline)
        mock_mlflow.set_experiment.assert_called_once_with("scaling-test")
        assert tracker.experiment_id == "exp_123"

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_start_pipeline_run(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test starting a parent pipeline run."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        mock_run = MagicMock()
        mock_run.info.run_id = "parent_run_123"
        mock_mlflow.start_run.return_value = mock_run

        tracker = MLflowTracker(sample_pipeline)
        run_id = tracker.start_pipeline_run()

        mock_mlflow.start_run.assert_called_once_with(run_name="scaling-test_pipeline")
        mock_mlflow.log_params.assert_called_once()
        assert run_id == "parent_run_123"

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_start_child_run(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test starting a child run for a specific agent config."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        parent_run = MagicMock()
        parent_run.info.run_id = "parent_123"
        mock_mlflow.start_run.return_value = parent_run

        child_run = MagicMock()
        child_run.info.run_id = "child_456"

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()

        # Reset for child run call
        mock_mlflow.start_run.return_value = child_run
        child_id = tracker.start_child_run(
            run_index=0, num_agents=2, subagents_per_agent=0
        )

        assert child_id == "child_456"

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_log_run_metrics(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test logging metrics for a completed run."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        tracker = MLflowTracker(sample_pipeline)

        tracker.log_run_metrics(
            completion_time_seconds=360.0,
            tasks_completed=10,
            tasks_total=12,
            blockers=2,
        )
        mock_mlflow.log_metrics.assert_called_once_with(
            {
                "completion_time_seconds": 360.0,
                "tasks_completed": 10,
                "tasks_total": 12,
                "blockers": 2,
                "completion_rate": 10 / 12,
            }
        )

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_end_child_run(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test ending a child run."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        tracker = MLflowTracker(sample_pipeline)
        tracker.end_child_run(status="FINISHED")
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_end_pipeline_run(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test ending the parent pipeline run."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        parent_run = MagicMock()
        parent_run.info.run_id = "parent_123"
        mock_mlflow.start_run.return_value = parent_run

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()
        tracker.end_pipeline_run()

        mock_mlflow.end_run.assert_called()

    @patch("posidonius.tracking.mlflow_tracker.mlflow")
    def test_log_run_metrics_zero_tasks(
        self,
        mock_mlflow: Mock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Test logging metrics handles zero tasks gracefully."""
        mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp_123")
        tracker = MLflowTracker(sample_pipeline)
        tracker.log_run_metrics(
            completion_time_seconds=0.0,
            tasks_completed=0,
            tasks_total=0,
            blockers=0,
        )
        call_args = mock_mlflow.log_metrics.call_args[0][0]
        assert call_args["completion_rate"] == 0.0
