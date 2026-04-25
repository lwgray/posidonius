"""Unit tests for MLflow tracking integration."""

from unittest.mock import MagicMock, call, patch

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


def _make_client_mock() -> MagicMock:
    """Return a MlflowClient mock with sensible defaults."""
    client = MagicMock()
    experiment = MagicMock()
    experiment.experiment_id = "exp_123"
    client.get_experiment_by_name.return_value = experiment

    parent_run = MagicMock()
    parent_run.info.run_id = "parent_run_123"
    child_run = MagicMock()
    child_run.info.run_id = "child_run_456"
    client.create_run.side_effect = [parent_run, child_run]
    return client


class TestMLflowTracker:
    """Test suite for MLflowTracker."""

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_init_gets_existing_experiment(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Tracker reuses existing experiment without creating a duplicate."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)

        client.get_experiment_by_name.assert_called_once_with("scaling-test")
        client.create_experiment.assert_not_called()
        assert tracker.experiment_id == "exp_123"

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_init_creates_experiment_when_missing(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Tracker creates the experiment when it does not exist yet."""
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client.create_experiment.return_value = "new_exp_999"
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)

        client.create_experiment.assert_called_once_with("scaling-test")
        assert tracker.experiment_id == "new_exp_999"

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_start_pipeline_run(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """start_pipeline_run creates a run and logs params via client."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)
        run_id = tracker.start_pipeline_run()

        client.create_run.assert_called_once_with(
            experiment_id="exp_123",
            run_name="scaling-test_pipeline",
        )
        client.log_batch.assert_called_once()
        assert run_id == "parent_run_123"
        assert tracker.parent_run_id == "parent_run_123"

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_start_child_run(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """start_child_run creates a nested run tagged with parent ID."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()
        child_id = tracker.start_child_run(run_index=0, num_agents=2, subagents_per_agent=0)

        assert child_id == "child_run_456"
        # Second create_run call should carry the parent tag
        second_call = client.create_run.call_args_list[1]
        assert second_call.kwargs["tags"]["mlflow.parentRunId"] == "parent_run_123"

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_log_run_metrics(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """log_run_metrics calls log_batch on the active child run."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()
        tracker.start_child_run(run_index=0, num_agents=2, subagents_per_agent=0)
        client.log_batch.reset_mock()

        tracker.log_run_metrics(
            completion_time_seconds=360.0,
            tasks_completed=10,
            tasks_total=12,
            blockers=2,
        )

        client.log_batch.assert_called_once()
        call_kwargs = client.log_batch.call_args
        run_id_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("run_id")
        assert run_id_arg == "child_run_456"
        metrics = {m.key: m.value for m in call_kwargs.kwargs.get("metrics", [])}
        assert metrics["completion_rate"] == pytest.approx(10 / 12)
        assert metrics["blockers"] == 2

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_log_run_metrics_zero_tasks(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """log_run_metrics sets completion_rate=0 when tasks_total is zero."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()
        tracker.start_child_run(run_index=0, num_agents=2, subagents_per_agent=0)
        client.log_batch.reset_mock()

        tracker.log_run_metrics(
            completion_time_seconds=0.0,
            tasks_completed=0,
            tasks_total=0,
            blockers=0,
        )

        metrics = {
            m.key: m.value
            for m in client.log_batch.call_args.kwargs.get("metrics", [])
        }
        assert metrics["completion_rate"] == 0.0

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_end_child_run(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """end_child_run terminates the active child run and clears it."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()
        tracker.start_child_run(run_index=0, num_agents=2, subagents_per_agent=0)
        tracker.end_child_run(status="FINISHED")

        client.set_terminated.assert_called_once_with("child_run_456", status="FINISHED")
        assert tracker._active_child_run_id is None

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_end_pipeline_run(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """end_pipeline_run terminates the parent run."""
        client = _make_client_mock()
        mock_client_cls.return_value = client

        tracker = MLflowTracker(sample_pipeline)
        tracker.start_pipeline_run()
        tracker.end_pipeline_run(status="FINISHED")

        client.set_terminated.assert_called_once_with("parent_run_123", status="FINISHED")

    @patch("posidonius.tracking.mlflow_tracker.MlflowClient")
    def test_concurrent_pipelines_do_not_share_state(
        self,
        mock_client_cls: MagicMock,
        sample_pipeline: PipelineConfig,
    ) -> None:
        """Two trackers created simultaneously have independent run IDs.

        This is the core regression test for the fluent-API global-state bug:
        mlflow.start_run() raises if called when a run is already active in
        the process. MlflowClient has no such constraint.
        """
        def make_client() -> MagicMock:
            c = MagicMock()
            exp = MagicMock()
            exp.experiment_id = "exp_x"
            c.get_experiment_by_name.return_value = exp
            run = MagicMock()
            run.info.run_id = f"run_{id(c)}"
            c.create_run.return_value = run
            return c

        clients = [make_client(), make_client()]
        mock_client_cls.side_effect = clients

        pipeline2 = sample_pipeline.model_copy(update={"name": "other-pipeline"})
        tracker1 = MLflowTracker(sample_pipeline)
        tracker2 = MLflowTracker(pipeline2)

        id1 = tracker1.start_pipeline_run()
        id2 = tracker2.start_pipeline_run()

        assert id1 != id2
        assert tracker1.parent_run_id != tracker2.parent_run_id
