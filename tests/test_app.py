"""Unit tests for the FastAPI experiment web app."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from posidonius.app import create_app


@pytest.fixture
def tmp_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create temporary directories for the app."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "config.yaml.template").write_text(
        "template"
    )
    (templates_dir / "agent_prompt.md").write_text("prompt")
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    return {
        "templates_dir": templates_dir,
        "experiments_dir": experiments_dir,
    }


@pytest.fixture
def client(tmp_dirs: dict[str, Path]) -> TestClient:
    """Create a test client for the app."""
    app = create_app(
        templates_dir=tmp_dirs["templates_dir"],
        experiments_dir=tmp_dirs["experiments_dir"],
    )
    return TestClient(app)


def _create_payload(
    name: str = "test-exp",
) -> dict[str, object]:
    """Helper to create a valid experiment payload."""
    return {
        "name": name,
        "project_name": "Test Project",
        "project_spec": "Build something",
        "complexity": "prototype",
        "runs": [{"num_agents": 2}],
    }


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_returns_ok(
        self, client: TestClient
    ) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestListExperiments:
    """Test suite for listing experiments."""

    def test_list_empty(self, client: TestClient) -> None:
        """Test listing when no experiments exist."""
        response = client.get("/api/experiments")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_after_create(
        self, client: TestClient
    ) -> None:
        """Test listing after creating an experiment."""
        client.post(
            "/api/experiments", json=_create_payload()
        )
        response = client.get("/api/experiments")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert (
            response.json()[0]["pipeline_name"] == "test-exp"
        )


class TestCreateExperiment:
    """Test suite for creating experiments."""

    def test_create_pipeline(
        self, client: TestClient
    ) -> None:
        """Test creating a new experiment pipeline."""
        payload = _create_payload("scaling-test")
        payload["runs"] = [
            {"num_agents": 2},
            {"num_agents": 5},
            {"num_agents": 10},
        ]
        response = client.post(
            "/api/experiments", json=payload
        )
        assert response.status_code == 201
        data = response.json()
        assert data["pipeline_name"] == "scaling-test"
        assert data["total_runs"] == 3
        assert data["status"] == "pending"

    def test_create_with_invalid_complexity(
        self, client: TestClient
    ) -> None:
        """Test creating experiment with invalid complexity."""
        payload = _create_payload()
        payload["complexity"] = "invalid"
        response = client.post(
            "/api/experiments", json=payload
        )
        assert response.status_code == 422

    def test_create_duplicate_name_fails(
        self, client: TestClient
    ) -> None:
        """Test creating experiment with duplicate name fails."""
        payload = _create_payload("dup-test")
        client.post("/api/experiments", json=payload)
        response = client.post(
            "/api/experiments", json=payload
        )
        assert response.status_code == 409

    def test_create_with_custom_agents(
        self, client: TestClient
    ) -> None:
        """Test creating experiment with explicit agent configs."""
        payload = _create_payload("custom-agents")
        payload["complexity"] = "standard"
        payload["runs"] = [
            {
                "num_agents": 2,
                "agents": [
                    {
                        "id": "a1",
                        "name": "BE",
                        "role": "backend",
                        "skills": ["python"],
                        "subagents": 2,
                    },
                    {
                        "id": "a2",
                        "name": "FE",
                        "role": "frontend",
                        "skills": ["react"],
                    },
                ],
            }
        ]
        response = client.post(
            "/api/experiments", json=payload
        )
        assert response.status_code == 201


class TestGetExperimentStatus:
    """Test suite for getting experiment status."""

    def test_get_status(self, client: TestClient) -> None:
        """Test getting status of an existing experiment."""
        client.post(
            "/api/experiments",
            json=_create_payload("status-test"),
        )
        response = client.get(
            "/api/experiments/status-test"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_name"] == "status-test"
        assert data["status"] == "pending"

    def test_get_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Test getting status of non-existent experiment."""
        response = client.get(
            "/api/experiments/nonexistent"
        )
        assert response.status_code == 404


class TestStartExperiment:
    """Test suite for starting experiments."""

    @patch(
        "posidonius.engine.pipeline."
        "ExperimentPipeline.start_run"
    )
    def test_start_experiment(
        self, mock_start: Mock, client: TestClient
    ) -> None:
        """Test starting an experiment launches agents."""
        mock_start.return_value = "marcus_test_exp_run_0"
        client.post(
            "/api/experiments",
            json=_create_payload("start-test"),
        )
        response = client.post(
            "/api/experiments/start-test/start"
        )
        assert response.status_code == 200
        data = response.json()
        assert (
            data["tmux_session"] == "marcus_test_exp_run_0"
        )
        assert "Started run 0" in data["message"]
        mock_start.assert_called_once_with(0)

    def test_start_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Test starting non-existent experiment."""
        response = client.post(
            "/api/experiments/nonexistent/start"
        )
        assert response.status_code == 404


class TestGetExperimentOutput:
    """Test suite for live tmux output capture."""

    @patch(
        "posidonius.engine.pipeline."
        "ExperimentPipeline.get_run_output"
    )
    def test_get_output(
        self, mock_output: Mock, client: TestClient
    ) -> None:
        """Test getting live output from agent panes."""
        mock_output.return_value = [
            {
                "target": "s:0.0",
                "title": "Creator",
                "output": "Working...",
                "status": "working",
            },
            {
                "target": "s:0.1",
                "title": "Worker 1",
                "output": "Waiting...",
                "status": "waiting",
            },
        ]
        client.post(
            "/api/experiments",
            json=_create_payload("output-test"),
        )
        response = client.get(
            "/api/experiments/output-test/output"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["status"] == "working"
        assert data[1]["status"] == "waiting"

    def test_output_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Test getting output for non-existent experiment."""
        response = client.get(
            "/api/experiments/nonexistent/output"
        )
        assert response.status_code == 404


class TestStopExperiment:
    """Test suite for stopping experiments."""

    def test_stop_experiment(
        self, client: TestClient
    ) -> None:
        """Test stopping an experiment."""
        client.post(
            "/api/experiments",
            json=_create_payload("stop-test"),
        )
        response = client.delete(
            "/api/experiments/stop-test"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

    def test_stop_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Test stopping non-existent experiment."""
        response = client.delete(
            "/api/experiments/nonexistent"
        )
        assert response.status_code == 404


class TestOptimizeEndpoint:
    """Test suite for the optimal agent pre-flight endpoint."""

    @patch(
        "posidonius.engine.optimizer."
        "OptimalAgentOptimizer.analyze_sync"
    )
    def test_optimize_returns_recommendation(
        self, mock_analyze: Mock, client: TestClient
    ) -> None:
        """Test optimize endpoint returns agent recommendation."""
        from posidonius.models import (
            ExperimentRunConfig,
            OptimalAgentResponse,
        )

        mock_analyze.return_value = OptimalAgentResponse(
            optimal_agents=6,
            max_parallelism=4,
            total_tasks=15,
            critical_path_hours=2.5,
            efficiency_gain_percent=75.0,
            recommended_runs=[
                ExperimentRunConfig(num_agents=3),
                ExperimentRunConfig(num_agents=6),
                ExperimentRunConfig(num_agents=12),
            ],
        )
        response = client.post(
            "/api/experiments/optimize",
            json={
                "project_spec": "Build a REST API",
                "complexity": "standard",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["optimal_agents"] == 6
        assert len(data["recommended_runs"]) == 3


class TestExportEndpoint:
    """Test suite for terminal output export."""

    @patch(
        "posidonius.engine.tmux.TmuxManager.list_panes"
    )
    @patch(
        "posidonius.engine.tmux.TmuxManager.capture_pane"
    )
    def test_export_returns_zip(
        self,
        mock_capture: Mock,
        mock_list: Mock,
        client: TestClient,
    ) -> None:
        """Test export returns a zip file with pane output."""
        mock_list.return_value = [
            {"target": "s:0.0", "title": "Creator"},
            {"target": "s:0.1", "title": "Worker 1"},
        ]
        mock_capture.return_value = "some output\n"

        client.post(
            "/api/experiments",
            json=_create_payload("export-test"),
        )
        pipelines = client.app.state.pipelines  # type: ignore[union-attr]
        pipelines[
            "export-test"
        ].active_tmux_session = "marcus_test"

        response = client.get(
            "/api/experiments/export-test/export"
        )
        assert response.status_code == 200
        assert (
            response.headers["content-type"]
            == "application/zip"
        )

    def test_export_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Test export for non-existent experiment."""
        response = client.get(
            "/api/experiments/nonexistent/export"
        )
        assert response.status_code == 404

    def test_export_no_session_returns_400(
        self, client: TestClient
    ) -> None:
        """Test export when no active session."""
        client.post(
            "/api/experiments",
            json=_create_payload("no-session"),
        )
        response = client.get(
            "/api/experiments/no-session/export"
        )
        assert response.status_code == 400


class TestHistoryEndpoint:
    """Test suite for MLflow experiment history."""

    @patch("posidonius.app.mlflow")
    def test_history_returns_experiments(
        self, mock_mlflow: Mock, client: TestClient
    ) -> None:
        """Test history returns past experiments from MLflow."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_experiment.name = "scaling-test"
        mock_experiment.lifecycle_stage = "active"
        mock_mlflow.search_experiments.return_value = [
            mock_experiment
        ]

        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_run.info.run_name = "run_0_5_agents"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1711000000000
        mock_run.info.end_time = 1711000360000
        mock_run.data.params = {
            "num_agents": "5",
            "pipeline_name": "scaling-test",
        }
        mock_run.data.metrics = {
            "completion_time_seconds": 360.0,
            "tasks_completed": 10,
            "tasks_total": 12,
            "blockers": 1,
            "completion_rate": 0.833,
        }
        mock_mlflow.search_runs.return_value = [mock_run]

        response = client.get("/api/experiments/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert (
            data[0]["experiment_name"] == "scaling-test"
        )
        assert len(data[0]["runs"]) == 1
        assert (
            data[0]["runs"][0]["metrics"][
                "tasks_completed"
            ]
            == 10
        )

    @patch("posidonius.app.mlflow")
    def test_history_empty(
        self, mock_mlflow: Mock, client: TestClient
    ) -> None:
        """Test history returns empty list when no experiments."""
        mock_mlflow.search_experiments.return_value = []
        response = client.get("/api/experiments/history")
        assert response.status_code == 200
        assert response.json() == []


class TestStaticFiles:
    """Test suite for static file serving."""

    def test_index_page(self, client: TestClient) -> None:
        """Test serving the index page."""
        response = client.get("/")
        assert response.status_code == 200
        # Will show minimal HTML since static not present
        assert "Experiment Dashboard" in response.text
