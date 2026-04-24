"""Unit tests for batch-parallel experiment launch.

Verifies that POST /api/experiments/batch-parallel creates N independent
pipelines simultaneously and injects per-instance MARCUS_URL /
SQLITE_KANBAN_DB_PATH into each subprocess environment.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from posidonius.app import create_app
from posidonius.engine.pipeline import ExperimentPipeline
from posidonius.models import ExperimentRunConfig, PipelineConfig


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create test client with temp dirs."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "config.yaml.template").write_text("template")
    (templates_dir / "agent_prompt.md").write_text("prompt")
    app = create_app(
        templates_dir=templates_dir,
        experiments_dir=tmp_path / "experiments",
    )
    return TestClient(app)


@pytest.fixture
def batch_request() -> dict:
    """Minimal valid batch-parallel request body."""
    return {
        "pipeline_config": {
            "name": "weather-dashboard",
            "project_name": "Weather Dashboard",
            "project_spec": "Build a weather dashboard",
            "complexity": "prototype",
            "runs": [{"num_agents": 2}],
        },
        "marcus_instances": [
            {"url": "http://localhost:4298/mcp", "db_path": "./data/exp_0.db"},
            {"url": "http://localhost:4299/mcp", "db_path": "./data/exp_1.db"},
        ],
    }


class TestBatchParallelEndpoint:
    """POST /api/experiments/batch-parallel creates N pipelines."""

    def test_endpoint_exists(self, client: TestClient, batch_request: dict) -> None:
        """Endpoint must exist and not return 404."""
        resp = client.post("/api/experiments/batch-parallel", json=batch_request)
        assert resp.status_code != 404

    def test_creates_n_pipelines(self, client: TestClient, batch_request: dict) -> None:
        """One pipeline per marcus_instance must be created."""
        resp = client.post("/api/experiments/batch-parallel", json=batch_request)
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["pipelines"]) == 2

    def test_pipeline_names_are_unique(
        self, client: TestClient, batch_request: dict
    ) -> None:
        """Each pipeline must have a distinct name."""
        resp = client.post("/api/experiments/batch-parallel", json=batch_request)
        assert resp.status_code == 201
        names = [p["pipeline_name"] for p in resp.json()["pipelines"]]
        assert len(set(names)) == len(names)

    def test_pipelines_visible_in_list(
        self, client: TestClient, batch_request: dict
    ) -> None:
        """Created pipelines must appear in GET /api/experiments."""
        client.post("/api/experiments/batch-parallel", json=batch_request)
        list_resp = client.get("/api/experiments")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 2

    def test_requires_at_least_one_instance(self, client: TestClient) -> None:
        """Request with empty marcus_instances must be rejected."""
        resp = client.post(
            "/api/experiments/batch-parallel",
            json={
                "pipeline_config": {
                    "name": "test",
                    "project_name": "Test",
                    "project_spec": "spec",
                    "complexity": "prototype",
                    "runs": [{"num_agents": 2}],
                },
                "marcus_instances": [],
            },
        )
        assert resp.status_code == 422  # validation error


class TestMarcusInstanceEnvInjection:
    """ExperimentPipeline must pass MARCUS_URL + SQLITE_KANBAN_DB_PATH to subprocess."""

    def _make_pipeline(
        self, tmp_path: Path, marcus_instance: dict
    ) -> ExperimentPipeline:
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(exist_ok=True)
        (templates_dir / "config.yaml.template").write_text("template")
        (templates_dir / "agent_prompt.md").write_text("prompt")
        config = PipelineConfig(
            name="test-pipeline",
            project_name="Test",
            project_spec="Build something",
            complexity="prototype",
            runs=[ExperimentRunConfig(num_agents=2)],
        )
        return ExperimentPipeline(
            config=config,
            templates_dir=templates_dir,
            base_dir=tmp_path / "experiments",
            marcus_instance=marcus_instance,
        )

    @patch("subprocess.Popen")
    @patch("posidonius.tracking.mlflow_tracker.MLflowTracker")
    def test_marcus_url_passed_to_popen(
        self, mock_tracker_cls: Mock, mock_popen: Mock, tmp_path: Path
    ) -> None:
        """MARCUS_URL must appear in env passed to subprocess.Popen."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        instance = {"url": "http://localhost:4299/mcp", "db_path": "./data/exp_1.db"}
        pipeline = self._make_pipeline(tmp_path, instance)

        with patch.object(
            pipeline.runner, "prepare_run", return_value=tmp_path / "run_0"
        ):
            with patch.object(
                pipeline.runner, "get_tmux_session_name", return_value="sess"
            ):
                with patch.object(pipeline.tmux, "session_exists", return_value=False):
                    import threading

                    t = threading.Thread(
                        target=pipeline.start_run, args=(0,), daemon=True
                    )
                    t.start()
                    t.join(timeout=3)

        assert mock_popen.called
        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("MARCUS_URL") == "http://localhost:4299/mcp"

    @patch("subprocess.Popen")
    @patch("posidonius.tracking.mlflow_tracker.MLflowTracker")
    def test_sqlite_db_path_passed_to_popen(
        self, mock_tracker_cls: Mock, mock_popen: Mock, tmp_path: Path
    ) -> None:
        """SQLITE_KANBAN_DB_PATH must appear in env passed to subprocess.Popen."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        instance = {"url": "http://localhost:4299/mcp", "db_path": "./data/exp_1.db"}
        pipeline = self._make_pipeline(tmp_path, instance)

        with patch.object(
            pipeline.runner, "prepare_run", return_value=tmp_path / "run_0"
        ):
            with patch.object(
                pipeline.runner, "get_tmux_session_name", return_value="sess"
            ):
                with patch.object(pipeline.tmux, "session_exists", return_value=False):
                    import threading

                    t = threading.Thread(
                        target=pipeline.start_run, args=(0,), daemon=True
                    )
                    t.start()
                    t.join(timeout=3)

        assert mock_popen.called
        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("SQLITE_KANBAN_DB_PATH") == "./data/exp_1.db"

    @patch("subprocess.Popen")
    @patch("posidonius.tracking.mlflow_tracker.MLflowTracker")
    def test_no_instance_uses_inherited_env(
        self, mock_tracker_cls: Mock, mock_popen: Mock, tmp_path: Path
    ) -> None:
        """Pipeline with no marcus_instance must not inject MARCUS_URL."""
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "config.yaml.template").write_text("template")
        (templates_dir / "agent_prompt.md").write_text("prompt")
        config = PipelineConfig(
            name="test-pipeline",
            project_name="Test",
            project_spec="spec",
            complexity="prototype",
            runs=[ExperimentRunConfig(num_agents=2)],
        )
        pipeline = ExperimentPipeline(
            config=config,
            templates_dir=templates_dir,
            base_dir=tmp_path / "experiments",
        )

        with patch.object(
            pipeline.runner, "prepare_run", return_value=tmp_path / "run_0"
        ):
            with patch.object(
                pipeline.runner, "get_tmux_session_name", return_value="sess"
            ):
                with patch.object(pipeline.tmux, "session_exists", return_value=False):
                    import threading

                    t = threading.Thread(
                        target=pipeline.start_run, args=(0,), daemon=True
                    )
                    t.start()
                    t.join(timeout=3)

        if mock_popen.called:
            call_kwargs = mock_popen.call_args[1]
            env = call_kwargs.get("env")
            # env should be None (inherit) or not contain MARCUS_URL override
            if env is not None:
                # If env is set, MARCUS_URL should come from the real environment
                assert env.get("MARCUS_URL") == os.environ.get("MARCUS_URL")
