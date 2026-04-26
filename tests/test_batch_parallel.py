"""Unit tests for batch-parallel experiment launch.

Verifies that POST /api/experiments/batch-parallel creates N independent
pipelines, applies per-slot overrides (num_agents, complexity, project_spec),
and injects per-instance MARCUS_URL / SQLITE_KANBAN_DB_PATH into each
subprocess environment.
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


class TestBatchSlotOverrides:
    """Per-slot overrides are applied on top of the base pipeline config."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
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

    def _base_request(self, instances: list) -> dict:
        return {
            "pipeline_config": {
                "name": "base-exp",
                "project_name": "Base Experiment",
                "project_spec": "Default spec",
                "complexity": "prototype",
                "runs": [{"num_agents": 2}],
            },
            "marcus_instances": instances,
        }

    def test_num_agents_override_applied(self, client: TestClient) -> None:
        """A slot with num_agents override must use that agent count, not the base."""
        req = self._base_request(
            [
                {"url": "http://localhost:4299/mcp", "num_agents": 5},
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        pipeline = resp.json()["pipelines"][0]
        assert pipeline["config_agents"] == 5

    def test_complexity_override_applied(self, client: TestClient) -> None:
        """A slot with complexity override must use that complexity."""
        req = self._base_request(
            [
                {"url": "http://localhost:4299/mcp", "complexity": "standard"},
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        # Verify via list — check the pipeline was created (deeper check via pipeline object)
        data = resp.json()
        assert len(data["pipelines"]) == 1
        assert data["pipelines"][0]["overrides"]["complexity"] == "standard"

    def test_project_spec_override_applied(self, client: TestClient) -> None:
        """A slot with project_spec override must surface its spec in overrides."""
        req = self._base_request(
            [
                {
                    "url": "http://localhost:4299/mcp",
                    "project_spec": "Custom spec for this slot",
                },
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        ov = resp.json()["pipelines"][0]["overrides"]
        assert "project_spec" in ov
        assert "Custom spec" in ov["project_spec"]

    def test_slots_without_overrides_inherit_base(self, client: TestClient) -> None:
        """A slot with no overrides must use the base config's agent count."""
        req = self._base_request(
            [
                {"url": "http://localhost:4299/mcp"},  # no overrides
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        pipeline = resp.json()["pipelines"][0]
        assert pipeline["config_agents"] == 2  # base default
        assert pipeline["overrides"] == {}  # nothing overridden

    def test_mixed_overrides_across_slots(self, client: TestClient) -> None:
        """Different slots can vary independently — agent counts differ across slots."""
        req = self._base_request(
            [
                {"url": "http://localhost:4299/mcp", "num_agents": 2, "label": "2a"},
                {"url": "http://localhost:4300/mcp", "num_agents": 4, "label": "4a"},
                {"url": "http://localhost:4301/mcp", "num_agents": 8, "label": "8a"},
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        pipelines = resp.json()["pipelines"]
        assert len(pipelines) == 3
        agent_counts = [p["overrides"]["num_agents"] for p in pipelines]
        assert agent_counts == [2, 4, 8]

    def test_slot_label_appears_in_pipeline_name(self, client: TestClient) -> None:
        """Custom label must appear in the generated pipeline name."""
        req = self._base_request(
            [
                {
                    "url": "http://localhost:4299/mcp",
                    "num_agents": 3,
                    "label": "control",
                },
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        name = resp.json()["pipelines"][0]["pipeline_name"]
        assert "control" in name

    def test_response_includes_overrides_summary(self, client: TestClient) -> None:
        """Each pipeline in the response must include an overrides dict."""
        req = self._base_request(
            [
                {
                    "url": "http://localhost:4299/mcp",
                    "num_agents": 3,
                    "complexity": "standard",
                },
            ]
        )
        resp = client.post("/api/experiments/batch-parallel", json=req)
        assert resp.status_code == 201
        pipeline = resp.json()["pipelines"][0]
        assert "overrides" in pipeline
        assert pipeline["overrides"]["num_agents"] == 3
        assert pipeline["overrides"]["complexity"] == "standard"


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

    def _run_start_run_and_wait_for_popen(
        self, pipeline: ExperimentPipeline, mock_popen: Mock, tmp_path: Path
    ) -> None:
        """Start a pipeline run and wait until Popen is actually called.

        ``start_run`` spawns a daemon thread that calls Popen. Joining the
        outer thread only waits for ``start_run`` to return, not for the
        inner daemon thread to call Popen. Poll until the mock records the
        call (or 5 s elapse) to avoid a race.
        """
        import threading
        import time

        t = threading.Thread(target=pipeline.start_run, args=(0,), daemon=True)
        t.start()
        t.join(timeout=3)
        # Poll until the inner daemon thread calls Popen (should be <100 ms).
        deadline = time.monotonic() + 5.0
        while not mock_popen.called and time.monotonic() < deadline:
            time.sleep(0.05)

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
                    self._run_start_run_and_wait_for_popen(pipeline, mock_popen, tmp_path)

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
                    self._run_start_run_and_wait_for_popen(pipeline, mock_popen, tmp_path)

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
