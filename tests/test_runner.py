"""Unit tests for experiment runner with tmux teardown."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from posidonius.engine.runner import ExperimentRunner
from posidonius.models import (
    AgentConfig,
    ExperimentRunConfig,
    PipelineConfig,
)


@pytest.fixture
def sample_pipeline() -> PipelineConfig:
    """Create a sample pipeline config for testing."""
    return PipelineConfig(
        name="test-pipeline",
        project_name="Test Project",
        project_spec="Build a test app",
        complexity="prototype",
        runs=[
            ExperimentRunConfig(num_agents=2),
            ExperimentRunConfig(num_agents=4),
        ],
    )


@pytest.fixture
def runner(sample_pipeline: PipelineConfig, tmp_path: Path) -> ExperimentRunner:
    """Create an ExperimentRunner instance for testing."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "config.yaml.template").write_text("template")
    (templates_dir / "agent_prompt.md").write_text("prompt")
    return ExperimentRunner(
        pipeline=sample_pipeline,
        templates_dir=templates_dir,
        base_dir=tmp_path / "experiments",
    )


class TestExperimentRunner:
    """Test suite for ExperimentRunner."""

    def test_init_creates_base_dir(
        self, runner: ExperimentRunner, tmp_path: Path
    ) -> None:
        """Test that runner creates base experiment directory."""
        assert (tmp_path / "experiments").exists()

    def test_generate_tmux_session_name(self, runner: ExperimentRunner) -> None:
        """Test tmux session name generation."""
        name = runner.get_tmux_session_name(0)
        assert name == "marcus_test_project-run_0-2_agents"

    def test_generate_agents_from_count(self, runner: ExperimentRunner) -> None:
        """Test auto-generating agent configs from num_agents."""
        run_config = ExperimentRunConfig(num_agents=3)
        agents = runner.generate_agents(run_config)
        assert len(agents) == 3
        assert agents[0].id == "agent_unicorn_1"
        assert agents[0].role == "full-stack"
        assert len(agents[0].skills) > 0

    def test_generate_agents_preserves_custom(self, runner: ExperimentRunner) -> None:
        """Test that explicit agent configs are preserved."""
        custom_agents = [
            AgentConfig(
                id="custom_1",
                name="Custom",
                role="backend",
                skills=["python"],
            )
        ]
        run_config = ExperimentRunConfig(num_agents=1, agents=custom_agents)
        agents = runner.generate_agents(run_config)
        assert len(agents) == 1
        assert agents[0].id == "custom_1"

    def test_generate_agents_with_subagents(self, runner: ExperimentRunner) -> None:
        """Test agent generation with subagents_per_agent."""
        run_config = ExperimentRunConfig(num_agents=2, subagents_per_agent=3)
        agents = runner.generate_agents(run_config)
        assert all(a.subagents == 3 for a in agents)

    def test_create_run_directory(self, runner: ExperimentRunner) -> None:
        """Test experiment directory creation for a run."""
        run_dir = runner.create_run_directory(0)
        assert run_dir.exists()
        assert (run_dir / "implementation").exists()
        assert (run_dir / "prompts").exists()
        assert (run_dir / "logs").exists()

    def test_generate_config_yaml(self, runner: ExperimentRunner) -> None:
        """Test config.yaml generation from pipeline config."""
        run_config = ExperimentRunConfig(num_agents=2)
        agents = runner.generate_agents(run_config)
        config_dict = runner.generate_config_dict(run_config, agents, 0)

        assert config_dict["project_name"] == "Test Project-run_0-2_agents"
        assert len(config_dict["agents"]) == 2
        assert config_dict["project_options"]["complexity"] == "prototype"

    @patch("posidonius.engine.runner.subprocess.run")
    def test_kill_tmux_session(self, mock_run: Mock, runner: ExperimentRunner) -> None:
        """Test clean tmux session teardown."""
        runner.kill_tmux_session("marcus_test_run_0")
        mock_run.assert_called_once_with(
            ["tmux", "kill-session", "-t", "marcus_test_run_0"],
            capture_output=True,
        )

    @patch("posidonius.engine.runner.subprocess.run")
    def test_kill_tmux_session_handles_missing(
        self, mock_run: Mock, runner: ExperimentRunner
    ) -> None:
        """Test teardown handles non-existent session gracefully."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "tmux")
        # Should not raise
        runner.kill_tmux_session("nonexistent")

    @patch("posidonius.engine.runner.subprocess.run")
    def test_list_active_sessions(
        self, mock_run: Mock, runner: ExperimentRunner
    ) -> None:
        """Test listing active tmux sessions."""
        mock_run.return_value = MagicMock(
            stdout="marcus_test_project-run_0-2_agents: 1 windows\n"
            "marcus_test_project-run_1-4_agents: 1 windows\n"
            "other_session: 1 windows\n",
            returncode=0,
        )
        sessions = runner.list_active_sessions()
        assert len(sessions) == 2
        assert "marcus_test_project-run_0-2_agents" in sessions

    @patch("posidonius.engine.runner.subprocess.run")
    def test_cleanup_all_sessions(
        self, mock_run: Mock, runner: ExperimentRunner
    ) -> None:
        """Test cleaning up all pipeline tmux sessions."""
        mock_run.return_value = MagicMock(
            stdout="marcus_test_project-run_0-2_agents: 1 windows\n"
            "marcus_test_project-run_1-4_agents: 1 windows\n",
            returncode=0,
        )
        runner.cleanup_all_sessions()
        assert mock_run.call_count == 3

    def test_write_project_spec(self, runner: ExperimentRunner, tmp_path: Path) -> None:
        """Test writing project spec to run directory."""
        run_dir = runner.create_run_directory(0)
        runner.write_project_spec(run_dir)
        spec_file = run_dir / "project_spec.md"
        assert spec_file.exists()
        assert spec_file.read_text() == "Build a test app"
