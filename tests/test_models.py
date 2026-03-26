"""Unit tests for experiment web UI Pydantic models."""

import pytest
from pydantic import ValidationError
from posidonius.models import (
    AgentConfig,
    ExperimentRunConfig,
    ExperimentStatus,
    OptimalAgentRequest,
    OptimalAgentResponse,
    PipelineConfig,
    PipelineStatus,
    RunStatus,
)


class TestAgentConfig:
    """Test suite for AgentConfig model."""

    def test_create_agent_with_all_fields(self) -> None:
        """Test creating an agent config with all fields specified."""
        agent = AgentConfig(
            id="agent_1",
            name="Developer 1",
            role="full-stack",
            skills=["python", "fastapi"],
            subagents=3,
        )
        assert agent.id == "agent_1"
        assert agent.name == "Developer 1"
        assert agent.role == "full-stack"
        assert agent.skills == ["python", "fastapi"]
        assert agent.subagents == 3

    def test_create_agent_with_defaults(self) -> None:
        """Test creating an agent config with default values."""
        agent = AgentConfig(
            id="agent_1",
            name="Developer 1",
            role="full-stack",
            skills=["python"],
        )
        assert agent.subagents == 0

    def test_agent_subagents_cannot_be_negative(self) -> None:
        """Test that subagents cannot be negative."""
        with pytest.raises(ValidationError):
            AgentConfig(
                id="agent_1",
                name="Developer 1",
                role="full-stack",
                skills=["python"],
                subagents=-1,
            )

    def test_agent_requires_at_least_one_skill(self) -> None:
        """Test that agent must have at least one skill."""
        with pytest.raises(ValidationError):
            AgentConfig(
                id="agent_1",
                name="Developer 1",
                role="full-stack",
                skills=[],
            )


class TestExperimentRunConfig:
    """Test suite for ExperimentRunConfig model."""

    def test_create_run_config_with_agent_count(self) -> None:
        """Test creating a run config that auto-generates agents."""
        run = ExperimentRunConfig(num_agents=5)
        assert run.num_agents == 5
        assert run.subagents_per_agent == 0

    def test_create_run_config_with_custom_agents(self) -> None:
        """Test creating a run config with explicit agent definitions."""
        agents = [
            AgentConfig(
                id="a1", name="A1", role="backend", skills=["python"]
            ),
            AgentConfig(
                id="a2", name="A2", role="frontend", skills=["react"]
            ),
        ]
        run = ExperimentRunConfig(num_agents=2, agents=agents)
        assert run.agents is not None and len(run.agents) == 2

    def test_run_config_requires_positive_agents(self) -> None:
        """Test that num_agents must be positive."""
        with pytest.raises(ValidationError):
            ExperimentRunConfig(num_agents=0)

    def test_run_config_with_subagents(self) -> None:
        """Test run config with subagents_per_agent."""
        run = ExperimentRunConfig(
            num_agents=3, subagents_per_agent=2
        )
        assert run.subagents_per_agent == 2


class TestPipelineConfig:
    """Test suite for PipelineConfig model."""

    def test_create_pipeline_with_sequential_runs(self) -> None:
        """Test creating a pipeline with multiple sequential runs."""
        pipeline = PipelineConfig(
            name="scaling-test",
            project_name="Snake Game",
            project_spec="Build a snake game",
            complexity="prototype",
            runs=[
                ExperimentRunConfig(num_agents=5),
                ExperimentRunConfig(num_agents=10),
                ExperimentRunConfig(num_agents=20),
            ],
        )
        assert pipeline.name == "scaling-test"
        assert len(pipeline.runs) == 3
        assert pipeline.runs[0].num_agents == 5
        assert pipeline.runs[2].num_agents == 20

    def test_pipeline_requires_at_least_one_run(self) -> None:
        """Test that pipeline must have at least one run."""
        with pytest.raises(ValidationError):
            PipelineConfig(
                name="empty",
                project_name="Test",
                project_spec="spec",
                complexity="prototype",
                runs=[],
            )

    def test_pipeline_default_values(self) -> None:
        """Test pipeline default values."""
        pipeline = PipelineConfig(
            name="test",
            project_name="Test",
            project_spec="spec",
            complexity="prototype",
            runs=[ExperimentRunConfig(num_agents=3)],
        )
        assert pipeline.provider == "planka"
        assert pipeline.mode == "new_project"
        assert pipeline.base_experiment_dir is None

    def test_pipeline_complexity_validation(self) -> None:
        """Test that complexity must be a valid option."""
        with pytest.raises(ValidationError):
            PipelineConfig(
                name="test",
                project_name="Test",
                project_spec="spec",
                complexity="invalid",
                runs=[ExperimentRunConfig(num_agents=3)],
            )

    def test_pipeline_default_skills(self) -> None:
        """Test pipeline default_skills are used when agents not specified."""
        pipeline = PipelineConfig(
            name="test",
            project_name="Test",
            project_spec="spec",
            complexity="prototype",
            default_skills=["python", "react"],
            runs=[ExperimentRunConfig(num_agents=3)],
        )
        assert pipeline.default_skills == ["python", "react"]

    def test_pipeline_timeouts(self) -> None:
        """Test pipeline timeout configuration."""
        pipeline = PipelineConfig(
            name="test",
            project_name="Test",
            project_spec="spec",
            complexity="prototype",
            runs=[ExperimentRunConfig(num_agents=3)],
            timeout_project_creation=600,
            timeout_agent_startup=120,
        )
        assert pipeline.timeout_project_creation == 600
        assert pipeline.timeout_agent_startup == 120


class TestOptimalAgentRequest:
    """Test suite for OptimalAgentRequest model."""

    def test_create_request(self) -> None:
        """Test creating an optimal agent request."""
        req = OptimalAgentRequest(
            project_spec="Build a REST API",
            complexity="standard",
        )
        assert req.project_spec == "Build a REST API"
        assert req.complexity == "standard"

    def test_request_defaults(self) -> None:
        """Test request default values."""
        req = OptimalAgentRequest(project_spec="spec")
        assert req.complexity == "standard"


class TestOptimalAgentResponse:
    """Test suite for OptimalAgentResponse model."""

    def test_create_response(self) -> None:
        """Test creating an optimal agent response."""
        resp = OptimalAgentResponse(
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
        assert resp.optimal_agents == 6
        assert len(resp.recommended_runs) == 3


class TestRunStatus:
    """Test suite for RunStatus model."""

    def test_create_run_status(self) -> None:
        """Test creating a run status."""
        status = RunStatus(
            run_index=0,
            num_agents=5,
            status=ExperimentStatus.RUNNING,
            tmux_session="marcus_test",
        )
        assert status.run_index == 0
        assert status.status == ExperimentStatus.RUNNING
        assert status.tasks_completed is None


class TestPipelineStatus:
    """Test suite for PipelineStatus model."""

    def test_create_pipeline_status(self) -> None:
        """Test creating a pipeline status."""
        status = PipelineStatus(
            pipeline_name="scaling-test",
            total_runs=3,
            current_run=1,
            status=ExperimentStatus.RUNNING,
            runs=[
                RunStatus(
                    run_index=0,
                    num_agents=5,
                    status=ExperimentStatus.COMPLETED,
                    tmux_session="marcus_test_0",
                ),
                RunStatus(
                    run_index=1,
                    num_agents=10,
                    status=ExperimentStatus.RUNNING,
                    tmux_session="marcus_test_1",
                ),
            ],
        )
        assert status.total_runs == 3
        assert status.current_run == 1
        assert len(status.runs) == 2
