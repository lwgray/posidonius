"""Pydantic models for the Posidonius experiment dashboard.

Parameters
----------
These models define the configuration schema for experiment pipelines,
agent configurations, and status tracking.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AgentConfig(BaseModel):
    """Configuration for a single agent in an experiment.

    Attributes
    ----------
    id : str
        Unique agent identifier.
    name : str
        Human-readable agent name.
    role : str
        Agent role (e.g., 'full-stack', 'backend', 'qa').
    skills : list[str]
        List of skill tags for task matching.
    subagents : int
        Number of subagents to register for this agent.
    """

    id: str
    name: str
    role: str
    skills: list[str] = Field(min_length=1)
    subagents: int = Field(default=0, ge=0)


class ExperimentRunConfig(BaseModel):
    """Configuration for a single experiment run within a pipeline.

    Attributes
    ----------
    num_agents : int
        Number of agents to spawn for this run.
    subagents_per_agent : int
        Default subagents per agent (used when agents not explicitly defined).
    agents : list[AgentConfig] | None
        Optional explicit agent definitions. If None, agents are
        auto-generated from num_agents.
    """

    num_agents: int = Field(gt=0)
    subagents_per_agent: int = Field(default=0, ge=0)
    agents: Optional[list[AgentConfig]] = None


class PipelineConfig(BaseModel):
    """Configuration for a sequential experiment pipeline.

    Attributes
    ----------
    name : str
        Pipeline name (used for MLflow experiment grouping).
    project_name : str
        Name of the project to create in Marcus.
    project_spec : str
        Full project specification / prompt text.
    complexity : str
        Project complexity level.
    provider : str
        Kanban provider.
    mode : str
        Project creation mode.
    default_skills : list[str]
        Default skills for auto-generated agents.
    runs : list[ExperimentRunConfig]
        Sequential experiment runs to execute.
    base_experiment_dir : str | None
        Base directory for experiment output.
    timeout_project_creation : int
        Timeout for project creation in seconds.
    timeout_agent_startup : int
        Timeout for agent startup in seconds.
    """

    name: str
    project_name: str
    project_spec: str
    complexity: str
    provider: str = "planka"
    mode: str = "new_project"
    default_skills: list[str] = Field(
        default_factory=lambda: [
            "python",
            "javascript",
            "typescript",
            "react",
            "fastapi",
            "sqlalchemy",
            "postgresql",
            "database-design",
            "rest-api",
            "jwt",
            "security",
            "pytest",
            "integration-testing",
        ]
    )
    runs: list[ExperimentRunConfig] = Field(min_length=1)
    base_experiment_dir: Optional[str] = None
    timeout_project_creation: int = 300
    timeout_agent_startup: int = 60

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        """Validate complexity is a known level."""
        valid = {"prototype", "standard", "enterprise"}
        if v not in valid:
            msg = f"complexity must be one of {valid}, got '{v}'"
            raise ValueError(msg)
        return v


class OptimalAgentRequest(BaseModel):
    """Request to estimate optimal agent count.

    Attributes
    ----------
    project_spec : str
        Project specification text for analysis.
    complexity : str
        Project complexity level.
    """

    project_spec: str
    complexity: str = "standard"


class OptimalAgentResponse(BaseModel):
    """Response from optimal agent count estimation.

    Attributes
    ----------
    optimal_agents : int
        Recommended number of agents.
    max_parallelism : int
        Maximum tasks that can run simultaneously.
    total_tasks : int
        Total number of tasks in the project.
    critical_path_hours : float
        Estimated critical path duration in hours.
    efficiency_gain_percent : float
        Efficiency gain vs single agent.
    recommended_runs : list[ExperimentRunConfig]
        Suggested pipeline runs for scaling analysis.
    """

    optimal_agents: int
    max_parallelism: int
    total_tasks: int
    critical_path_hours: float
    efficiency_gain_percent: float
    recommended_runs: list[ExperimentRunConfig] = Field(default_factory=list)


class ExperimentStatus(str, Enum):
    """Status of an experiment run or pipeline.

    Attributes
    ----------
    PENDING : str
        Not yet started.
    RUNNING : str
        Currently executing.
    COMPLETED : str
        Finished successfully.
    FAILED : str
        Failed with error.
    STOPPED : str
        Manually stopped by user.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class RunStatus(BaseModel):
    """Status of a single experiment run.

    Attributes
    ----------
    run_index : int
        Index of this run in the pipeline.
    num_agents : int
        Number of agents in this run.
    status : ExperimentStatus
        Current status.
    tmux_session : str | None
        Tmux session name if running.
    tasks_completed : int | None
        Number of tasks completed so far.
    tasks_total : int | None
        Total number of tasks.
    mlflow_run_id : str | None
        MLflow run ID for tracking.
    error : str | None
        Error message if failed.
    """

    run_index: int
    num_agents: int
    status: ExperimentStatus
    tmux_session: Optional[str] = None
    tasks_completed: Optional[int] = None
    tasks_total: Optional[int] = None
    mlflow_run_id: Optional[str] = None
    error: Optional[str] = None


class PipelineStatus(BaseModel):
    """Status of a full experiment pipeline.

    Attributes
    ----------
    pipeline_name : str
        Name of the pipeline.
    total_runs : int
        Total number of runs in the pipeline.
    current_run : int
        Index of the currently executing run.
    status : ExperimentStatus
        Overall pipeline status.
    runs : list[RunStatus]
        Status of each individual run.
    mlflow_experiment_id : str | None
        MLflow parent experiment ID.
    """

    pipeline_name: str
    total_runs: int
    current_run: int
    status: ExperimentStatus
    runs: list[RunStatus] = Field(default_factory=list)
    mlflow_experiment_id: Optional[str] = None
