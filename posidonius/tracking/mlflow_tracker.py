"""MLflow tracking integration for experiment pipelines.

Manages parent/child MLflow runs for sequential experiment pipelines,
tracking metrics like completion time, task counts, and blockers
across agent configurations.
"""

from typing import Optional

import mlflow

from posidonius.models import PipelineConfig


class MLflowTracker:
    """Tracks experiment pipeline runs in MLflow.

    Creates a parent MLflow experiment for the pipeline with child runs
    for each agent configuration, enabling comparison across scaling tests.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        experiment = mlflow.set_experiment(config.name)
        self.experiment_id: str = experiment.experiment_id
        self.parent_run_id: Optional[str] = None

    def start_pipeline_run(self) -> str:
        """Start the parent MLflow run for the pipeline.

        Returns
        -------
        str
            MLflow run ID for the parent pipeline run.
        """
        run = mlflow.start_run(run_name=f"{self.config.name}_pipeline")
        self.parent_run_id = run.info.run_id
        mlflow.log_params(
            {
                "pipeline_name": self.config.name,
                "project_name": self.config.project_name,
                "complexity": self.config.complexity,
                "total_runs": len(self.config.runs),
                "run_agent_counts": str([r.num_agents for r in self.config.runs]),
            }
        )
        return self.parent_run_id

    def start_child_run(
        self,
        run_index: int,
        num_agents: int,
        subagents_per_agent: int,
    ) -> str:
        """Start a child MLflow run for a specific agent configuration.

        Parameters
        ----------
        run_index : int
            Index of the run in the pipeline.
        num_agents : int
            Number of agents in this run.
        subagents_per_agent : int
            Number of subagents per agent.

        Returns
        -------
        str
            MLflow run ID for the child run.
        """
        run = mlflow.start_run(
            run_name=f"run_{run_index}_{num_agents}_agents",
            nested=True,
        )
        mlflow.log_params(
            {
                "run_index": run_index,
                "num_agents": num_agents,
                "subagents_per_agent": subagents_per_agent,
                "total_workers": num_agents + (num_agents * subagents_per_agent),
            }
        )
        run_id: str = run.info.run_id
        return run_id

    def log_run_metrics(
        self,
        completion_time_seconds: float,
        tasks_completed: int,
        tasks_total: int,
        blockers: int,
    ) -> None:
        """Log metrics for a completed run.

        Parameters
        ----------
        completion_time_seconds : float
            Time taken to complete the run in seconds.
        tasks_completed : int
            Number of tasks completed.
        tasks_total : int
            Total number of tasks.
        blockers : int
            Number of blockers encountered.
        """
        completion_rate = tasks_completed / tasks_total if tasks_total > 0 else 0.0
        mlflow.log_metrics(
            {
                "completion_time_seconds": completion_time_seconds,
                "tasks_completed": tasks_completed,
                "tasks_total": tasks_total,
                "blockers": blockers,
                "completion_rate": completion_rate,
            }
        )

    def end_child_run(self, status: str = "FINISHED") -> None:
        """End the current child run.

        Parameters
        ----------
        status : str
            MLflow run status (FINISHED, FAILED, KILLED).
        """
        mlflow.end_run(status=status)

    def end_pipeline_run(self, status: str = "FINISHED") -> None:
        """End the parent pipeline run.

        Parameters
        ----------
        status : str
            MLflow run status.
        """
        mlflow.end_run(status=status)
