"""MLflow tracking integration for experiment pipelines.

Manages parent/child MLflow runs for sequential experiment pipelines,
tracking metrics like completion time, task counts, and blockers
across agent configurations.

Uses MlflowClient directly (not the fluent API) to avoid the
process-level global active-run singleton, which breaks when multiple
pipelines start concurrently in the same process.
"""

import time
from typing import Optional

import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from posidonius.models import PipelineConfig


class MLflowTracker:
    """Tracks experiment pipeline runs in MLflow.

    Creates a parent MLflow experiment for the pipeline with child runs
    for each agent configuration, enabling comparison across scaling tests.

    Uses ``MlflowClient`` directly to avoid the fluent-API global active-run
    state, which causes ``start_run`` to fail when multiple pipelines are
    launched in the same process simultaneously.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.client = MlflowClient()
        # get_or_create experiment without touching fluent global state
        experiment = self.client.get_experiment_by_name(config.name)
        if experiment is None:
            experiment_id = self.client.create_experiment(config.name)
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id: str = experiment_id
        self.parent_run_id: Optional[str] = None
        self._active_child_run_id: Optional[str] = None

    def start_pipeline_run(self) -> str:
        """Start the parent MLflow run for the pipeline.

        Returns
        -------
        str
            MLflow run ID for the parent pipeline run.
        """
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=f"{self.config.name}_pipeline",
        )
        self.parent_run_id = run.info.run_id
        self.client.log_batch(
            self.parent_run_id,
            params=[
                Param("pipeline_name", self.config.name),
                Param("project_name", self.config.project_name),
                Param("complexity", str(self.config.complexity)),
                Param("total_runs", str(len(self.config.runs))),
                Param(
                    "run_agent_counts",
                    str([r.num_agents for r in self.config.runs]),
                ),
            ],
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
        tags = {}
        if self.parent_run_id:
            tags["mlflow.parentRunId"] = self.parent_run_id
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=f"run_{run_index}_{num_agents}_agents",
            tags=tags,
        )
        self._active_child_run_id = run.info.run_id
        self.client.log_batch(
            self._active_child_run_id,
            params=[
                Param("run_index", str(run_index)),
                Param("num_agents", str(num_agents)),
                Param("subagents_per_agent", str(subagents_per_agent)),
                Param(
                    "total_workers",
                    str(num_agents + (num_agents * subagents_per_agent)),
                ),
            ],
        )
        run_id: str = self._active_child_run_id
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
        if not self._active_child_run_id:
            return
        completion_rate = tasks_completed / tasks_total if tasks_total > 0 else 0.0
        ts = int(time.time() * 1000)
        self.client.log_batch(
            self._active_child_run_id,
            metrics=[
                Metric("completion_time_seconds", completion_time_seconds, ts, 0),
                Metric("tasks_completed", tasks_completed, ts, 0),
                Metric("tasks_total", tasks_total, ts, 0),
                Metric("blockers", blockers, ts, 0),
                Metric("completion_rate", completion_rate, ts, 0),
            ],
        )

    def end_child_run(self, status: str = "FINISHED") -> None:
        """End the current child run.

        Parameters
        ----------
        status : str
            MLflow run status (FINISHED, FAILED, KILLED).
        """
        if self._active_child_run_id:
            self.client.set_terminated(self._active_child_run_id, status=status)
            self._active_child_run_id = None

    def end_pipeline_run(self, status: str = "FINISHED") -> None:
        """End the parent pipeline run.

        Parameters
        ----------
        status : str
            MLflow run status.
        """
        if self.parent_run_id:
            self.client.set_terminated(self.parent_run_id, status=status)
