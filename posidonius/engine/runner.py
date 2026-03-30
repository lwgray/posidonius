"""Experiment runner that generates configs and manages run directories.

This module handles directory creation, config generation, tmux lifecycle
management, and cleanup between sequential runs. It shells out to
run_experiment.py rather than importing Marcus internals directly.
"""

import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import yaml

from posidonius.models import AgentConfig, ExperimentRunConfig, PipelineConfig


class ExperimentRunner:
    """Manages single experiment runs with clean tmux teardown.

    Parameters
    ----------
    pipeline : PipelineConfig
        Pipeline configuration defining runs and project settings.
    templates_dir : Path
        Path to the experiment templates directory.
    base_dir : Path
        Base directory where experiment run directories are created.
    """

    def __init__(
        self,
        pipeline: PipelineConfig,
        templates_dir: Path,
        base_dir: Path,
    ) -> None:
        self.pipeline = pipeline
        self.templates_dir = templates_dir
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._session_prefix = (
            f"marcus_{pipeline.project_name.lower().replace(' ', '_')}"
        )

    def get_tmux_session_name(self, run_index: int) -> str:
        """Generate tmux session name for a specific run.

        Must match Marcus's AgentSpawner naming convention:
            f"marcus_{project_name.lower().replace(' ', '_')}"
        where project_name is the full name from config.yaml.

        Parameters
        ----------
        run_index : int
            Index of the run in the pipeline.

        Returns
        -------
        str
            Tmux session name.
        """
        run_config = self.pipeline.runs[run_index]
        project_name = (
            f"{self.pipeline.project_name}"
            f"-run_{run_index}-{run_config.num_agents}_agents"
        )
        return f"marcus_{project_name.lower().replace(' ', '_')}"

    def generate_agents(self, run_config: ExperimentRunConfig) -> list[AgentConfig]:
        """Generate agent configurations for a run.

        If run_config.agents is set, returns those directly.
        Otherwise, auto-generates agents from num_agents and pipeline
        default_skills.

        Parameters
        ----------
        run_config : ExperimentRunConfig
            Run configuration.

        Returns
        -------
        list[AgentConfig]
            List of agent configurations.
        """
        if run_config.agents:
            return run_config.agents

        agents: list[AgentConfig] = []
        for i in range(1, run_config.num_agents + 1):
            agents.append(
                AgentConfig(
                    id=f"agent_unicorn_{i}",
                    name=f"Unicorn Developer {i}",
                    role="full-stack",
                    skills=list(self.pipeline.default_skills),
                    subagents=run_config.subagents_per_agent,
                )
            )
        return agents

    def create_run_directory(self, run_index: int) -> Path:
        """Create directory structure for a single run.

        Parameters
        ----------
        run_index : int
            Index of the run in the pipeline.

        Returns
        -------
        Path
            Path to the run directory.
        """
        run_dir = self.base_dir / f"{self.pipeline.name}_run_{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "implementation").mkdir(exist_ok=True)
        (run_dir / "prompts").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        return run_dir

    def generate_config_dict(
        self,
        run_config: ExperimentRunConfig,
        agents: list[AgentConfig],
        run_index: int,
    ) -> dict[str, Any]:
        """Generate config.yaml content as a dictionary.

        Parameters
        ----------
        run_config : ExperimentRunConfig
            Run configuration.
        agents : list[AgentConfig]
            Agent configurations for this run.
        run_index : int
            Index of the run.

        Returns
        -------
        dict
            Configuration dictionary ready for YAML serialization.
        """
        return {
            "project_name": (
                f"{self.pipeline.project_name}"
                f"-run_{run_index}"
                f"-{run_config.num_agents}_agents"
            ),
            "project_spec_file": "project_spec.md",
            "project_options": {
                "complexity": self.pipeline.complexity,
                "provider": self.pipeline.provider,
                "mode": self.pipeline.mode,
            },
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "role": a.role,
                    "skills": a.skills,
                    "subagents": a.subagents,
                }
                for a in agents
            ],
            "timeouts": {
                "project_creation": self.pipeline.timeout_project_creation,
                "agent_startup": self.pipeline.timeout_agent_startup,
            },
        }

    def write_config_yaml(self, run_dir: Path, config_dict: dict[str, Any]) -> Path:
        """Write config.yaml to the run directory.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.
        config_dict : dict
            Configuration dictionary.

        Returns
        -------
        Path
            Path to the written config.yaml file.
        """
        config_file = run_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        return config_file

    def write_project_spec(self, run_dir: Path) -> Path:
        """Write project specification to the run directory.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        Path
            Path to the written spec file.
        """
        spec_file = run_dir / "project_spec.md"
        with open(spec_file, "w") as f:
            f.write(self.pipeline.project_spec)
        return spec_file

    def kill_tmux_session(self, session_name: str) -> None:
        """Kill a tmux session cleanly.

        Parameters
        ----------
        session_name : str
            Name of the tmux session to kill.
        """
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            pass

    def list_active_sessions(self) -> list[str]:
        """List active tmux sessions belonging to this pipeline.

        Returns
        -------
        list[str]
            List of active tmux session names.
        """
        try:
            result = subprocess.run(
                ["tmux", "list-sessions"],
                capture_output=True,
                text=True,
            )
            sessions: list[str] = []
            for line in result.stdout.strip().split("\n"):
                if line and line.startswith(self._session_prefix):
                    session_name = line.split(":")[0]
                    sessions.append(session_name)
            return sessions
        except subprocess.CalledProcessError:
            return []

    def cleanup_all_sessions(self) -> None:
        """Kill all tmux sessions belonging to this pipeline."""
        sessions = self.list_active_sessions()
        for session in sessions:
            self.kill_tmux_session(session)

    def prepare_run(self, run_index: int) -> Path:
        """Prepare a run directory with config and spec files.

        Parameters
        ----------
        run_index : int
            Index of the run in the pipeline.

        Returns
        -------
        Path
            Path to the prepared run directory.
        """
        run_config = self.pipeline.runs[run_index]
        agents = self.generate_agents(run_config)
        run_dir = self.create_run_directory(run_index)
        config_dict = self.generate_config_dict(run_config, agents, run_index)
        # Add project_root so Marcus validation knows where source code lives
        config_dict["project_options"]["project_root"] = str(run_dir / "implementation")
        self.write_config_yaml(run_dir, config_dict)
        self.write_project_spec(run_dir)
        return run_dir
