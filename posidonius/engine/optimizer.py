"""Optimal agent count pre-flight estimator.

Connects to Marcus MCP via HTTP (httpx) to create a project, run CPM
analysis on the real task dependency graph, and return optimal agent
recommendations. Requires a running Marcus server — no heuristic fallback.
"""

import asyncio
import concurrent.futures
import json
from typing import Any

import httpx
from posidonius.models import ExperimentRunConfig, OptimalAgentResponse


class OptimalAgentOptimizer:
    """Estimates optimal agent count via Marcus CPM analysis.

    Connects to a running Marcus HTTP server to create the project,
    analyze the task dependency graph, and calculate optimal parallelism.

    Parameters
    ----------
    marcus_url : str
        Marcus MCP HTTP endpoint URL.
    """

    def __init__(
        self, marcus_url: str = "http://localhost:4298/mcp"
    ) -> None:
        self.marcus_url = marcus_url

    async def _call_mcp_tool(
        self,
        client: httpx.AsyncClient,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a Marcus MCP tool via HTTP.

        Parameters
        ----------
        client : httpx.AsyncClient
            HTTP client.
        tool_name : str
            MCP tool name.
        arguments : dict[str, Any]
            Tool arguments.

        Returns
        -------
        dict[str, Any]
            Tool response content.
        """
        response = await client.post(
            self.marcus_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(
                f"MCP error: {result['error']}"
            )
        return result.get("result", {})  # type: ignore[no-any-return]

    async def analyze_with_marcus(
        self,
        project_name: str,
        project_spec: str,
        complexity: str = "standard",
    ) -> dict[str, Any]:
        """Connect to Marcus and run real CPM analysis.

        Creates the project in Marcus (which generates tasks +
        dependencies), then calls get_optimal_agent_count for real
        dependency graph analysis.

        Parameters
        ----------
        project_name : str
            Name for the project to create.
        project_spec : str
            Full project specification text.
        complexity : str
            Project complexity level.

        Returns
        -------
        dict[str, Any]
            Raw MCP response with optimal_agents, max_parallelism, etc.

        Raises
        ------
        ConnectionError
            If Marcus server is unavailable.
        RuntimeError
            If project creation or analysis fails.
        """
        async with httpx.AsyncClient() as client:
            # Authenticate as admin
            await self._call_mcp_tool(
                client,
                "authenticate",
                {
                    "client_id": "experiment-optimizer",
                    "client_type": "admin",
                    "role": "admin",
                    "metadata": {"source": "experiment_dashboard"},
                },
            )

            # Create the project (generates tasks + dependencies)
            await self._call_mcp_tool(
                client,
                "create_project",
                {
                    "project_name": (
                        f"{project_name} (optimization)"
                    ),
                    "description": project_spec,
                    "options": {
                        "complexity": complexity,
                        "provider": "planka",
                        "mode": "new_project",
                    },
                },
            )

            # Run CPM analysis on the real dependency graph
            result = await self._call_mcp_tool(
                client,
                "get_optimal_agent_count",
                {"include_details": True},
            )

            if result and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text", "{}")
                    parsed: dict[str, Any] = json.loads(text)
                    return parsed

            if result:
                return result

            raise RuntimeError(
                "Empty response from get_optimal_agent_count"
            )

    def analyze_sync(
        self,
        project_name: str,
        project_spec: str,
        complexity: str = "standard",
    ) -> OptimalAgentResponse:
        """Run Marcus CPM analysis synchronously.

        Parameters
        ----------
        project_name : str
            Name for the project.
        project_spec : str
            Project specification text.
        complexity : str
            Complexity level.

        Returns
        -------
        OptimalAgentResponse
            Optimal agent recommendation from real CPM analysis.

        Raises
        ------
        ConnectionError
            If Marcus server is unavailable.
        """
        with concurrent.futures.ThreadPoolExecutor() as pool:
            mcp_result = pool.submit(
                lambda: asyncio.run(
                    self.analyze_with_marcus(
                        project_name, project_spec, complexity
                    )
                )
            ).result(timeout=120)

        return self.parse_mcp_response(mcp_result)

    def build_recommended_runs(
        self, optimal_agents: int
    ) -> list[ExperimentRunConfig]:
        """Generate recommended pipeline runs from an optimal agent count.

        Creates three runs: ~50% of optimal, optimal, and ~200% of optimal
        to test scaling behavior.

        Parameters
        ----------
        optimal_agents : int
            The recommended optimal agent count.

        Returns
        -------
        list[ExperimentRunConfig]
            Three run configurations for scaling analysis.
        """
        safe_optimal = max(1, optimal_agents)
        half = max(1, safe_optimal // 2)
        double = max(2, safe_optimal * 2)
        return [
            ExperimentRunConfig(num_agents=half),
            ExperimentRunConfig(num_agents=safe_optimal),
            ExperimentRunConfig(num_agents=double),
        ]

    def parse_mcp_response(
        self, mcp_response: dict[str, Any]
    ) -> OptimalAgentResponse:
        """Parse the MCP get_optimal_agent_count response.

        Parameters
        ----------
        mcp_response : dict[str, Any]
            Raw response from the MCP tool.

        Returns
        -------
        OptimalAgentResponse
            Parsed and enriched response with recommended runs.
        """
        optimal = mcp_response["optimal_agents"]
        recommended_runs = self.build_recommended_runs(optimal)

        return OptimalAgentResponse(
            optimal_agents=optimal,
            max_parallelism=mcp_response.get(
                "max_parallelism", 0
            ),
            total_tasks=mcp_response.get("total_tasks", 0),
            critical_path_hours=mcp_response.get(
                "critical_path_hours", 0.0
            ),
            efficiency_gain_percent=mcp_response.get(
                "efficiency_gain_percent", 0.0
            ),
            recommended_runs=recommended_runs,
        )

    def create_scaling_runs(
        self,
        agent_counts: list[int],
        subagents_per_agent: int = 0,
    ) -> list[ExperimentRunConfig]:
        """Create custom scaling run configurations.

        Parameters
        ----------
        agent_counts : list[int]
            List of agent counts to test.
        subagents_per_agent : int
            Number of subagents per agent for all runs.

        Returns
        -------
        list[ExperimentRunConfig]
            Run configurations for each agent count.
        """
        return [
            ExperimentRunConfig(
                num_agents=count,
                subagents_per_agent=subagents_per_agent,
            )
            for count in agent_counts
        ]
