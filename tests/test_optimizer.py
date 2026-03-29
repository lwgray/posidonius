"""Unit tests for optimal agent count pre-flight."""

from unittest.mock import Mock, patch

import pytest

from posidonius.engine.optimizer import OptimalAgentOptimizer
from posidonius.models import OptimalAgentResponse


class TestOptimalAgentOptimizer:
    """Test suite for OptimalAgentOptimizer."""

    def test_build_recommended_runs_from_optimal(self) -> None:
        """Test generating recommended runs from optimal count."""
        optimizer = OptimalAgentOptimizer()
        runs = optimizer.build_recommended_runs(optimal_agents=8)
        assert len(runs) == 3
        assert runs[0].num_agents < 8
        assert runs[1].num_agents == 8
        assert runs[2].num_agents > 8

    def test_build_recommended_runs_small_optimal(self) -> None:
        """Test recommended runs when optimal is small."""
        optimizer = OptimalAgentOptimizer()
        runs = optimizer.build_recommended_runs(optimal_agents=2)
        assert len(runs) == 3
        assert runs[0].num_agents >= 1

    def test_build_recommended_runs_large_optimal(self) -> None:
        """Test recommended runs when optimal is large."""
        optimizer = OptimalAgentOptimizer()
        runs = optimizer.build_recommended_runs(optimal_agents=20)
        assert len(runs) == 3
        assert runs[2].num_agents == 40

    def test_parse_mcp_response(self) -> None:
        """Test parsing MCP get_optimal_agent_count response."""
        optimizer = OptimalAgentOptimizer()
        mcp_response = {
            "optimal_agents": 6,
            "max_parallelism": 4,
            "total_tasks": 15,
            "critical_path_hours": 2.5,
            "efficiency_gain_percent": 75.0,
        }
        response = optimizer.parse_mcp_response(mcp_response)
        assert isinstance(response, OptimalAgentResponse)
        assert response.optimal_agents == 6
        assert len(response.recommended_runs) == 3

    def test_parse_mcp_response_missing_fields(self) -> None:
        """Test parsing response with minimal fields."""
        optimizer = OptimalAgentOptimizer()
        mcp_response = {"optimal_agents": 4}
        response = optimizer.parse_mcp_response(mcp_response)
        assert response.optimal_agents == 4
        assert response.max_parallelism == 0

    def test_create_scaling_runs(self) -> None:
        """Test creating custom scaling run configurations."""
        optimizer = OptimalAgentOptimizer()
        runs = optimizer.create_scaling_runs([3, 6, 12, 24])
        assert len(runs) == 4
        assert [r.num_agents for r in runs] == [3, 6, 12, 24]

    def test_create_scaling_runs_with_subagents(self) -> None:
        """Test creating scaling runs with subagents."""
        optimizer = OptimalAgentOptimizer()
        runs = optimizer.create_scaling_runs([5, 10], subagents_per_agent=2)
        assert all(r.subagents_per_agent == 2 for r in runs)

    def test_analyze_sync_raises_when_marcus_unavailable(
        self,
    ) -> None:
        """Test analyze_sync raises when Marcus is down."""
        optimizer = OptimalAgentOptimizer(marcus_url="http://localhost:99999/mcp")
        with pytest.raises(Exception):
            optimizer.analyze_sync("test", "Build something", "prototype")

    @patch("posidonius.engine.optimizer." "OptimalAgentOptimizer.analyze_with_marcus")
    def test_analyze_sync_calls_marcus(self, mock_analyze: Mock) -> None:
        """Test analyze_sync delegates to Marcus MCP."""

        async def fake_analyze(*args: object, **kwargs: object) -> dict[str, object]:
            return {
                "optimal_agents": 5,
                "max_parallelism": 3,
                "total_tasks": 12,
                "critical_path_hours": 1.5,
                "efficiency_gain_percent": 60.0,
            }

        mock_analyze.side_effect = fake_analyze
        optimizer = OptimalAgentOptimizer()
        result = optimizer.analyze_sync("test", "Build an API", "standard")
        assert result.optimal_agents == 5
        assert len(result.recommended_runs) == 3
