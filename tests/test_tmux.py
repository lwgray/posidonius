"""Unit tests for tmux session management and output capture."""

import subprocess
from unittest.mock import MagicMock, Mock, patch

from posidonius.engine.tmux import TmuxManager


class TestTmuxManager:
    """Test suite for TmuxManager."""

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_capture_pane_output(self, mock_run: Mock) -> None:
        """Test capturing output from a tmux pane."""
        mock_run.return_value = MagicMock(
            stdout="Agent working on task-123...\nProgress: 50%\n",
            returncode=0,
        )
        mgr = TmuxManager()
        output = mgr.capture_pane("marcus_test:0.0")
        mock_run.assert_called_once_with(
            [
                "tmux",
                "capture-pane",
                "-t",
                "marcus_test:0.0",
                "-p",
                "-S",
                "-50",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert "Progress: 50%" in output

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_capture_pane_handles_error(self, mock_run: Mock) -> None:
        """Test capture returns empty string on error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "tmux")
        mgr = TmuxManager()
        output = mgr.capture_pane("nonexistent:0.0")
        assert output == ""

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_capture_pane_handles_timeout(self, mock_run: Mock) -> None:
        """Test capture returns empty string on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("tmux", 5)
        mgr = TmuxManager()
        output = mgr.capture_pane("marcus_test:0.0")
        assert output == ""

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_list_panes(self, mock_run: Mock) -> None:
        """Test listing all panes in a session."""
        mock_run.return_value = MagicMock(
            stdout=(
                "marcus_test:0.0||Project Creator||bash\n"
                "marcus_test:0.1||Unicorn Developer 1||bash\n"
                "marcus_test:0.2||Unicorn Developer 2||bash\n"
                "marcus_test:0.3||Monitor||bash\n"
            ),
            returncode=0,
        )
        mgr = TmuxManager()
        panes = mgr.list_panes("marcus_test")
        assert len(panes) == 4
        assert panes[0]["target"] == "marcus_test:0.0"
        assert panes[0]["title"] == "Project Creator"

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_list_panes_empty_session(self, mock_run: Mock) -> None:
        """Test listing panes for non-existent session."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "tmux")
        mgr = TmuxManager()
        panes = mgr.list_panes("nonexistent")
        assert panes == []

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_capture_all_panes(self, mock_run: Mock) -> None:
        """Test capturing output from all panes in a session."""
        list_result = MagicMock(
            stdout=(
                "marcus_test:0.0||Creator||bash\n" "marcus_test:0.1||Worker 1||bash\n"
            ),
            returncode=0,
        )
        capture_1 = MagicMock(stdout="Creating project...\n", returncode=0)
        capture_2 = MagicMock(stdout="Working on task-456...\n", returncode=0)
        mock_run.side_effect = [list_result, capture_1, capture_2]

        mgr = TmuxManager()
        all_output = mgr.capture_all_panes("marcus_test")
        assert len(all_output) == 2
        assert all_output[0]["title"] == "Creator"
        assert "Creating project" in all_output[0]["output"]
        assert all_output[0]["status"] == "working"
        assert "Working on task" in all_output[1]["output"]

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_session_exists_true(self, mock_run: Mock) -> None:
        """Test checking if a tmux session exists."""
        mock_run.return_value = MagicMock(returncode=0)
        mgr = TmuxManager()
        assert mgr.session_exists("marcus_test") is True

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_session_exists_false(self, mock_run: Mock) -> None:
        """Test session_exists returns False when no session."""
        mock_run.return_value = MagicMock(returncode=1)
        mgr = TmuxManager()
        assert mgr.session_exists("nonexistent") is False

    @patch("posidonius.engine.tmux.subprocess.run")
    def test_kill_session(self, mock_run: Mock) -> None:
        """Test killing a tmux session."""
        mock_run.return_value = MagicMock(returncode=0)
        mgr = TmuxManager()
        mgr.kill_session("marcus_test")
        mock_run.assert_called_once_with(
            ["tmux", "kill-session", "-t", "marcus_test"],
            capture_output=True,
        )

    def test_detect_agent_status_working(self) -> None:
        """Test detecting agent is actively working."""
        mgr = TmuxManager()
        assert mgr.detect_agent_status("Writing file: src/main.py\n") == "working"

    def test_detect_agent_status_waiting(self) -> None:
        """Test detecting agent is waiting."""
        mgr = TmuxManager()
        assert mgr.detect_agent_status("Waiting for project creation...\n") == "waiting"

    def test_detect_agent_status_complete(self) -> None:
        """Test detecting agent has completed."""
        mgr = TmuxManager()
        assert mgr.detect_agent_status("Work Complete\n==========\n") == "complete"

    def test_detect_agent_status_idle(self) -> None:
        """Test detecting agent is idle."""
        mgr = TmuxManager()
        assert mgr.detect_agent_status("") == "idle"

    def test_detect_agent_status_error(self) -> None:
        """Test detecting agent error state."""
        mgr = TmuxManager()
        assert mgr.detect_agent_status("Error: connection failed\n") == "error"
