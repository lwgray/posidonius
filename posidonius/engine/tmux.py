"""Tmux session management and live output capture.

Provides real tmux session lifecycle management including pane listing,
output capture, agent status detection, and clean teardown.
"""

import subprocess  # nosec B404
from typing import Any


class TmuxManager:
    """Manages tmux sessions and captures live pane output.

    Used by the web UI to show real-time agent activity without
    requiring manual tmux attachment.
    """

    def capture_pane(self, target: str, lines: int = 50) -> str:
        """Capture recent output from a tmux pane.

        Parameters
        ----------
        target : str
            Tmux pane target (e.g., 'session:window.pane').
        lines : int
            Number of lines to capture from the bottom.

        Returns
        -------
        str
            Captured pane output, or empty string on error.
        """
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "capture-pane",
                    "-t",
                    target,
                    "-p",
                    "-S",
                    f"-{lines}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    def list_panes(self, session_name: str) -> list[dict[str, str]]:
        """List all panes in a tmux session with their titles.

        Parameters
        ----------
        session_name : str
            Tmux session name.

        Returns
        -------
        list[dict[str, str]]
            List of pane info dicts with 'target' and 'title' keys.
        """
        try:
            # List all panes across all windows in this session only
            result = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-s",
                    "-t",
                    session_name,
                    "-F",
                    "#{session_name}:#{window_index}.#{pane_index}"
                    "||#{pane_title}||#{pane_current_command}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
            panes: list[dict[str, str]] = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("||")
                target = parts[0]
                title = parts[1] if len(parts) > 1 and parts[1] else ""
                command = parts[2] if len(parts) > 2 else ""
                # Use title if set, otherwise use command, otherwise "Pane N"
                display = title or command or f"Pane {len(panes)}"
                panes.append({"target": target, "title": display})
            return panes
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ):
            return []

    def capture_all_panes(
        self, session_name: str
    ) -> list[dict[str, Any]]:
        """Capture output from all panes in a session.

        Parameters
        ----------
        session_name : str
            Tmux session name.

        Returns
        -------
        list[dict[str, Any]]
            List of dicts with 'target', 'title', 'output', and 'status'.
        """
        panes = self.list_panes(session_name)
        results: list[dict[str, Any]] = []
        for pane in panes:
            output = self.capture_pane(pane["target"])
            results.append(
                {
                    "target": pane["target"],
                    "title": pane["title"],
                    "output": output,
                    "status": self.detect_agent_status(output),
                }
            )
        return results

    def session_exists(self, session_name: str) -> bool:
        """Check if a tmux session exists.

        Parameters
        ----------
        session_name : str
            Tmux session name.

        Returns
        -------
        bool
            True if the session exists.
        """
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
        )
        return result.returncode == 0

    def kill_session(self, session_name: str) -> None:
        """Kill a tmux session.

        Parameters
        ----------
        session_name : str
            Tmux session name to kill.
        """
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
        )

    def detect_agent_status(self, output: str) -> str:
        """Detect agent status from pane output.

        Analyzes the last few lines of pane output to determine
        if the agent is working, waiting, complete, or idle.

        Parameters
        ----------
        output : str
            Captured pane output.

        Returns
        -------
        str
            One of: 'working', 'waiting', 'complete', 'idle', 'error'.
        """
        if not output.strip():
            return "idle"

        lower = output.lower()
        last_chunk = lower[-500:] if len(lower) > 500 else lower

        if "work complete" in last_chunk or "complete" in last_chunk:
            return "complete"
        if "error" in last_chunk or "failed" in last_chunk:
            return "error"
        if "waiting" in last_chunk or "sleep" in last_chunk:
            return "waiting"
        if any(
            kw in last_chunk
            for kw in [
                "writing",
                "working",
                "task",
                "creating",
                "running",
                "progress",
                "commit",
            ]
        ):
            return "working"
        return "idle"
