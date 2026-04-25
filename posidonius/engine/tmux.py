"""Tmux session management and live output capture.

Provides real tmux session lifecycle management including pane listing,
output capture, agent status detection, and clean teardown.
"""

import subprocess  # nosec B404
import time
from typing import Any


class TmuxManager:
    """Manages tmux sessions and captures live pane output.

    Used by the web UI to show real-time agent activity without
    requiring manual tmux attachment.
    """

    def __init__(self) -> None:
        # Role cache: pane target → detected role.
        # Roles are permanent (creator/monitor/worker never changes mid-run),
        # so we cache the first definitive detection. This prevents
        # "EXPERIMENT MONITOR" from being missed once the startup header
        # scrolls past the capture_pane tail buffer.
        self._pane_roles: dict[str, str] = {}

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

    def capture_all_panes(self, session_name: str) -> list[dict[str, Any]]:
        """Capture output from all panes in a session.

        Parameters
        ----------
        session_name : str
            Tmux session name.

        Returns
        -------
        list[dict[str, Any]]
            List of dicts with 'target', 'title', 'output', 'status', and 'role'.
        """
        panes = self.list_panes(session_name)
        results: list[dict[str, Any]] = []
        for pane in panes:
            output = self.capture_pane(pane["target"])
            target = pane["target"]

            # Use cached role if already definitively identified.
            # The startup header ("EXPERIMENT MONITOR", "PROJECT CREATOR AGENT")
            # scrolls past the 50-line tail buffer after the experiment runs
            # for a while. Once we detect a non-worker role, cache it
            # permanently — roles never change mid-run.
            cached = self._pane_roles.get(target)
            if cached in ("creator", "monitor"):
                role = cached
            else:
                role = self.detect_agent_role(output)
                if role in ("creator", "monitor"):
                    self._pane_roles[target] = role

            results.append(
                {
                    "target": target,
                    "title": pane["title"],
                    "output": output,
                    "status": self.detect_agent_status(output),
                    "role": role,
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

    def confirm_trust_if_prompted(
        self,
        pane_target: str,
        timeout: float = 5.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """Poll a tmux pane and auto-confirm Claude trust/permission dialogs.

        Claude Code can pause on a directory trust prompt or a
        --dangerously-skip-permissions confirmation dialog when launched
        in a fresh directory. This detects those screens and sends the
        appropriate keystrokes to proceed.

        Parameters
        ----------
        pane_target : str
            Tmux pane target (e.g. ``session:window.pane``).
        timeout : float
            Maximum seconds to poll before giving up.
        poll_interval : float
            Seconds between polls.

        Returns
        -------
        bool
            True if a prompt was detected and confirmed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.capture_pane(pane_target, lines=30).lower()

            # Trust prompt: "Do you trust this folder?"
            if ("trust this folder" in text or "trust the contents" in text) and (
                "enter to confirm" in text
                or "press enter" in text
                or "enter to continue" in text
            ):
                subprocess.run(
                    ["tmux", "send-keys", "-t", pane_target, "Enter"],
                    capture_output=True,
                )
                time.sleep(0.5)
                return True

            # --dangerously-skip-permissions confirmation dialog
            if "yes, i accept" in text and (
                "dangerously-skip-permissions" in text
                or "skip permissions" in text
                or "permission" in text
                or "approval" in text
            ):
                subprocess.run(
                    [
                        "tmux",
                        "send-keys",
                        "-t",
                        pane_target,
                        "-l",
                        "\x1b[B",
                    ],
                    capture_output=True,
                )
                time.sleep(0.2)
                subprocess.run(
                    ["tmux", "send-keys", "-t", pane_target, "Enter"],
                    capture_output=True,
                )
                time.sleep(0.5)
                return True

            # Early exit: trust/permission prompts appear immediately
            # on startup and dominate the pane. If the pane has
            # substantial content without trust-related keywords,
            # Claude has started normally and no prompt is coming.
            if (
                len(text) > 200
                and "trust" not in text
                and "permission" not in text
                and "approval" not in text
            ):
                return False

            time.sleep(poll_interval)

        return False

    def auto_confirm_trust(self, session_name: str, timeout: float = 5.0) -> int:
        """Auto-confirm trust prompts on all panes in a session.

        Parameters
        ----------
        session_name : str
            Tmux session name.
        timeout : float
            Per-pane polling timeout in seconds.

        Returns
        -------
        int
            Number of panes where a prompt was confirmed.
        """
        panes = self.list_panes(session_name)
        confirmed = 0
        for pane in panes:
            if self.confirm_trust_if_prompted(pane["target"], timeout=timeout):
                confirmed += 1
        return confirmed

    def detect_agent_role(self, output: str) -> str:
        """Detect agent role from pane output content.

        Reads the script header echoed at startup by spawn_agents.py to
        determine whether the pane is a creator, monitor, or worker.
        This is more reliable than pane title since Claude's TUI may
        reset the terminal title.

        Parameters
        ----------
        output : str
            Captured pane output.

        Returns
        -------
        str
            One of: 'creator', 'monitor', 'worker'.
        """
        # spawn_agents.py echoes these headers near the start of each script
        head = output[:1000]
        if "PROJECT CREATOR AGENT" in head:
            return "creator"
        if "EXPERIMENT MONITOR" in head:
            return "monitor"
        return "worker"

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

        last_chunk = output[-600:] if len(output) > 600 else output
        lower = last_chunk.lower()

        # Specific terminal markers from spawn_agents.py completion echoes
        if (
            "work complete" in lower
            or "creator complete" in lower
            or "monitor - complete" in lower
        ):
            return "complete"

        # Error signals
        if any(
            kw in lower
            for kw in ["error:", "traceback (most recent", "exit code 1", "failed:"]
        ):
            return "error"

        # Claude interactive-mode activity indicators
        if "esc to interrupt" in lower or "tokens used" in lower:
            return "working"

        # Shell-level waiting signals
        if "waiting for project" in lower or "waiting for" in lower:
            return "waiting"

        # Generic busy indicators (less reliable, checked after specific ones)
        if any(
            kw in lower
            for kw in ["writing", "creating", "running", "updating", "committing"]
        ):
            return "working"

        return "idle"
