"""Per-pane interactive terminal for tmux.

Uses tmux capture-pane for reading and tmux send-keys for writing,
giving true per-pane isolation. Each WebSocket gets its own pane view.
"""

import asyncio
import subprocess  # nosec B404


class TmuxTerminalSession:
    """Interactive terminal session for a single tmux pane.

    Uses tmux commands directly instead of pty attach, so each
    WebSocket connection is isolated to its own pane.

    Parameters
    ----------
    pane_target : str
        Tmux pane target (e.g., 'marcus_test:0.1').
    """

    def __init__(self, pane_target: str) -> None:
        self.pane_target = pane_target
        self._last_line_count: int = 0
        self._alive: bool = True
        self._rows: int = 24

    def start(self, rows: int = 24, cols: int = 80) -> None:
        """Initialize the session.

        Parameters
        ----------
        rows : int
            Terminal rows (used for capture window).
        cols : int
            Terminal columns.
        """
        self._rows = rows
        self._alive = True

    def resize(self, rows: int, cols: int) -> None:
        """Update capture size on resize.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        """
        self._rows = rows

    def write(self, data: bytes) -> None:
        """Send input to the tmux pane via send-keys.

        Parameters
        ----------
        data : bytes
            Input data from the user.
        """
        try:
            text = data.decode("utf-8", errors="replace")
            # send-keys with -l (literal) to avoid key name interpretation
            subprocess.run(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    self.pane_target,
                    "-l",
                    text,
                ],
                capture_output=True,
                timeout=2,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    def send_key(self, key: str) -> None:
        """Send a special key to the pane (Enter, Escape, etc).

        Parameters
        ----------
        key : str
            Tmux key name (e.g., 'Enter', 'Escape', 'C-c').
        """
        try:
            subprocess.run(
                ["tmux", "send-keys", "-t", self.pane_target, key],
                capture_output=True,
                timeout=2,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass

    def read(self, size: int = 4096) -> bytes:
        """Capture current pane content.

        Returns the full visible pane content each time.
        The frontend xterm.js handles rendering.

        Parameters
        ----------
        size : int
            Unused, kept for interface compatibility.

        Returns
        -------
        bytes
            Captured pane content.
        """
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "capture-pane",
                    "-t",
                    self.pane_target,
                    "-p",
                    "-e",  # Include escape sequences (colors)
                    "-S",
                    f"-{self._rows}",
                ],
                capture_output=True,
                timeout=2,
            )
            return result.stdout
        except (subprocess.TimeoutExpired, OSError):
            return b""

    async def read_async(self, size: int = 4096) -> bytes:
        """Async read from the pane.

        Parameters
        ----------
        size : int
            Unused, kept for interface compatibility.

        Returns
        -------
        bytes
            Captured pane content.
        """
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.read(size)),
                timeout=2.0,
            )
        except (asyncio.TimeoutError, OSError):
            return b""

    def stop(self) -> None:
        """Mark session as stopped."""
        self._alive = False

    @property
    def is_alive(self) -> bool:
        """Check if the session is still active."""
        if not self._alive:
            return False
        # Check if the pane still exists
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    self.pane_target.split(":")[0],
                ],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
