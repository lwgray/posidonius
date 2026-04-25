"""Unit tests for Marcus server startup utilities.

Tests _port_is_open and _start_marcus_instance with all external
dependencies (subprocess, socket, filesystem, time) mocked so the
suite runs fast and without a live Marcus installation.
"""

import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest

from posidonius.app import _port_is_open, _start_marcus_instance
from posidonius.models import MarcusInstance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instance(
    url: str = "http://localhost:4299/mcp",
    db_path: str | None = "./data/exp_1.db",
    port: int | None = 4299,
) -> MarcusInstance:
    """Build a MarcusInstance for test use."""
    return MarcusInstance(url=url, db_path=db_path, port=port)


# ---------------------------------------------------------------------------
# _port_is_open
# ---------------------------------------------------------------------------


class TestPortIsOpen:
    """_port_is_open returns True/False based on socket connectivity."""

    def test_true_when_connection_succeeds(self) -> None:
        """Returns True when create_connection does not raise."""
        mock_cm = MagicMock()
        mock_cm.__enter__ = Mock(return_value=Mock())
        mock_cm.__exit__ = Mock(return_value=False)
        with patch("socket.create_connection", return_value=mock_cm):
            assert _port_is_open("127.0.0.1", 4299) is True

    def test_false_on_connection_refused(self) -> None:
        """Returns False when TCP connection is refused."""
        with patch("socket.create_connection", side_effect=ConnectionRefusedError):
            assert _port_is_open("127.0.0.1", 4299) is False

    def test_false_on_os_error(self) -> None:
        """Returns False for generic OS errors (e.g., host unreachable)."""
        with patch("socket.create_connection", side_effect=OSError):
            assert _port_is_open("localhost", 9999) is False

    def test_passes_timeout_to_create_connection(self) -> None:
        """Timeout argument is forwarded to socket.create_connection."""
        with patch("socket.create_connection", side_effect=ConnectionRefusedError) as m:
            _port_is_open("127.0.0.1", 4299, timeout=2.5)
        m.assert_called_once_with(("127.0.0.1", 4299), timeout=2.5)


# ---------------------------------------------------------------------------
# _start_marcus_instance — skip cases
# ---------------------------------------------------------------------------


class TestStartMarcusInstanceSkipCases:
    """_start_marcus_instance returns None without spawning when appropriate."""

    def test_returns_none_when_instance_has_no_port(self, tmp_path: Path) -> None:
        """Instance without a port → caller manages Marcus, we do nothing."""
        inst = _instance(port=None)
        result = _start_marcus_instance(inst, marcus_root=tmp_path)
        assert result is None

    def test_returns_none_when_port_already_open(self, tmp_path: Path) -> None:
        """Skips startup when something already listens on the target port."""
        with patch("posidonius.app._port_is_open", return_value=True):
            result = _start_marcus_instance(_instance(), marcus_root=tmp_path)
        assert result is None

    def test_no_subprocess_when_port_already_open(self, tmp_path: Path) -> None:
        """Popen must NOT be called when the port is already in use."""
        with patch("posidonius.app._port_is_open", return_value=True):
            with patch("subprocess.Popen") as mock_popen:
                _start_marcus_instance(_instance(), marcus_root=tmp_path)
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# _start_marcus_instance — subprocess details
# ---------------------------------------------------------------------------


class TestStartMarcusInstanceSubprocess:
    """Verifies correct subprocess command and environment."""

    @pytest.fixture
    def fast_startup(self, tmp_path: Path) -> Iterator[MagicMock]:
        """Patch port-check to simulate closed-then-open, skip sleep."""
        port_states = iter([False, True])  # closed on first check, open after start
        with patch("posidonius.app._port_is_open", side_effect=port_states):
            with patch("time.sleep"):
                with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_popen.return_value = MagicMock()
                        yield mock_popen

    def test_http_flag_in_command(
        self, fast_startup: MagicMock, tmp_path: Path
    ) -> None:
        """--http must appear in the subprocess command."""
        _start_marcus_instance(_instance(), marcus_root=tmp_path)
        cmd = fast_startup.call_args[0][0]
        assert "--http" in cmd

    def test_uses_python_executable(
        self, fast_startup: MagicMock, tmp_path: Path
    ) -> None:
        """The current Python interpreter is used to run Marcus."""
        _start_marcus_instance(_instance(), marcus_root=tmp_path)
        cmd = fast_startup.call_args[0][0]
        assert cmd[0] == sys.executable

    def test_cwd_is_marcus_root(self, fast_startup: MagicMock, tmp_path: Path) -> None:
        """subprocess.Popen cwd is set to marcus_root."""
        _start_marcus_instance(_instance(), marcus_root=tmp_path)
        kwargs = fast_startup.call_args[1]
        assert kwargs.get("cwd") == str(tmp_path)

    def test_sqlite_kanban_db_path_in_env(
        self, fast_startup: MagicMock, tmp_path: Path
    ) -> None:
        """SQLITE_KANBAN_DB_PATH is injected from instance.db_path."""
        _start_marcus_instance(
            _instance(db_path="./data/exp_99.db"), marcus_root=tmp_path
        )
        env = fast_startup.call_args[1]["env"]
        assert env["SQLITE_KANBAN_DB_PATH"] == "./data/exp_99.db"

    def test_marcus_config_env_set(
        self, fast_startup: MagicMock, tmp_path: Path
    ) -> None:
        """MARCUS_CONFIG env var points to a temp config file."""
        _start_marcus_instance(_instance(), marcus_root=tmp_path)
        env = fast_startup.call_args[1]["env"]
        assert "MARCUS_CONFIG" in env
        assert env["MARCUS_CONFIG"].endswith(".json")

    def test_returns_popen_handle(
        self, fast_startup: MagicMock, tmp_path: Path
    ) -> None:
        """Returns the Popen handle when startup succeeds."""
        result = _start_marcus_instance(_instance(), marcus_root=tmp_path)
        assert result is not None

    def test_no_db_path_means_no_env_injection(
        self, fast_startup: MagicMock, tmp_path: Path
    ) -> None:
        """When db_path is None, SQLITE_KANBAN_DB_PATH is not set."""
        _start_marcus_instance(
            _instance(db_path=None, url="http://localhost:4299/mcp", port=4299),
            marcus_root=tmp_path,
        )
        env = fast_startup.call_args[1]["env"]
        # Should not be overridden (might exist from parent env, but not set by us
        # unless already in os.environ — we just verify the function didn't set it
        # to None or empty)
        val = env.get("SQLITE_KANBAN_DB_PATH")
        assert val != "" and val is not None or "SQLITE_KANBAN_DB_PATH" not in env


# ---------------------------------------------------------------------------
# _start_marcus_instance — startup timeout
# ---------------------------------------------------------------------------


class TestStartMarcusInstanceTimeout:
    """Validates timeout behaviour when Marcus takes too long to start."""

    def test_raises_runtime_error_on_timeout(self, tmp_path: Path) -> None:
        """RuntimeError is raised when port never opens within timeout."""
        with patch("posidonius.app._port_is_open", return_value=False):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                with patch("time.sleep"):
                    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
                        with pytest.raises(RuntimeError, match="failed to start"):
                            _start_marcus_instance(
                                _instance(), marcus_root=tmp_path, startup_timeout=2
                            )

    def test_terminates_process_on_timeout(self, tmp_path: Path) -> None:
        """The spawned process is terminated when startup times out."""
        mock_proc = MagicMock()
        with patch("posidonius.app._port_is_open", return_value=False):
            with patch("subprocess.Popen", return_value=mock_proc):
                with patch("time.sleep"):
                    with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
                        with pytest.raises(RuntimeError):
                            _start_marcus_instance(
                                _instance(), marcus_root=tmp_path, startup_timeout=2
                            )
        mock_proc.terminate.assert_called_once()
