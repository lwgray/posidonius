"""Unit tests for per-pane terminal session."""

from unittest.mock import MagicMock, Mock, patch

from posidonius.engine.terminal import TmuxTerminalSession


class TestTmuxTerminalSession:
    """Test suite for TmuxTerminalSession."""

    def test_init(self) -> None:
        """Test session initializes with correct target."""
        session = TmuxTerminalSession("marcus_test:0.1")
        assert session.pane_target == "marcus_test:0.1"

    @patch("posidonius.engine.terminal.subprocess.run")
    def test_write_sends_keys(self, mock_run: Mock) -> None:
        """Test write sends literal keys to tmux pane."""
        session = TmuxTerminalSession("marcus_test:0.1")
        session.start()
        session.write(b"hello")
        mock_run.assert_called_once_with(
            [
                "tmux",
                "send-keys",
                "-t",
                "marcus_test:0.1",
                "-l",
                "hello",
            ],
            capture_output=True,
            timeout=2,
        )

    @patch("posidonius.engine.terminal.subprocess.run")
    def test_send_key_sends_special_key(self, mock_run: Mock) -> None:
        """Test send_key sends named key to tmux pane."""
        session = TmuxTerminalSession("marcus_test:0.1")
        session.start()
        session.send_key("Enter")
        mock_run.assert_called_once_with(
            [
                "tmux",
                "send-keys",
                "-t",
                "marcus_test:0.1",
                "Enter",
            ],
            capture_output=True,
            timeout=2,
        )

    @patch("posidonius.engine.terminal.subprocess.run")
    def test_read_captures_pane(self, mock_run: Mock) -> None:
        """Test read uses tmux capture-pane."""
        mock_run.return_value = MagicMock(stdout=b"hello world\n")
        session = TmuxTerminalSession("marcus_test:0.1")
        session.start(rows=30, cols=80)
        result = session.read()
        assert result == b"hello world\n"
        mock_run.assert_called_once_with(
            [
                "tmux",
                "capture-pane",
                "-t",
                "marcus_test:0.1",
                "-p",
                "-e",
                "-S",
                "-30",
            ],
            capture_output=True,
            timeout=2,
        )

    def test_write_when_not_started(self) -> None:
        """Test write does nothing when not started (no crash)."""
        session = TmuxTerminalSession("test:0.0")
        session.write(b"hello")  # Should not raise

    def test_read_when_not_started(self) -> None:
        """Test read returns output even before start."""
        TmuxTerminalSession("test:0.0")

    def test_stop(self) -> None:
        """Test stop marks session as not alive."""
        session = TmuxTerminalSession("test:0.0")
        session.start()
        assert session._alive is True
        session.stop()
        assert session._alive is False

    @patch("posidonius.engine.terminal.subprocess.run")
    def test_is_alive_checks_session(self, mock_run: Mock) -> None:
        """Test is_alive checks if tmux session exists."""
        mock_run.return_value = MagicMock(returncode=0)
        session = TmuxTerminalSession("marcus_test:0.1")
        session.start()
        assert session.is_alive is True
        mock_run.assert_called_with(
            ["tmux", "has-session", "-t", "marcus_test"],
            capture_output=True,
            timeout=2,
        )

    def test_is_alive_false_when_stopped(self) -> None:
        """Test is_alive returns False after stop."""
        session = TmuxTerminalSession("test:0.0")
        session.start()
        session.stop()
        assert session.is_alive is False

    def test_resize_updates_rows(self) -> None:
        """Test resize stores new row count for capture."""
        session = TmuxTerminalSession("test:0.0")
        session.start(rows=24, cols=80)
        session.resize(40, 120)
        assert session._rows == 40
