"""Run the Posidonius Experiment Dashboard web server."""

import argparse
import logging
import logging.handlers
from pathlib import Path

import uvicorn

from posidonius.app import create_app


def _setup_logging(experiments_dir: Path) -> None:
    """Configure root logger with a rotating file handler.

    All Python logging (Posidonius + libraries) goes to
    ``{experiments_dir}/logs/posidonius.log``.  WARNING+ also echoes to
    the console so the terminal isn't completely silent.

    Parameters
    ----------
    experiments_dir : Path
        Base directory for experiment output; ``logs/`` subdir is
        created if absent.
    """
    log_dir = experiments_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "posidonius.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file: 10 MB per file, keep 5 → up to 50 MB total
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console: WARNING+ only — don't drown uvicorn's own output
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    # Pull uvicorn access + error logs into the same file
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = True

    logging.getLogger(__name__).info("Logging to %s", log_file)


def main() -> None:
    """Start the experiment dashboard server."""
    parser = argparse.ArgumentParser(description="Posidonius Experiment Dashboard")
    parser.add_argument("--port", type=int, default=8420, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=str(Path.home() / "experiments"),
        help="Base directory for experiment output",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=None,
        help="Path to Marcus experiment templates directory",
    )
    parser.add_argument(
        "--marcus-python",
        type=str,
        default=None,
        help=(
            "Python interpreter for Marcus subprocesses "
            "(default: current interpreter). "
            "Required when Marcus is installed in a different environment. "
            "Examples: "
            "conda: $(conda run -n <env> which python), "
            "venv: /path/to/venv/bin/python, "
            "pyenv: $(pyenv which python)"
        ),
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    _setup_logging(experiments_dir)

    templates_dir = Path(args.templates_dir) if args.templates_dir else None
    app = create_app(
        templates_dir=templates_dir,
        experiments_dir=experiments_dir,
        marcus_python=args.marcus_python,
    )

    # Disable uvicorn's default log config so our handlers take effect
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
