"""Run the Posidonius Experiment Dashboard web server."""

import argparse
from pathlib import Path

import uvicorn
from posidonius.app import create_app


def main() -> None:
    """Start the experiment dashboard server."""
    parser = argparse.ArgumentParser(
        description="Posidonius Experiment Dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Port to listen on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to",
    )
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
    args = parser.parse_args()

    templates_dir = (
        Path(args.templates_dir) if args.templates_dir else None
    )

    app = create_app(
        templates_dir=templates_dir,
        experiments_dir=Path(args.experiments_dir),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
