"""Pipeline event logger for debugging experiment lifecycle.

Writes structured events to a JSONL file so you can see exactly what
Posidonius did, when, and why — without digging through multiple logs.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PipelineEventLog:
    """Append-only event log for a pipeline.

    Parameters
    ----------
    base_dir : Path
        Experiment base directory. Log written to
        ``{base_dir}/pipeline_events.jsonl``.
    pipeline_name : str
        Name of the pipeline for log context.
    """

    def __init__(self, base_dir: Path, pipeline_name: str) -> None:
        self.log_file = base_dir / "pipeline_events.jsonl"
        self.pipeline_name = pipeline_name
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event: str,
        detail: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Append an event to the log.

        Parameters
        ----------
        event : str
            Event name (e.g. "RUN_STARTED", "COMPLETION_DETECTED").
        detail : str, optional
            Human-readable description.
        **kwargs
            Additional structured data.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": self.pipeline_name,
            "event": event,
        }
        if detail:
            entry["detail"] = detail
        entry.update(kwargs)

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write event log: {e}")

        # Also log to Python logger for console visibility
        logger.info(f"[{event}] {detail or ''}")
