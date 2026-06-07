from pathlib import Path
import json
from datetime import datetime


class MetricsLogger:

    def __init__(self):

        # metrics_logger.py lives inside
        # rag-pipeline/observability/

        self.log_file = (
            Path(__file__).resolve().parent
            / "query_metrics.jsonl"
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(self, event):

        event["timestamp"] = (
            datetime.utcnow().isoformat()
        )

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(event)
                + "\n"
            )