import sys
from pathlib import Path
from collections import Counter

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


from observability.metrics_aggregator import (
    MetricsAggregator
)

agg = MetricsAggregator()

agg.generate_report()