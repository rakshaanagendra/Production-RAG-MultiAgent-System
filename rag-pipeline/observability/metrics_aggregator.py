import json
from pathlib import Path
from collections import Counter


class MetricsAggregator:

    def __init__(self):

        self.log_file = (
            Path(__file__).resolve().parent
            / "query_metrics.jsonl"
        )

    def load_events(self):

        events = []

        if not self.log_file.exists():
            return events

        with open(
            self.log_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                events.append(
                    json.loads(line)
                )

        return events

    def generate_report(self):

        events = self.load_events()

        if not events:
            print("No events found.")
            return

        total_queries = len(events)

        retry_events = [
            e
            for e in events
            if e["used_retry"]
        ]

        retry_reason_counts = Counter()
        for e in retry_events:

            for reason in e.get(
                "retry_reasons",
                []
            ):

                retry_reason_counts[
                    reason
                ] += 1


        # Previous success definition answerability_delta > 0 was too strict
        # We want to allow for some noise in the metrics, so we set a threshold of 0.05
        # Retry controller is optimizing retrieval quality not just answerability
        successful_retries = [
            e
            for e in retry_events
            if (
                e["answerability_delta"] > 0.05
                or
                e["relevance_delta"] > 0.05
                or
                e["diversity_delta"] > 0.05
            )
        ]

        retry_success_rate = (
            len(successful_retries)
            / len(retry_events)
            if retry_events
            else 0
        )

        intent_counts = Counter(
            e["intent"]
            for e in events
        )

        strategy_counts = Counter(
            e["strategy"]
            for e in events
        )

        retry_rate = (
            sum(
                e["used_retry"]
                for e in events
            )
            / total_queries
        )

        ood_rate = (
            sum(
                e["is_ood"]
                for e in events
            )
            / total_queries
        )

        avg_initial_answerability = (
            sum(
                e["initial_answerability"]
                for e in events
            )
            / total_queries
        )

        avg_final_answerability = (
            sum(
                e["final_answerability"]
                for e in events
            )
            / total_queries
        )

        avg_answerability_delta = (
            sum(
                e["answerability_delta"]
                for e in events
            )
            / total_queries
        )

        avg_relevance_delta = (
            sum(
                e["relevance_delta"]
                for e in events
            )
            / total_queries
        )

        avg_diversity_delta = (
            sum(
                e["diversity_delta"]
                for e in events
            )
            / total_queries
        )

        print("\n" + "=" * 50)
        print("OBSERVABILITY REPORT")
        print("=" * 50)

        print(f"\nTotal Queries: {total_queries}")

        print("\nIntent Distribution:")
        for k, v in intent_counts.items():
            print(f"  {k}: {v}")

        print("\nStrategy Distribution:")
        for k, v in strategy_counts.items():
            print(f"  {k}: {v}")

        print(f"\nRetry Rate: {retry_rate:.2%}")
        print(f"OOD Rate: {ood_rate:.2%}")

        print(
            f"\nAverage Initial Answerability: "
            f"{avg_initial_answerability:.3f}"
        )

        print(
            f"Average Final Answerability: "
            f"{avg_final_answerability:.3f}"
        )

        print(
            f"Average Answerability Improvement: "
            f"{avg_answerability_delta:.3f}"
        )

        retried_queries = retry_events

        print("\nRETRIED QUERIES:")

        for e in retried_queries:

            print("\n" + "=" * 50)

            print(f"Query: {e['query']}")

            print(
                f"Retry Reasons: "
                f"{e.get('retry_reasons', [])}"
            )

            print("\nANSWERABILITY")

            print(
                f"  Initial: "
                f"{e['initial_answerability']:.3f}"
            )

            print(
                f"  Final: "
                f"{e['final_answerability']:.3f}"
            )

            print(
                f"  Delta: "
                f"{e['answerability_delta']:.3f}"
            )

            print("\nRELEVANCE")

            print(
                f"  Initial: "
                f"{e['initial_relevance']:.3f}"
            )

            print(
                f"  Final: "
                f"{e['final_relevance']:.3f}"
            )

            print(
                f"  Delta: "
                f"{e['relevance_delta']:.3f}"
            )

            print("\nDIVERSITY")

            print(
                f"  Initial: "
                f"{e['initial_diversity']:.3f}"
            )

            print(
                f"  Final: "
                f"{e['final_diversity']:.3f}"
            )

            print(
                f"  Delta: "
                f"{e['diversity_delta']:.3f}"
            )

            print(
                f"\nRetry Success Rate: "
                f"{retry_success_rate:.2%}"
            )

            print("\nRetry Reason Distribution:")

            for reason, count in retry_reason_counts.items():

                print(
                    f"  {reason}: {count}"
                )

            print(
                f"Average Relevance Improvement: "
                f"{avg_relevance_delta:.3f}"
            )

            print(
                f"Average Diversity Improvement: "
                f"{avg_diversity_delta:.3f}"
            )