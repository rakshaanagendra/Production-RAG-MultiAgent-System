import json
from pathlib import Path
from collections import Counter


class MetricsAggregator:


    def __init__(self):

        self.log_file = (
            Path(__file__).resolve().parent
            / "query_metrics.jsonl"
        )


    def percentile(self, values, percentile):

        if not values:
            return 0

        values = sorted(values)

        index = int(
            (percentile / 100)
            * (len(values) - 1)
        )

        return values[index]

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

        cache_hits = sum(
            e.get(
                "query_cache_hit",
                False
            )
            for e in events
        )

        cache_misses = total_queries - cache_hits

        cache_hit_rate = (
            cache_hits / total_queries
            if total_queries
            else 0
        )

        total_latency_saved_ms = sum(
            e.get(
                "query_cache_latency_saved_ms",
                0
            )
            for e in events
        )

        avg_latency_saved_ms = (
            total_latency_saved_ms
            / cache_hits
            if cache_hits
            else 0
        )

        retrieval_latencies = [
            e.get(
                "retrieval_latency_ms",
                0
            )
            for e in events
        ]

        query_expansion_latencies = [
            e.get("query_expansion_latency_ms", 0)
            for e in events
        ]

        avg_query_expansion_latency = (
            sum(query_expansion_latencies)
            / len(query_expansion_latencies)
        )

        avg_domain_fit = (
        sum(
            e.get(
                "domain_fit",
                0
            )
            for e in events
        )
        / total_queries
    )

        total_latencies = [
            e.get(
                "total_search_latency_ms",
                0
            )
            for e in events
        ]

        avg_retrieval_latency = (
            sum(retrieval_latencies)
            / len(retrieval_latencies)
        )

        avg_total_latency = (
            sum(total_latencies)
            / len(total_latencies)
        )

        p50_latency = self.percentile(
            total_latencies,
            50
        )

        p95_latency = self.percentile(
            total_latencies,
            95
        )

        rerank_latencies = [
            e.get("rerank_latency_ms", 0)
            for e in events
        ]

        avg_rerank_latency = (
            sum(rerank_latencies)
            / len(rerank_latencies)
        )

        p95_rerank_latency = self.percentile(
            rerank_latencies,
            95
        )

        rerank_candidate_counts = [
            e.get("rerank_candidates_count", 0)
            for e in events
        ]

        avg_rerank_candidates = (
            sum(rerank_candidate_counts)
            / len(rerank_candidate_counts)
        )

        latency_per_candidate_values = [
            e.get("latency_per_candidate_ms", 0)
            for e in events
        ]

        avg_latency_per_candidate = (
            sum(latency_per_candidate_values)
            / len(latency_per_candidate_values)
        )

        diagnostics_latencies = [
            e.get("diagnostics_latency_ms", 0)
            for e in events
        ]

        avg_diagnostics_latency = (
            sum(diagnostics_latencies)
            / len(diagnostics_latencies)
        )


        retry_latencies = [
            e.get("retry_latency_ms", 0)
            for e in events
            if e.get("used_retry", False)
        ]

        avg_retry_latency = (
            sum(retry_latencies)
            / len(retry_latencies)
            if retry_latencies
            else 0
        )


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

        confidence_action_counts = Counter(
            e.get(
                "confidence_action",
                "unknown"
            )
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

        avg_initial_risk = (
            sum(
                e.get(
                    "initial_risk_score",
                    0
                )
                for e in events
            )
            / total_queries
        )

        avg_final_risk = (
            sum(
                e.get(
                    "final_risk_score",
                    0
                )
                for e in events
            )
            / total_queries
        )

        avg_expanded_queries = (
            sum(
                e.get(
                    "expanded_queries_count",
                    0
                )
                for e in events
            )
            / total_queries
        )

        avg_merged_candidates = (
            sum(
                e.get(
                    "merged_candidates",
                    0
                )
                for e in events
            )
            / total_queries
        )

        avg_reranked_candidates = (
            sum(
                e.get(
                    "reranked_candidates",
                    0
                )
                for e in events
            )
            / total_queries
        )

        avg_context_chunks = (
            sum(
                e.get(
                    "context_chunks",
                    0
                )
                for e in events
            )
            / total_queries
        )

        print("\n" + "=" * 50)
        print("OBSERVABILITY REPORT")
        print("=" * 50)

        print(f"\nTotal Queries: {total_queries}")

        print("\nRETRIEVAL SIZES")

        print(
            f"Average Expanded Queries: "
            f"{avg_expanded_queries:.2f}"
        )

        print(
            f"Average Merged Candidates: "
            f"{avg_merged_candidates:.2f}"
        )

        print(
            f"Average Reranked Candidates: "
            f"{avg_reranked_candidates:.2f}"
        )

        print(
            f"Average Context Chunks: "
            f"{avg_context_chunks:.2f}"
        )

        print("\nCACHE REPORT")

        print(
            f"Cache Hits: "
            f"{cache_hits}"
        )

        print(
            f"Cache Misses: "
            f"{cache_misses}"
        )

        print(
            f"Cache Hit Rate: "
            f"{cache_hit_rate:.2%}"
        )

        retrieval_cache_hits = sum(
            e.get(
                "retrieval_cache_hit",
                False
            )
            for e in events
        )

        retrieval_cache_hit_rate = (
            retrieval_cache_hits
            / total_queries
            if total_queries
            else 0
        )

        print(
            f"Retrieval Cache Hit Rate: "
            f"{retrieval_cache_hit_rate:.2%}"
        )

        print(
            f"Total Latency Saved: "
            f"{total_latency_saved_ms:.2f} ms"
        )

        print(
            f"Average Latency Saved Per Hit: "
            f"{avg_latency_saved_ms:.2f} ms"
        )

        print("\nLATENCY REPORT")

        print(
            f"Average Retrieval Latency: "
            f"{avg_retrieval_latency:.2f} ms"
        )

        print(
            f"Average Total Latency: "
            f"{avg_total_latency:.2f} ms"
        )

        print(
            f"P50 Latency: "
            f"{p50_latency:.2f} ms"
        )

        print(
            f"P95 Latency: "
            f"{p95_latency:.2f} ms"
        )

        print(
            f"Average Rerank Latency: "
            f"{avg_rerank_latency:.2f} ms"
        )

        print(
            f"P95 Rerank Latency: "
            f"{p95_rerank_latency:.2f} ms"
        )

        print(
            f"Average Rerank Candidates: "
            f"{avg_rerank_candidates:.2f}"
        )

        print(
            f"Average Latency Per Candidate: "
            f"{avg_latency_per_candidate:.2f} ms"
        )

        print(
            f"Average Retry Latency: "
            f"{avg_retry_latency:.2f} ms"
        )

        print(
            f"Average Query Expansion Latency: "
            f"{avg_query_expansion_latency:.2f} ms"
        )

        print(
            f"Average Diagnostics Latency: "
            f"{avg_diagnostics_latency:.2f} ms"
        )


        print("\nIntent Distribution:")
        for k, v in intent_counts.items():
            print(f"  {k}: {v}")

        print("\nDOMAIN GATE")

        print(
            f"Average Domain Fit: "
            f"{avg_domain_fit:.3f}"
        )

        print("\nCONFIDENCE ROUTING")

        for action, count in confidence_action_counts.items():

            pct = count / total_queries

            print(
                f"  {action}: "
                f"{count} "
                f"({pct:.2%})"
            )

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

        print("\nRISK ANALYSIS")

        print(
            f"Average Initial Risk: "
            f"{avg_initial_risk:.3f}"
        )

        print(
            f"Average Final Risk: "
            f"{avg_final_risk:.3f}"
        )

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