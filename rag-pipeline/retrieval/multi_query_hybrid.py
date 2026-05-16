from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT)) 

from retrieval.retriever import Retriever
from retrieval.multi_query_retriever import MultiQueryRetriever
from retrieval.reranker import Reranker


class MultiQueryHybridRetriever:
    def __init__(self, index_path=None, model_name="qwen2.5:3b", client=None):
        self.retriever = Retriever(index_path=index_path)
        self.query_generator = MultiQueryRetriever(
            model_name=model_name,
            client=client,
        )
        self.reranker = Reranker()

    def _deduplicate_by_text(self, chunks):
        seen = set()
        unique = []

        for chunk in chunks:
            text = chunk.get("text", "")
            key = text.strip()

            if key and key not in seen:
                unique.append(chunk)
                seen.add(key)

        return unique

    def _aggregate_results(self, query_results, k=60, max_per_source: Optional[int] = 3):
        scores = {}
        combined = {}

        for query_rank, chunks in enumerate(query_results):
            for rank, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                key = text.strip()

                if not key:
                    continue

                rrf_score = 1.0 / (k + rank + 1)

                if key not in combined:
                    combined[key] = chunk.copy()

                combined[key]["rrf_score"] = combined[key].get("rrf_score", 0.0) + rrf_score
                scores[key] = scores.get(key, 0.0) + rrf_score

        for key in combined:
            combined[key]["multi_query_rrf_score"] = scores.get(key, 0.0)

        ranked = sorted(
            combined.values(),
            key=lambda x: x["multi_query_rrf_score"],
            reverse=True,
        )

        if max_per_source is None:
            return ranked

        balanced = []
        per_source_counts = {}

        for chunk in ranked:
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            count = per_source_counts.get(source, 0)

            if count >= max_per_source:
                continue

            per_source_counts[source] = count + 1
            balanced.append(chunk)

        return balanced

    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        include_scores: bool = True,
    ) -> str:
        """
        Convert the top reranked chunks into a clean prompt-ready context string.
        """
        if not chunks:
            return ""

        selected_chunks = chunks[:top_k]
        context_parts = []

        for i, chunk in enumerate(selected_chunks, start=1):
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            page = chunk.get("page") or chunk.get("metadata", {}).get("page") or "unknown"
            text = chunk.get("text", "").strip()

            if not text:
                continue

            header_lines = [f"[Chunk {i}]"]
            header_lines.append(f"Source: {source}")
            header_lines.append(f"Page: {page}")

            if include_scores:
                rerank_score = chunk.get("rerank_score", None)
                rrf_score = chunk.get("multi_query_rrf_score", None)

                if rerank_score is not None:
                    header_lines.append(f"Rerank Score: {rerank_score:.4f}")
                if rrf_score is not None:
                    header_lines.append(f"RRF Score: {rrf_score:.4f}")

            header = "\n".join(header_lines)
            context_parts.append(f"{header}\nText:\n{text}")

        return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(context_parts) if context_parts else ""


    def search(
        self,
        query,
        num_queries=4,
        retrieve_k=60,
        final_k=20,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
    ):
        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=num_queries,
        )

        query_results = []

        for q in expanded_queries:
            results = self.retriever.hybrid_search(
                q,
                retrieve_k=retrieve_k,
                final_k=final_k,
                k=k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                adaptive_weights=adaptive_weights,
                min_dense_similarity=min_dense_similarity,
                min_bm25_score=min_bm25_score,
                max_per_source=max_per_source,
            )
            query_results.append(results)

        merged = self._aggregate_results(
            query_results,
            k=k,
            max_per_source=max_per_source,
        )

        # Rerank merged results with Reranker
        reranked = self.reranker.rerank(
            query=query,
            retrieved_chunks=merged,
            top_k=final_k,
        )

        return reranked

    def search_with_context(
        self,
        query,
        num_queries=4,
        retrieve_k=60,
        final_k=20,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        adaptive_weights=False,
        min_dense_similarity=0.15,
        min_bm25_score=0.0,
        max_per_source: Optional[int] = 3,
        context_top_k: int = 5,
        include_scores: bool = True,
    ):
        """
        Returns both the reranked chunks and a prompt-ready context string.
        """
        # Generate expanded queries (useful to return to callers)
        expanded_queries = self.query_generator.generate_queries(
            query,
            num_queries=num_queries,
        )

        # Perform the multi-query search and rerank
        reranked_chunks = self.search(
            query=query,
            num_queries=num_queries,
            retrieve_k=retrieve_k,
            final_k=final_k,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            adaptive_weights=adaptive_weights,
            min_dense_similarity=min_dense_similarity,
            min_bm25_score=min_bm25_score,
            max_per_source=max_per_source,
        )

        # Build the prompt-ready context string and also expose the chunks used
        context = self.build_context(
            reranked_chunks,
            top_k=context_top_k,
            include_scores=include_scores,
        )

        context_chunks = reranked_chunks[:context_top_k]

        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "reranked_chunks": reranked_chunks,
            "context_chunks": context_chunks,
            "context": context,
        }

## Basic test cases for MultiQueryHybridRetriever
# if __name__ == "__main__":
#     multi_query_retriever = MultiQueryHybridRetriever()

#     test_queries = [
#         "What is dense and sparse retrieval?",
#         "that thing where llms forget the middle part",
#         "how do vector databases work",
#         "bm25 vs embeddings",
#     ]

#     for query in test_queries:
#         print("\n==================================================")
#         print("ORIGINAL QUERY:")
#         print(query)

#         expanded_queries = multi_query_retriever.query_generator.generate_queries(
#             query,
#             num_queries=4,
#         )

#         print("\nEXPANDED QUERIES:\n")

#         for i, q in enumerate(expanded_queries, start=1):
#             print(f"[{i}] {q}")

#         results = multi_query_retriever.search(
#             query,
#             num_queries=4,
#             retrieve_k=40,
#             final_k=10,
#         )

#         print("\nFINAL RERANKED RESULTS:\n")

#         for i, c in enumerate(results, start=1):
#             print(f"[{i}] source={c.get('source', 'unknown')}")

#             if "rerank_score" in c:
#                 print(f"rerank_score={c['rerank_score']:.4f}")

#             if "multi_query_rrf_score" in c:
#                 print(f"multi_query_rrf_score={c['multi_query_rrf_score']:.4f}")

#             print(c.get("text", "")[:300])

#             print("-" * 80)

## Additional test case for search_with_context i.e. retrieving results along with a prompt-ready context string
if __name__ == "__main__":
    multi_query_retriever = MultiQueryHybridRetriever()

    test_queries = [
        "What is dense and sparse retrieval?",
        "that thing where llms forget the middle part",
        "how do vector databases work",
        "bm25 vs embeddings",
    ]

    for query in test_queries:
        print("\n==================================================")
        print("ORIGINAL QUERY:")
        print(query)

        expanded_queries = multi_query_retriever.query_generator.generate_queries(
            query,
            num_queries=4,
        )

        print("\nEXPANDED QUERIES:\n")

        for i, q in enumerate(expanded_queries, start=1):
            print(f"[{i}] {q}")

        result = multi_query_retriever.search_with_context(
            query,
            num_queries=4,
            retrieve_k=40,
            final_k=10,
            context_top_k=5,
        )

        print("\nFINAL RERANKED RESULTS:\n")

        for i, c in enumerate(result["reranked_chunks"], start=1):
            print(f"[{i}] source={c.get('source', 'unknown')}")

            if "rerank_score" in c:
                print(f"rerank_score={c['rerank_score']:.4f}")

            if "multi_query_rrf_score" in c:
                print(f"multi_query_rrf_score={c['multi_query_rrf_score']:.4f}")

            print(c.get("text", "")[:300])
            print("-" * 80)

        print("\nPROMPT-READY CONTEXT:\n")
        print(result["context"])