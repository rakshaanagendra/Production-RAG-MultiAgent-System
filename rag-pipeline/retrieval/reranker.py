from sentence_transformers import CrossEncoder
import argparse
from collections import Counter


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query, retrieved_chunks, top_k=5, threshold_ratio=0.05):
        """
        query: str
        retrieved_chunks: list of dicts [{ "text": ..., "source": ... }]
        top_k: how many final results to return
        threshold_ratio: score cutoff relative to best score.
            0.05 keeps chunks with score >= 5% of the best score.
            Adaptive fallback guarantees at least top_k candidates survive filtering.
        """

        if not retrieved_chunks:
            return []

        # Step 1: Prepare (query, chunk) pairs
        pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

        # Step 2: Get relevance scores
        scores = self._get_model().predict(pairs)

        # Step 3: Attach scores to chunks
        for chunk, score in zip(retrieved_chunks, scores):
            chunk["rerank_score"] = float(score)

        # Step 4: Sort by score (descending)
        reranked = sorted(retrieved_chunks, 
                        key=lambda x: x["rerank_score"], 
                        reverse=True)

        # Optional: drop weak reranker results.
        # Keep threshold conservative and guarantee enough candidates for top_k.
        top_score = reranked[0]["rerank_score"]
        raw_threshold = top_score * threshold_ratio
        topk_anchor_idx = min(max(top_k - 1, 0), len(reranked) - 1)
        anchor_threshold = reranked[topk_anchor_idx]["rerank_score"]
        score_threshold = min(raw_threshold, anchor_threshold)
        reranked = [c for c in reranked if c["rerank_score"] >= score_threshold]
        
        # Step 5: Return top-k
        return reranked[:top_k]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test reranker with optional retrieval step.")
    # parser.add_argument("query", nargs="?", default="What is hybrid search? List the problems and a solution for hybrid search.", help="Query text")
    # parser.add_argument("--retrieve-k", "--retrieve_k", type=int, default=30, help="Number of chunks to retrieve first")
    # parser.add_argument("--top-k", "--top_k", type=int, default=5, help="Number of reranked chunks to return")
    # parser.add_argument(
    #     "--preview-chars",
    #     "--preview_chars",
    #     type=int,
    #     default=600,
    #     help="Number of characters to print per chunk preview",
    # )
    # parser.add_argument(
    #     "--full-text",
    #     "--full_text",
    #     action="store_true",
    #     help="Print full chunk text instead of truncating",
    # )
    # parser.add_argument(
    #     "--max-per-source",
    #     "--max_per_source",
    #     type=int,
    #     default=4,
    #     help="Cap retrieved chunks per source before reranking",
    # )
    # args = parser.parse_args()

    from retriever import Retriever

    retriever = Retriever()
    reranker = Reranker()

    # retrieved = retriever.hybrid_search(
    #     args.query,
    #     retrieve_k=args.retrieve_k,
    #     final_k=args.retrieve_k,  # Get more for reranking
    # )

    # # 🔥 ADD THIS FILTER HERE
    # filtered = []
    # for c in retrieved:
    #     text = c.get("text", "").lower()

    #     if len(text) < 40:   # 🔥 reduce from 80 → 40
    #         continue

    #     if "references" in text[:100]:   # 🔥 less strict
    #         continue

    #     filtered.append(c)
    
    # print(f"Before filter: {len(retrieved)}")
    # print(f"After filter: {len(filtered)}")

    # before_sources = Counter(c.get("source", "unknown") for c in filtered)
    # print(f"Unique sources before rerank: {len(before_sources)}")
    # print(f"Top source counts before rerank: {before_sources.most_common(5)}")

    # reranked = reranker.rerank(args.query, filtered, top_k=args.top_k)

    # after_sources = Counter(c.get("source", "unknown") for c in reranked)
    # print(f"Unique sources after rerank: {len(after_sources)}")
    # print(f"Top source counts after rerank: {after_sources.most_common(5)}")

    # print(f"Query: {args.query}")
    # print(f"Retrieved: {len(retrieved)}")
    # print(f"Reranked (top_k={args.top_k}): {len(reranked)}")

    # if retrieved:
    #     highest_cosine = max(item.get("cosine_score", float("-inf")) for item in retrieved)
    #     print(f"Highest cosine similarity in retrieved set: {highest_cosine:.4f} (higher is better)")

    # def format_chunk_text(text):
    #     clean_text = text.replace("\n", " ")
    #     if args.full_text:
    #         return clean_text
    #     return clean_text[:args.preview_chars]

    # print("\n--- Retrieved Chunks (Before Reranking) ---")
    # if not retrieved:
    #     print("No retrieved chunks found.")
    # else:
    #     for i, item in enumerate(retrieved, start=1):
    #         source = item.get("source", "unknown")
    #         cosine_score = item.get("cosine_score", float("nan"))
    #         text = item.get("text", "")
    #         print(f"\n[{i}] cosine_score={cosine_score:.4f} source={source}")
    #         print(format_chunk_text(text))

    # print("\n--- Reranked Chunks (After Reranking) ---")
    # if not reranked:
    #     print("No reranked results found.")
    # else:
    #     for i, item in enumerate(reranked, start=1):
    #         source = item.get("source", "unknown")
    #         rerank_score = item.get("rerank_score", 0.0)
    #         cosine_score = item.get("cosine_score", float("nan"))
    #         text = item.get("text", "")
    #         print(f"\n[{i}] rerank_score={rerank_score:.4f} cosine_score={cosine_score:.4f} source={source}")
    #         print(format_chunk_text(text))

    query = "who invented transformers?"

    chunk = """
    The Transformer was introduced in the seminal
    2017 paper Attention Is All You Need.

    Authors:
    Ashish Vaswani
    Noam Shazeer
    Niki Parmar
    Jakob Uszkoreit
    Llion Jones
    Aidan Gomez
    Lukasz Kaiser
    Illia Polosukhin
    """

    print(reranker._get_model().predict([(query, chunk)]))