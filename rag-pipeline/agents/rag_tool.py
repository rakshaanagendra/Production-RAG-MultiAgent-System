from langchain_core.tools import tool
from typing import Any, Dict
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Path setup
# Why: rag_tool.py lives in rag-pipeline/agents/
# but it needs to import from rag-pipeline/retrieval/
# So we add rag-pipeline/ to Python's search path
# --------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from retrieval.multi_query_hybrid import MultiQueryHybridRetriever

# --------------------------------------------------------------------------
# Initialise retriever once at module load time
# Why: loading the FAISS index + models is expensive (seconds)
# We don't want to reload it every time the agent calls the tool
# --------------------------------------------------------------------------
_retriever = MultiQueryHybridRetriever()


@tool
def rag_search(query: str) -> Dict[str, Any]:
    """Use this tool to answer questions about AI, LLMs, RAG pipelines,
    retrieval systems, agents, multi-agent systems, LLM engineering,
    fine-tuning, and related research topics. Input must be the user's
    question as a plain string."""

    result = _retriever.adaptive_search_with_retry(
        query=query,
        num_queries=4,
        retrieve_k=40,
        final_k=10,
        context_top_k=5,
    )

    final = result["final_result"]
    diagnostics = final["diagnostics"]
    confidence_route = result.get("confidence_route", {})

    # Extract sources from context chunks
    sources = list({
        chunk.get("source", "unknown")
        for chunk in final.get("context_chunks", [])
    })

    return {
        "query": query,
        "context": final.get("context", ""),
        "confidence": confidence_route.get("confidence", "low"),
        "action": confidence_route.get("action", "retry_or_abstain"),
        "answerable": diagnostics.get("answerability", {}).get("can_answer", False),
        "sources": sources,
        "retrieval_strategy": diagnostics.get("retrieval_strategy", "unknown"),
    }

if __name__ == "__main__":

    # Test 1: In-domain query
    print("=== TEST 1: IN-DOMAIN QUERY ===")
    result = rag_search.invoke({"query": "What is ReAct?"})
    print(f"query             : {result['query']}")
    print(f"confidence        : {result['confidence']}")
    print(f"action            : {result['action']}")
    print(f"answerable        : {result['answerable']}")
    print(f"retrieval_strategy: {result['retrieval_strategy']}")
    print(f"sources           : {result['sources']}")
    print(f"context length    : {len(result['context'])} chars")
    print(f"context preview   :\n{result['context'][:300]}")

    # Test 2: OOD query
    print("\n=== TEST 2: OOD QUERY ===")
    result2 = rag_search.invoke({"query": "What is the capital of France?"})
    print(f"answerable        : {result2['answerable']}")
    print(f"action            : {result2['action']}")
    print(f"confidence        : {result2['confidence']}")
    print(f"sources           : {result2['sources']}")
    print(f"context length    : {len(result2['context'])} chars")