# import re
import sys
import time
import mlflow
from pathlib import Path
from typing import cast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------
# Path setup
# Why: main.py lives at project root, but multi_agent_graph.py is inside
# rag-pipeline/agents/. We add rag-pipeline to sys.path so Python can
# find it without relative import errors.
# -----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = PROJECT_ROOT / "rag-pipeline"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from agents.multi_agent_graph import graph
from agents.state import MultiAgentState

mlflow.set_tracking_uri("sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix())
mlflow.set_experiment("production-rag-multi-agent")
# -----------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------
app = FastAPI(
    title="Production RAG Multi-Agent System",
    description="A production-grade RAG and multi-agent AI system with retrieval, writing, and grounding evaluation.",
    version="1.0.0"
)

# -----------------------------------------------------------------------
# In-memory metrics store
# Why: simple counter to track usage stats exposed via /metrics
# This gets reset when the server restarts — MLflow will persist properly later
# -----------------------------------------------------------------------
metrics = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "grounded_count": 0,
    "ungrounded_count": 0,
}

# -----------------------------------------------------------------------
# Pydantic models
# Why: FastAPI uses these as contracts — it validates incoming JSON against
# QueryRequest automatically and rejects malformed requests before they
# reach the graph. QueryResponse defines exactly what the client gets back.
# -----------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str   # the user's query — only required field

class QueryResponse(BaseModel):
    answer: str
    context: str = ""
    confidence: str
    grounded: bool
    critique: str
    sources: list[str]
    latency_ms: float
    node_latencies: dict
    agent_log: list[str]

# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Why: load balancers and cloud platforms (Render, Railway) ping this
    to know if the service is alive. Must return 200 OK quickly.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main inference endpoint.
    Accepts a question, runs the full multi-agent pipeline,
    returns the answer with metadata.
    """
    # Validate input is not empty
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Time the full pipeline
    start = time.time()

    try:
        result = graph.invoke(
            cast(MultiAgentState, {"query": request.question})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    total_latency_ms = round((time.time() - start) * 1000, 2)
    node_latencies = result.get("node_latencies", {})

    # Update in-memory metrics
    metrics["total_queries"] += 1
    metrics["total_latency_ms"] += total_latency_ms
    if result.get("answer_grounded"):
        metrics["grounded_count"] += 1
    else:
        metrics["ungrounded_count"] += 1

    # ------------------------------------------------------------------
    # MLflow GenAI trace
    # ------------------------------------------------------------------
    try:
        with mlflow.start_span(name="rag-query", span_type="CHAIN") as span:
            span.set_inputs({"question": request.question})
            span.set_outputs({
                "answer": result.get("final_answer", ""),
                "grounded": result.get("answer_grounded", False),
                "confidence": result.get("confidence", "unknown"),
            })
            span.set_attribute("retrieval_strategy", result.get("retrieval_strategy", "unknown"))
            span.set_attribute("action", result.get("action", "unknown"))
            span.set_attribute("total_latency_ms", total_latency_ms)
            span.set_attribute("sources_count", len(result.get("sources", [])))
            span.set_attribute("answer_length_chars", len(result.get("final_answer", "")))
            for node_name, latency in node_latencies.items():
                span.set_attribute(f"latency_{node_name}_ms", latency)
    except Exception as e:
        print(f"MLflow tracing failed: {e}", flush=True)


    return QueryResponse(
        answer=result.get("final_answer", ""),
        # context=clean_context,
        confidence=result.get("confidence", "unknown"),
        grounded=result.get("answer_grounded", False),
        critique=result.get("critique", ""),
        sources=result.get("sources", []),
        latency_ms=total_latency_ms,
        node_latencies=node_latencies,
        agent_log=result.get("agent_log", [])
    )


@app.get("/metrics")
def get_metrics():
    """
    Basic observability endpoint.
    Returns aggregate stats across all queries since server start.
    Will be replaced by MLflow dashboard in the next step.
    """
    total = metrics["total_queries"]
    avg_latency = (
        round(metrics["total_latency_ms"] / total, 2) if total > 0 else 0.0
    )
    grounding_rate = (
        round(metrics["grounded_count"] / total * 100, 1) if total > 0 else 0.0
    )

    return {
        "total_queries": total,
        "avg_latency_ms": avg_latency,
        "grounding_rate_pct": grounding_rate,
        "grounded_count": metrics["grounded_count"],
        "ungrounded_count": metrics["ungrounded_count"],
        "mlflow_ui": "run 'mlflow ui' then visit http://localhost:5000"
    }