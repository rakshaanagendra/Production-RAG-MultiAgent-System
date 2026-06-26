# Production RAG & Multi-Agent AI System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)
![RAG](https://img.shields.io/badge/RAG-Production--Grade-orange)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)
![License](https://img.shields.io/badge/License-MIT-green)

> A production-oriented RAG and multi-agent AI system built from the ground up — not a demo, not a tutorial clone. Every component is engineered to solve a real problem that naive RAG implementations fail at.

---

## What This Is

Most RAG projects look like this:

```
PDF → Embeddings → Vector Search → LLM → Answer
```

This project treats that as the starting point, not the finish line.

Built in phases, this system progressively adds the engineering layers that make AI systems reliable in production: retrieval engineering, adaptive routing, observability, agentic decision-making, multi-tool orchestration, and multi-agent collaboration with LLM-as-judge evaluation.

**The core question driving every design decision:** *Why do RAG systems fail, and what does it take to make them not fail?*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT LAYER                           │
│                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────┐  │
│   │  Research Node  │──▶│   Writer Node   │──▶│ Critic Node │  │
│   │  (ReAct Agent)  │   │  (Qwen2.5:7b)   │   │ (llama3.1)  │  │
│   │                 │   │  tone routing   │   │ LLM-as-judge│  │
│   │  rag_search     │   │  hard gates     │   │ grounding   │  │
│   │  web_search     │   │                 │   │ check       │  │
│   └────────┬────────┘   └─────────────────┘   └─────────────┘  │
│            │                    MultiAgentState (shared)        │
└────────────┼────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE LAYER                           │
│                                                                 │
│  Query Cache → Domain Gate → Multi-Query Expansion             │
│       → Retrieval Cache → Dense + BM25 Retrieval               │
│       → RRF Fusion → Cross-Encoder Reranking                   │
│       → Context Compression → Confidence Routing               │
│       → Grounded Generation → Semantic Validation              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Observability                         │  │
│  │  Latency · Cache Hits · Retry Analytics · Health Score  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE                               │
│  Research Papers (AI, Agents, RAG, LLM Engineering, LLMOps)   │
│  FAISS Vector Store · BM25 Index · Metadata Store             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Engineering Phases

### Phase 1–2 — Retrieval Engineering

The foundation. Built to understand and solve the hard problems in retrieval before adding any agent layer on top.

**What was built:**

- Hybrid retrieval combining dense (FAISS + `BAAI/bge-small-en-v1.5`) and sparse (BM25) search
- Reciprocal Rank Fusion (RRF) for merging dense and sparse rankings without score normalisation
- Cross-encoder reranking (`BAAI/bge-reranker-base`) as a second-pass relevance filter
- Multi-query expansion — LLM rewrites each query into N variants to improve recall
- Adaptive retrieval controller — selects strategy (dense/hybrid/sparse) based on query type
- Confidence-aware routing — routes low-confidence retrievals to retry or abstention
- OOD domain gate — filters queries outside the corpus domain before retrieval
- Context compression — reduces prompt context by 14–50% while preserving critical evidence
- Query and retrieval caching — stateful retrieval across repeated workloads
- Full observability framework — latency, cache hit rate, rerank lift, retry effectiveness, source diversity

**Measured outcomes:**

| Metric | Before | After |
|---|---|---|
| Unnecessary retrieval retries | 19.2% | 11.5% (↓40%) |
| Retry effectiveness | 20% | 33.3% (↑66%) |
| Prompt context reduction | baseline | 14–50% |

---

### Phase 3–4 — Agentic AI & Multi-Agent Systems

Built a full agentic layer on top of the retrieval pipeline.

**Phase 3 — ReAct Agent:**

- ReAct-style agent in LangGraph: `llm_node` (Qwen2.5:7b with tools bound) + `tool_node` + `tools_condition` routing
- Two tools: `rag_search` (wraps the full RAG pipeline as a callable tool) and `web_search` (Tavily API)
- Agent autonomously selects tool based on query topic — no explicit routing instruction per turn
- Short-term memory via LangGraph `MemorySaver` — maintains full message history per `thread_id`
- Validated thread isolation: different `thread_id` values produce completely independent states
- Empirically settled on Qwen2.5:7b as the minimum reliable model for tool calling — 3B stripped queries, Mistral 7B hallucinated tool calls

**Phase 4 — Multi-Agent Pipeline:**

- Three-node pipeline wired in LangGraph with shared `MultiAgentState` (TypedDict with `operator.add` for append-only observability logging)
- `research_node` — wraps the ReAct agent as a callable sub-graph, parses structured tool output via `json.loads`, maps fields to shared state
- `writer_node` — hard gates on `answerable`, `research_context`, and `action` fields before generating; confidence-aware tone routing (confident vs cautious); structured JSON output with markdown fence stripping and newline normalisation
- `critic_node` — LLM-as-judge using `llama3.1:8b` to evaluate `qwen2.5:7b` output; strict grounding check comparing every claim against retrieved context; returns `answer_grounded` boolean + detailed `critique`
- `uuid`-based thread isolation per research node call — prevents state bleed from `MemorySaver` across invocations
- Diagnosed and fixed two silent inter-agent parsing failures: `ast.literal_eval` rejecting valid JSON from `ToolMessage` content, and raw newline control characters breaking `json.loads` on local model output

---

## Technology Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLMs (local) | Qwen2.5:7b (agent/writer), llama3.1:8b (critic) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Vector Store | FAISS |
| Sparse Retrieval | BM25 |
| Web Search | Tavily API |
| Document Parsing | PyMuPDF, PyPDF2 |
| Data Processing | NumPy, Pandas |
| Inference Runtime | Ollama |
| Version Control | Git |

---

## Project Structure

```
Production-RAG-MultiAgent-System/
│
├── rag-pipeline/
│   ├── agents/
│   │   ├── state.py               # MultiAgentState schema
│   │   ├── rag_tool.py            # RAG pipeline wrapped as LangChain tool
│   │   ├── web_search_tool.py     # Tavily web search tool
│   │   ├── rag_agent.py           # ReAct agent (llm_node + tool_node)
│   │   ├── research_node.py       # Research agent node
│   │   ├── writer_node.py         # Writer agent node
│   │   ├── critic_node.py         # Critic agent node (LLM-as-judge)
│   │   └── multi_agent_graph.py   # Outer orchestration graph
│   │
│   ├── retrieval/
│   │   ├── multi_query_hybrid.py  # Main retrieval orchestrator
│   │   ├── retriever.py           # Dense retrieval
│   │   ├── sparse_retriever.py    # BM25 sparse retrieval
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_rewriter.py      # Multi-query expansion
│   │   └── multi_query_retriever.py
│   │
│   ├── generation/
│   │   └── generator.py           # Grounded generation + semantic validation
│   │
│   ├── caching/
│   │   ├── cache_manager.py
│   │   ├── query_cache.json
│   │   └── retrieval_cache.json
│   │
│   └── observability/
│       ├── metrics_logger.py
│       ├── metrics_aggregator.py
│       └── query_metrics.jsonl
│
├── vectorstore/
│   └── faiss_indexer.py
│
├── data/
│   └── raw/                       # Research papers by topic
│       ├── agents/
│       ├── rag/
│       ├── multi_agent/
│       ├── llm_engineering/
│       ├── langgraph/
│       ├── llmops/
│       └── deployment/
│
├── scripts/
│   └── ingest.py
│
├── requirements.txt
└── README.md
```

---

## Key Engineering Decisions

**Why local models only?**
The entire system runs on Ollama with no external API calls for generation (except Tavily for web search). This was a deliberate constraint — it forces real engineering solutions to small model limitations rather than hiding them behind a frontier model.

**Why Qwen2.5:7b as the floor for tool calling?**
Empirically tested 3B (query stripping), Mistral 7B (hallucinated tool calls), and Qwen2.5:7b (reliable). 7B is the minimum viable size for consistent tool-calling behaviour on this stack — a real production constraint on cost/latency vs reliability.

**Why a separate critic model?**
Using `llama3.1:8b` as the critic and `qwen2.5:7b` as the writer mirrors the production LLM-as-judge pattern — different models selected per role, not a single model doing everything. The critic's job is evaluation, not generation, so a different model with stricter instruction-following is the right choice.

**Why `operator.add` on `agent_log`?**
Each node appends its own log entry without overwriting prior entries. At pipeline end, `agent_log` contains the full decision trace across all three agents — research strategy, writer tone decision, critic grounding verdict — in one field. That's observability built into the state schema.

---

## What's Next (Phase 5)

- FastAPI endpoint wrapping the multi-agent graph
- MLflow experiment tracking — log confidence, grounding, retrieval strategy per run
- Per-node latency logging
- Docker containerisation
- CI/CD via GitHub Actions
- Cloud deployment (free tier)
- Observability dashboard

---

## Author

**Raksha Nagendra**
Master's Student in Information Technology — Universität Stuttgart
Seeking AI/LLM/Agentic AI Engineer roles | Willing to relocate

[LinkedIn](https://www.linkedin.com/in/raksha-nagendra) · [GitHub](https://github.com/rakshaanagendra)