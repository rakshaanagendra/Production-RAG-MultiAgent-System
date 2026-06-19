# 🚀 LLM-Powered RAG Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Engineering-orange)
![LLM](https://img.shields.io/badge/LLM-Systems-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

## 🎯 Overview

A production-oriented Retrieval-Augmented Generation (RAG) system built to explore modern **Retrieval Engineering**, **LLM Systems Engineering**, **Evaluation Frameworks**, **Observability**, and **Hallucination Mitigation** techniques used in real-world AI applications.

Unlike traditional RAG demos that stop at:

```text
PDF → Embeddings → Vector Search → LLM
```

this project focuses on solving practical engineering challenges such as:

* Retrieval quality optimization
* Hybrid search strategies
* Query expansion
* Cross-encoder reranking
* Retrieval observability
* Confidence estimation
* Adaptive retrieval
* Evaluation pipelines
* Caching systems
* Hallucination mitigation

The long-term goal is to evolve this project into a **production-grade Agentic AI system** using LangGraph, planning workflows, tool use, and multi-agent orchestration.

---

# ✨ Key Engineering Achievements

### Retrieval Engineering

✅ Hybrid Retrieval (Dense + Sparse)

✅ BM25 Sparse Retrieval

✅ Dense Vector Retrieval using Sentence Transformers

✅ Reciprocal Rank Fusion (RRF)

✅ Multi-Query Retrieval

✅ LLM-Driven Query Expansion

✅ Dynamic Retrieval Parameters

✅ Adaptive Retrieval Strategies

---

### Ranking & Relevance

✅ Cross-Encoder Reranking (`BAAI/bge-reranker-base`)

✅ Candidate Quality Analysis

✅ Rerank Lift Diagnostics

✅ Retrieval Failure Analysis

---

### Evaluation & Reliability

✅ Retrieval Health Scoring

✅ Answerability Estimation

✅ Confidence Scoring

✅ Semantic Validation

✅ Safe Abstention Handling

---

### Observability & Diagnostics

✅ Retrieval Latency Monitoring

✅ End-to-End Latency Tracking

✅ Duplicate Detection

✅ Source Diversity Analysis

✅ Rerank Analysis

✅ Retry Effectiveness Analysis

✅ Retrieval Diagnostics Framework

---

### Performance Optimization

✅ Query Cache

✅ Retrieval Cache

✅ Dynamic Context Construction

✅ Confidence-Based Routing

✅ Adaptive Retry Policies

---

# 🏗️ System Architecture

```text
                                ┌─────────────────────┐
                                │    Knowledge Base   │
                                │                     │
                                │ Research Papers     │
                                │ Technical Notes     │
                                │ AI Engineering Docs │
                                │ FAISS Vector Store  │
                                └──────────┬──────────┘
                                           │
                                           ▼

User Query
    │
    ▼
Query Cache
    │
    ▼
Query Analysis & Domain Gate
    │
    ▼
Multi-Query Expansion
    │
    ▼
Retrieval Cache
    │
    ▼
Dense Retrieval + BM25 Sparse Retrieval
    │
    ▼
Reciprocal Rank Fusion (RRF)
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Context Compression
    │
    ▼
Retrieval Diagnostics
    │
    ▼
Answerability Estimation
    │
    ▼
Confidence Routing
    │
    ▼
Grounded Generation
    │
    ▼
Semantic Validation
    │
    ▼
Final Response


┌─────────────────────────────────────────────┐
│                Observability                │
├─────────────────────────────────────────────┤
│ • Query Latency Tracking                    │
│ • Retrieval Latency Tracking                │
│ • Reranking Metrics                         │
│ • Cache Hit/Miss Monitoring                 │
│ • Retrieval Diagnostics                     │
│ • Confidence Routing Analytics              │
│ • Query Logging & Evaluation                │
└─────────────────────────────────────────────┘
```

# 📚 Knowledge Base Ingestion Pipeline

```text
Raw Documents
(PDFs, Research Papers, Notes)
            │
            ▼
Document Loader
(PyMuPDF / TXT Loader)
            │
            ▼
Text Cleaning
            │
            ▼
Document Chunking
            │
            ▼
Embedding Generation
(BAAI/bge-small-en-v1.5)
            │
            ▼
FAISS Index Construction
            │
            ▼
Metadata Store
            │
            ▼
Knowledge Base Ready
```

---

# 🔍 Current Capabilities

## Retrieval Layer

* Dense Retrieval
* BM25 Sparse Retrieval
* Hybrid Search
* Reciprocal Rank Fusion (RRF)
* Multi-Query Expansion
* Adaptive Retrieval
* Dynamic Retrieval Configuration

## Ranking Layer

* CrossEncoder Reranking
* Candidate Quality Evaluation
* Retrieval Relevance Analysis

## Evaluation Layer

* Retrieval Health Scoring
* Answerability Estimation
* Confidence Estimation
* Semantic Validation

## Observability Layer

* Retrieval Diagnostics
* Source Diversity Monitoring
* Duplicate Analysis
* Rerank Diagnostics
* Latency Monitoring
* Retry Analysis

## Optimization Layer

* Query Cache
* Retrieval Cache
* Confidence Routing
* Dynamic Context Selection
* Context compression and prompt ready context
* Retry Policies

---

# 📊 Engineering Metrics Tracked

The system continuously measures and analyzes:

| Metric                   | Purpose                           |
| ------------------------ | --------------------------------- |
| Retrieval Latency        | Retrieval performance             |
| Rerank Latency           | Ranking performance               |
| Total Pipeline Latency   | End-to-end performance            |
| Duplicate Retrieval Rate | Retrieval efficiency              |
| Source Diversity         | Retrieval robustness              |
| Rerank Lift              | Reranker effectiveness            |
| Cache Hit Rate           | Optimization effectiveness        |
| Retry Success Rate       | Adaptive retrieval effectiveness  |
| Retrieval Health Score   | Retrieval quality                 |
| Answerability Score      | Likelihood of answering correctly |

---

# 🧪 Evaluation Framework

The system is evaluated using a benchmark suite containing:

* Fact Lookup Queries
* Definition Queries
* Comparison Queries
* How-To Queries
* Multi-Hop Reasoning Queries
* Exploratory Queries
* Timeline Queries
* Verification Queries
* Out-of-Domain Queries

Current benchmark size:

**60+ evaluation queries**

designed to stress-test retrieval quality and answer grounding.

---

# 🛠️ Technology Stack

## LLM & NLP

* Ollama 
* Qwen 2.5:3B
* Hugging Face Transformers
* Sentence Transformers
* CrossEncoder Models

## Retrieval

* Embedding - BAAI/bge-small-en-v1.5
* FAISS vector store
* BM25
* Hybrid Search
* Reciprocal Rank Fusion (RRF)

## Ranking

* CrossEncoder Reranking - BAAI/bge-reranker-base

## Document parsing

* PyMuPDF (fitz)
* PyPDF2 (PdfReader)

## Data Processing

* NumPy
* Pandas
* LangChain

## Engineering

* Python
* Git
* JSON-Based Caching

---

# 📁 Project Structure

```text
Production-RAG-AI-Assistant-with-Reranking-and-Evaluation-Pipeline/

├── data/
│   ├── raw/
│   ├── processed/
│   └── queries/
│
├── rag-pipeline/
│
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── multi_query_retriever.py
│   │   ├── multi_query_hybrid.py
│   │   ├── query_rewriter.py
│   │   └── reranker.py
│
│   ├── generation/
│   │   └── generator.py
│   
│   ├── caching/
│   │   ├── cache_manager.py
│   │   ├── query_cache.json
|   |   ├── retrieval_cache.py
│   │   ├── retrieval_cache.json
│   
│   ├── observability/ tests/
│   │   ├── test_analytics.py
│   │   ├── metrics_logger..py
│   │   ├── metrics_aggregator.py
│   │   ├── query_metrics.jsonl
|
├── outputs/
├── scripts/ ingest.py
├── vectorstore/ faiss_indexer.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 💡 Why This Project Stands Out

Most RAG projects demonstrate only retrieval and generation.

This project explores the engineering aspects required to build reliable AI systems:

* Retrieval Engineering
* Hybrid Search
* Query Expansion
* Reranking
* Confidence Estimation
* Hallucination Mitigation
* Evaluation Frameworks
* Retrieval Observability
* Adaptive Retrieval
* Caching
* Production-Oriented Diagnostics

The focus is on understanding **why retrieval systems succeed or fail**, and building mechanisms to diagnose, measure, and improve them.

---


---

# 🔮 Future Roadmap

## Retrieval Engineering

* Metadata-Aware Retrieval
* Advanced Retrieval Diagnostics

## Agentic AI

* Query Routing
* Retrieval Critics
* Planning-Based Retrieval
* Agent Memory
* Tool-Using Agents
* LangGraph Workflows

## Multi-Agent Systems

* Research Agent
* Retrieval Agent
* Critic Agent
* Writer Agent
* Multi-Agent Collaboration

## Production AI Systems

* FastAPI Backend
* Interactive Frontend
* MLflow Integration
* Tracing & Monitoring
* Cloud Deployment
* Scalable Inference Infrastructure

---

# 👨‍💻 Author

**Raksha Nagendra**

Master's Student in Information Technology
University of Stuttgart

### Interests

* LLM Engineering
* Agentic AI
* Retrieval Engineering
* MLOps
* Production AI Systems
* Applied AI

This project serves as an evolving AI Engineering platform for exploring Retrieval-Augmented Generation, evaluation systems, observability, and future Agentic AI workflows.
