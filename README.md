# RAG Agent with DuckDB & Open-Source LLMs

A lightweight, reproducible **Retrieval-Augmented Generation (RAG) agent** built as a **data engineering portfolio project**.

---

## **Project Goals**
- Demonstrate an end-to-end RAG pipeline, starting with text and evolving toward quantitative reasoning.
- Use **DuckDB** as a single-file, portable backend for logging, metadata, and (eventually) vector storage.
- Emphasize **provenance, source citation, and structured logging** from day one.
- Keep the entire stack **Dockerized** for reproducibility and easy deployment.
- Showcase **professional data engineering practices**: clean architecture, intentional schema design, error handling, and analytics on usage.

---

## **Current Milestone**
**v0.1 — Stable Dummy Agent (Complete)** ✅

- Provenance-aware responses with source citation.
- Structured DuckDB logging (`query_log` table with JSON sources).
- Interactive REPL, single-query CLI, and built-in `stats` command.
- Graceful error handling and timezone-aware timestamps.
- Minimal external dependencies (Click + DuckDB).

> This milestone validates the **full agent architecture**: interface → core logic → persistence → provenance → response flow.

---

## **Roadmap**

### **v0.2 — LLM-Backed Agent (Local Inference)**
**Goal:** Replace dummy responses with real language synthesis using a local open-source LLM.  
**Key Features**
- Local inference via llama.cpp or Ollama (e.g., Phi-3-mini, Mistral-7B-Instruct, or LLaMA-3-8B).
- Prompt engineering for coherent, concise responses.
- Unchanged CLI/REPL and DuckDB logging.
**Portfolio Highlight:** Hands-on local LLM integration without cloud APIs — demonstrates self-contained AI pipelines.

### **v0.3 — Text RAG (Retrieval-Augmented Generation)**
**Goal:** Ground LLM responses in a real text corpus for factual, source-backed answers.  
**Key Features**
- Text chunking + embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- DuckDB native vector storage (FLOAT[384] arrays) and cosine similarity search.
- Top-k retrieval → context-augmented prompts → source citation in responses.
**Portfolio Highlight:** Core technical achievement — building grounded retrieval from scratch in a lightweight stack.

### **v0.4 — Standalone Agent Server**
**Goal:** Expose the full RAG agent as a reusable service.  
**Key Features**
- FastAPI server with `/v1/query` endpoint (returns response + sources + metadata).
- Dockerized service ready for local or remote consumption.
- Optional CLI as HTTP client.
**Portfolio Highlight:** Turns the intelligent agent into a production-like microservice consumable by external systems.

### **v0.5 — Quantitative Data Integration**
**Goal:** Enable hybrid reasoning over text + structured numeric datasets.  
**Key Features**
- Load complementary open datasets (e.g., CMS hospital readmissions, CDC statistics) into DuckDB.
- SQL-based numeric retrieval + synthesis with textual context.
- Provenance tracking for both text chunks and numeric sources.
**Portfolio Highlight:** Real-world analytical capability — bridging unstructured and structured data.

### **v0.6 — Incremental Updates & Optional Frontend**
**Goal:** Add production-grade incremental data pipeline and polished demo interface.  
**Key Features**
- dbt project targeting DuckDB for incremental materializations (text embeddings + numeric models).
- Optional Streamlit/Gradio frontend for interactive demos.
- Full CI/CD via GitHub Actions.
**Portfolio Highlight:** Mature, analytics-engineering workflow with continuous updates and observability.

---

## **Non-Goals**
- Training or fine-tuning LLMs from scratch.
- Distributed or cloud-scale deployment.
- Handling massive (>TB) datasets.
- Complex orchestration frameworks.

> These are intentionally out of scope to keep the project **focused, reproducible, and demonstrable** on any machine.

---

## **Technical Highlights**
- **DuckDB** as the single-file analytical backbone — logging, metadata, and vector storage.
- **Provenance-first** design — every response cites its sources from v0.1 onward.
- **Modular, replaceable core** — dummy → LLM → RAG → server without breaking prior versions.
- Fully **Dockerized** for instant reproducibility.
- CI/CD-ready structure for future GitHub Actions.

---

## **Getting Started (Current: v0.1)**



```bash

## Quick Start (Local)

```bash
# Clone and enter the repo
git clone https://github.com/michaelkrot/rag-agent-duckdb.git
cd rag-agent-duckdb

# Install dependencies
pip install -r requirements.txt

# Interactive REPL (recommended for demo)
python agent.py repl

# Single query
python agent.py query "What is RAG?"

# View usage statistics
python agent.py stats

# Show version
python agent.py --version```

## Docker instructions


### Build the docker image 
```bash
# Build the image
docker build -t rag-agent .

# Interactive REPL (default – great for demos)
docker run -it --rm -v $(pwd)/data:/app/data rag-agent

# Single query
docker run -it --rm -v $(pwd)/data:/app/data rag-agent python agent.py query "What database are you using?"

# View stats
docker run -it --rm -v $(pwd)/data:/app/data rag-agent python agent.py stats```

