# RAG Agent with DuckDB & Open-Source LLMs

A lightweight, reproducible **Retrieval-Augmented Generation (RAG) agent** built as a demonstration of basic RAG and agent concepts in a portfolio project.  

---

## **Project Goals**
- Demonstrate an end-to-end RAG pipeline, starting with text and evolving toward quantitative reasoning.
- Use **DuckDB** as a single-file, portable backend for logging, metadata, and (eventually) vector storage.
- Emphasize **provenance, source citation, and structured logging** from day one.
- Showcase: clean architecture, intentional schema design, error handling, and analytics on usage.

---

## **Current Milestone**
**v0.1 — Stable Dummy Agent (Complete)** ✅

- Provenance-aware responses with source citation.
- Structured DuckDB logging (`query_log` table with JSON sources).
- Interactive REPL, single-query CLI, and built-in `stats` command.
- Graceful error handling and timezone-aware timestamps.
- Minimal external dependencies (Click + DuckDB).

> This milestone validates the **full agent architecture**: interface → core logic → persistence → provenance → response flow.


## v0.3 — Text RAG Complete ✅

- Changed plans and skipped 0.2 temporarily (will return)
- Real vector retrieval over TMDB movie overviews (2015–2025)
- HNSW index for fast cosine similarity search
- Provenance-aware responses with movie title/year citation
- Embedding model loaded once for performance
- Small Pytest suite validating schema, retrieval, and core logic

Run:
```bash
python agent.py repl
# Ask: "What is Dune about?" or "Movies like Inception"
```
---

## **Roadmap**

### **v0.2 — LLM-Backed Agent (Local Inference)**
**Goal:** Replace dummy responses with real language synthesis using a local open-source LLM.  
**Key Features**
- Local inference via llama.cpp or Ollama (e.g., Phi-3-mini, Mistral-7B-Instruct, or LLaMA-3-8B).
- Prompt engineering for coherent, concise responses.
- Unchanged CLI/REPL and DuckDB logging.
**Portfolio Highlight:** Hands-on local LLM integration without cloud APIs — demonstrates self-contained AI pipelines.

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

## Data Sources

### Text Corpus (v0.3 RAG Retrieval)
- **TMDB Movies Dataset 2024** (1M+ films, up to 2025 releases)  
  Link: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies  
  License: ODC-BY 1.0 (attribution provided)  
  Used: "overview" column for plot summaries

In order to run this locally, the data was filtered by date, popularity, and number of reviews to get a set of movies ~15K that could be ingested in a reasonable time (this is configurable in the code).

## Quick Start (Local)

```bash
# Clone and enter the repo
git clone https://github.com/michaelkrot/rag-agent-duckdb.git
cd rag-agent-duckdb

# Install dependencies
pip install -r requirements.txt

#The agent requires a pre-built vector index of movie overviews.
#Run once (or when updating data):
python build_corpus.py

# Interactive REPL (recommended for demo)
python agent.py repl

# Single query
python agent.py query "What is RAG?"

# View usage statistics
python agent.py stats

# Show version
python agent.py --version```

<!--
## Docker instructions

#hiding the docker stuff for now as it's not quite there
### Build the docker image (needs work/hasn't been updates)
```bash
# Build the image
docker build -t rag-agent .

# Interactive REPL 
docker run -it --rm -v $(pwd)/data:/app/data rag-agent

# Single query
docker run -it --rm -v $(pwd)/data:/app/data rag-agent python agent.py query "What database are you using?"

# View stats
docker run -it --rm -v $(pwd)/data:/app/data rag-agent python agent.py stats```

-->

