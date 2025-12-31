# RAG Agent with DuckDB & Open-Source LLMs

A lightweight, reproducible Retrieval-Augmented Generation (RAG) agent built for a data engineering portfolio.

## Project Goals
- Demonstrate end-to-end RAG pipeline (text → eventually quantitative data)
- Use DuckDB for vector storage and lightweight querying
- Keep everything Dockerized for easy demo
- Show provenance / logging of responses

## Current Stage
**v0 – Dummy Agent**  
A simple CLI that accepts a query, returns a hardcoded response, and logs the interaction.

## Quick Start (Local)

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

