# AI Engineering — E-commerce Assistant

A multi-agent e-commerce shopping assistant. Users chat with the system to
get product recommendations, ask questions about items and reviews, manage
a shopping cart, and check/reserve warehouse stock — all handled by a team
of LLM agents coordinated with LangGraph.

## Architecture

```
Streamlit UI  --->  FastAPI backend  --->  LangGraph multi-agent system
                                            |
                                            |-- Coordinator agent
                                            |-- Product Q&A agent   --> Qdrant (hybrid search) + Cohere rerank
                                            |-- Shopping cart agent --> Postgres
                                            |-- Warehouse manager agent --> Postgres
```

- The **coordinator agent** interprets the user's request, plans which
  agent(s) to delegate to, and produces the final answer.
- The **product Q&A agent** answers product/review questions using RAG over
  Qdrant (hybrid dense + BM25 search, fused with RRF, reranked with Cohere).
- The **shopping cart agent** adds, removes, and lists items in the user's
  cart (Postgres-backed).
- The **warehouse manager agent** checks stock availability across
  warehouses and reserves items transactionally (Postgres-backed).


## Tech stack

- **LLMs**: OpenAI (chat + embeddings)
- **Agent orchestration**: LangGraph + LangChain
- **Structured output**: Pydantic tool-calling schemas
- **Vector search**: Qdrant (hybrid dense + BM25, RRF fusion)
- **Reranking**: Cohere
- **Observability**: LangSmith tracing on every LLM call, retrieval, and tool
- **Evaluation**: Ragas
- **Agent-tool protocol**: MCP (via FastMCP)
- **Backend**: FastAPI, Uvicorn, SSE streaming
- **Frontend**: Streamlit
- **Data stores**: PostgreSQL, Qdrant

## Getting started

1. fill in your API keys (OpenAI, Cohere, LangSmith).
2. Start everything with Docker Compose: uv sync && docker compose up --build
3. Open the chat UI at [http://localhost:8501](http://localhost:8501).

The API is available at `http://localhost:8000`, and the Qdrant dashboard at
`http://localhost:6333/dashboard`.
