# Reflective-RAG Research Assistant

## Overview
Standard Retrieval-Augmented Generation (RAG) pipelines often suffer from hallucinations or missed context due to their linear, one-shot nature. The **Reflective-RAG Research Assistant** solves this by implementing a cyclical, multi-agent workflow. Instead of generating a single summary, the system utilizes a "Researcher" and a "Fact-Checker" that iteratively debate and refine the output until it is strictly supported by the source document.

## The Agentic Workflow
This project is orchestrated using **LangGraph** to manage the state and cyclic routing between agents:

1. **Document Ingestion:** PDFs are processed locally. Text is chunked, embedded using Ollama, and stored in a local vector database.
2. **The Researcher (Agent 1):** Retrieves relevant chunks from the vector database based on the user's query and drafts an initial summary.
3. **The Fact-Checker (Agent 2):** Audits the draft strictly against the retrieved source chunks.
    * If claims are unsupported or missing, it generates a "Correction Note" (Reject).
    * If the draft is highly accurate, it approves the summary (Approve).
4. **The Loop:** If rejected, the LangGraph router sends the feedback back to the Researcher to rewrite the draft. This cycle repeats until the Fact-Checker approves or a maximum loop count is reached.

## Tech Stack
* **Orchestration:** LangGraph
* **Cloud LLM (Reasoning):** Gemini 1.5 Flash (via Google AI Studio) or Groq API
* **Local Embeddings:** Ollama (`nomic-embed-text`)
* **Vector Database:** FAISS or ChromaDB
* **Frontend UI:** Streamlit

## Project Structure
```text
reflective-rag/
├── app.py                 # Streamlit frontend UI
├── graph.py               # LangGraph orchestration (nodes, conditional edges, state)
├── agents.py              # System prompts and LLM execution for Researcher and Fact-Checker
├── retriever.py           # PDF ingestion, Ollama embedding generation, and Vector DB logic
├── requirements.txt       # Project dependencies
└── .env                   # Environment variables (API keys)