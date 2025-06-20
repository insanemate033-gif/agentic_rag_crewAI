# Agentic RAG with CrewAI & OpenAI

This project implements an advanced Retrieval-Augmented Generation (RAG) system orchestrated by CrewAI, capable of answering user queries by leveraging both a local PDF knowledge base and real-time web search capabilities. The system is designed to provide concise, coherent, and grounded responses.

## Table of Contents
1.  [Features](#features)
2.  [Architecture & Flow](#architecture--flow)
3.  [Tech Stack](#tech-stack)
4.  [API Keys & Setup](#api-keys--setup)
5.  [Installation & Setup](#installation--setup)
6.  [How to Run the Application](#how-to-run-the-application)
7.  [Troubleshooting Tips](#troubleshooting-tips)
8.  [Repository Link](#repository-link)

## Features
* **Agentic Workflow:** Orchestrates multiple specialized AI agents for complex reasoning, from routing queries to synthesizing final answers.
* **Dual-Source RAG:** Answers queries using both a private, local PDF knowledge base and real-time web search capabilities for comprehensive results.
* **Real-time Web Search:** Integrates `SerperDevTool` for up-to-date information retrieval from the internet.
* **Web Scraping (Optional/Advanced):** Employs `FireCrawlWebSearchTool` to scrape content from specific URLs found during web searches, allowing for deeper information extraction.
* **Robust LLM Integration:** Configured to use OpenAI's powerful `gpt-4o` model via LangChain.
* **Interactive UI:** Powered by Streamlit for an intuitive chat interface.
* **PDF Document Search:** Utilizes `DocumentSearchTool` to extract and semantically search within uploaded PDF files using Qdrant (in-memory for this setup) and Sentence Transformers for embeddings.

## Architecture & Flow

The system employs a sequential multi-agent architecture to process user queries:

1.  **Router Agent:** Analyzes the incoming user query and decides whether to route it to the internal vectorstore (for PDF knowledge) or to a web search.
2.  **Retriever Agent:** Based on the router's decision, it uses either the `DocumentSearchTool` (for PDF) or the `SerperDevTool` (for web search) to retrieve relevant information. If a relevant URL is found via Serper, it can optionally use `FireCrawlWebSearchTool` to scrape detailed content from that URL.
3.  **Grader Agent:** Assesses the relevance of the retrieved content to the original question.
4.  **Hallucination Grader Agent:** Checks if the answer generated from the retrieved content is grounded in the facts and aligns with the question, filtering out potential hallucinations.
5.  **Answer Grader Agent:** Evaluates the usefulness of the final answer. If the answer is not relevant, it can trigger another web search as a fallback.
6.  **Response Synthesizer Agent:** Synthesizes the final, verified information into a concise and coherent response for the user.

[YOUR_ARCHITECTURE_DIAGRAM_IMAGE_LINK_HERE]

**Note:** Please replace `[YOUR_ARCHITECTURE_DIAGRAM_IMAGE_LINK_HERE]` with the direct URL to your architecture diagram image after uploading it (e.g., to GitHub Issues, Imgur, etc.).

## Tech Stack

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web UI (`app.py`).
* **CrewAI:** The framework for orchestrating AI agents.
* **LangChain:** The underlying framework for interacting with LLMs and building agent components (`langchain-openai`, `langchain-community`).
* **OpenAI:** The Large Language Model provider (`gpt-4o`).
* **SerperDevTool:** A tool from `crewai-tools` for conducting Google searches.
* **Custom `FireCrawlWebSearchTool`:** (Defined in `src/agentic_rag/tools/custom_tool.py`) For scraping web page content from provided URLs.
* **Custom `DocumentSearchTool`:** (Defined in `src/agentic_rag/tools/custom_tool.py`) For RAG over local PDF documents using Qdrant (in-memory) and `sentence-transformers` for embeddings.
* **`python-dotenv`:** For managing environment variables (`.env`).
* **`markitdown`:** For PDF text extraction in `DocumentSearchTool`.
* **`chonkie`:** For semantic text chunking.
* **`qdrant-client`:** For vector database operations (in-memory).
* **`sentence-transformers`:** For generating embeddings (used by `DocumentSearchTool`).
* **`pydantic` & `rich`:** Core dependencies for data validation and rich console output (managed by CrewAI/LangChain).

## API Keys & Setup

This project requires API keys for accessing external services. These keys should be stored in a `.env` file at the root of your project directory (`/Users/avinashgohite/Desktop/agentic_RAG/.env`).

Create a file named `.env` and add your keys as follows:

```ini
OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY"
SERPER_API_KEY="YOUR_SERPER_API_KEY"
FIRECRAWL_API_KEY="YOUR_FIRECRAWL_API_KEY"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
MODEL=gpt-4o
