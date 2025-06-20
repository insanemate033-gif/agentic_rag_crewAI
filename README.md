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

OPENAI_API_KEY="sk-proj-YOUR_OPENAI_API_KEY"
SERPER_API_KEY="YOUR_SERPER_API_KEY"
FIRECRAWL_API_KEY="YOUR_FIRECRAWL_API_KEY"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
MODEL=gpt-4o

## API Keys & Setup

**Where to get your API Keys:**
* **OpenAI API Key:** Visit [OpenAI Platform](https://platform.openai.com/account/api-keys).
* **Serper API Key:** Visit [Serper.dev](https://serper.dev/api-key).
* **Firecrawl API Key:** Visit [Firecrawl.dev](https://www.firecrawl.dev/api-keys).
* **Tavily API Key:** Visit [Tavily AI](https://app.tavily.com/home).
* **Groq API Key:** Visit [Groq Console](https://console.groq.com/keys).

## Installation & Setup

Follow these steps to get the project set up on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/insanemate033-gif/agentic_rag_crewAI.git](https://github.com/insanemate033-gif/agentic_rag_crewAI.git)
    cd agentic_rag_crewAI
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    * **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
    * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    Your terminal prompt should now show `(.venv)` at the beginning, indicating the virtual environment is active.

4.  **Install Dependencies:**
    First, ensure you have a `requirements.txt` file in your project root with the following content:

    ```plaintext
    # requirements.txt content
    streamlit
    crewai
    crewai-tools
    langchain-openai
    langchain-groq
    python-dotenv
    langchain-community
    markitdown
    chonkie
    qdrant-client
    firecrawl-py
    sentence-transformers
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Verify Project Structure:**
    Ensure your project structure looks like this (crucial for imports):
    ```
    agentic_RAG/
    ├── app.py
    ├── .env
    ├── .venv/
    ├── assets/
    │   └── crewai.png
    ├── knowledge/
    │   └── dspy.pdf # (Optional, if you have a default PDF)
    └── src/
        ├── __init__.py
        └── agentic_rag/
            ├── __init__.py
            ├── tools/
            │   ├── __init__.py
            │   └── custom_tool.py
            └── ... (other src files like crew.py, main.py, demo_llama3.2.ipynb)
    ```
    Ensure all `__init__.py` files are present as empty files in their respective directories.

## How to Run the Application

Once installed and configured:

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the project's root directory** in your terminal (where `app.py` is located).
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will open a new tab in your web browser with the interactive chat UI.

4.  **Interact with the UI:**
    * In the Streamlit sidebar, use "Choose a PDF file" to upload a document for RAG.
    * Type your questions in the chat input box at the bottom. The system will use its agents and tools to provide responses.

## Troubleshooting Tips

* **`ModuleNotFoundError: No module named 'src.agentic_rag.tools'`**:
    * Ensure all `__init__.py` files are present in `src/`, `src/agentic_rag/`, and `src/agentic_rag/tools/`. They must be empty files.
    * Verify the exact spelling and casing of all directory names (`src`, `agentic_rag`, `tools`).
* **`ImportError: cannot import name 'BaseTool' from 'crewai.tools'` (or similar for other modules)**:
    * Ensure `src/agentic_rag/tools/custom_tool.py` has the correct `BaseTool` import: `from crewai.tools import BaseTool` (as determined during our debugging).
    * Restart your Jupyter Kernel or terminal session after modifying any Python files.
* **`RecursionError: maximum recursion depth exceeded`**:
    * This often happens with excessive logging. Ensure `verbose=False` for all `Agent` and `Crew` instantiations in your `app.py`.
    * You can temporarily increase Python's recursion limit for debugging, but it's not a permanent solution for logical loops: `import sys; sys.setrecursionlimit(5000)`.
    * Review agent goals and tool descriptions. Ensure they provide clear instructions and escape conditions (e.g., "if information not found, respond with X"). Ambiguous goals can cause agents to loop.
* **`BadRequestError: LLM Provider NOT provided` or API key issues**:
    * Double-check your `.env` file for correct variable names (`OPENAI_API_KEY`, `SERPER_API_KEY`, etc.) and accurate keys.
    * Ensure `load_dotenv()` is called early in your `app.py`.
    * Confirm your `llm` setup in `app.py` uses `ChatOpenAI` and `os.getenv("OPENAI_API_KEY")`.
* **General Issues**:
    * Always **restart your Streamlit app** (Ctrl+C in terminal, then `streamlit run app.py` again) after making any code changes.
    * Ensure your virtual environment is **active and only active** when running `streamlit run`. Conflicts with other Python environments (like Conda `(base)`) are a common cause of unexpected errors.

## Repository Link

This project is hosted on GitHub:
[https://github.com/insanemate033-gif/agentic_rag_crewAI](https://github.com/insanemate033-gif/agentic_rag_crewAI)
