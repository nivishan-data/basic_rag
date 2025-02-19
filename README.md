# Rag Model

# Document Q&A Bot

Document Q&A Bot is a Retrieval-Augmented Generation (RAG) application that allows you to ask questions about your documents. The application uses [PydanticAI](https://ai.pydantic.dev/) for its agent framework, a Chroma vector store for document retrieval, and provides two user interfaces built with Gradio and Streamlit.

## Features

- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with LLM reasoning to generate context-aware answers.
- **Multiple Interfaces:** Interact via a Gradio web interface or a Streamlit chat interface.
- **Local LLM Integration:** Uses a locally hosted Ollama model via PydanticAI.
- **Vector Store:** Uses LangChain’s Chroma vector store and HuggingFace embeddings to manage and search through ingested document data.

## Installation

### Prerequisites

- Python 3.11 (or higher)
- pip
- A virtual environment (recommended)

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/your_repository_name.git
   cd your_repository_name

2. **Create and Activate a Virtual Environment:**

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**

    pip install -r requirements.txt

4. **Configure Environment Variables:**
    REASONING_MODEL_ID=deepseek-r1:7b-8k
    TOOL_MODEL_ID=llama3.2:latest

5. **Populate the Vector Store:**
    Ensure that your vector store (e.g., the chroma_db2 directory) is populated with document data. Use your ingestion script (e.g., ingest.py) to load and process your PDFs if needed.

6. **Running the Streamlit App**
    1. streamlit run streamlit_app.py or venv/bin/streamlit run streamlit_app.py
    2. Open the provided local URL (e.g., http://localhost:8501) in your browser to use the chat interface.


