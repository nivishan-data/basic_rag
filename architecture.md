# RAG Application Architecture

## System Architecture
```mermaid
graph TB
    subgraph Frontend["Frontend Layer"]
        ST[Streamlit UI]
        style ST fill:#A52A2A
    end

    subgraph Processing["Processing Layer"]
        subgraph RAG["RAG Engine"]
            PA[Primary Agent]
            RA[Reasoning Agent]
            style PA fill:#A52A2A
            style RA fill:#A52A2A
        end
        
        subgraph Models["AI Models"]
            LLM[Local Ollama LLM]
            EMB[HuggingFace Embeddings]
            style LLM fill:#A52A2A
            style EMB fill:#A52A2A
        end
        
        subgraph Ingestion["PDF Processing"]
            PL[PDF Loader]
            TS[Text Splitter]
            style PL fill:#A52A2A
            style TS fill:#A52A2A
        end
    end

    subgraph Storage["Storage Layer"]
        VDB[(ChromaDB<br/>Vector Store)]
        PDFs[(PDF Files)]
        style VDB fill:#A52A2A
        style PDFs fill:#A52A2A
    end

    %% Frontend to Processing connections
    ST --> PA

    %% Processing internal connections
    PA --> RA
    PA --> VDB
    RA --> LLM
    PL --> TS
    TS --> VDB
    VDB --> EMB

    %% Storage connections
    PDFs --> PL
```

## Process Flow
```mermaid
sequenceDiagram
    participant User
    participant ST as Streamlit UI
    participant PA as Primary Agent
    participant RA as Reasoning Agent
    participant VDB as ChromaDB
    participant LLM as Ollama LLM

    %% PDF Ingestion Flow
    rect rgb(165, 42, 42)
        Note over User,LLM: Document Ingestion Process
        User->>ST: Upload PDF
        ST->>VDB: Process Document
        VDB->>VDB: 1. Load PDF
        VDB->>VDB: 2. Split into Chunks
        VDB->>VDB: 3. Generate Embeddings
        VDB->>VDB: 4. Store Vectors
    end

    %% Query Flow
    rect rgb(165, 42, 42)
        Note over User,LLM: Query Process
        User->>ST: Submit Question
        ST->>PA: Forward Query
        PA->>VDB: Search Similar Docs
        VDB-->>PA: Return Relevant Chunks
        PA->>RA: Generate Response
        RA->>LLM: Get Answer
        LLM-->>RA: Return Answer
        RA-->>PA: Process Answer
        PA-->>ST: Return Response
        ST-->>User: Display Answer
    end
```

## Component Details

### Frontend Layer
- **Streamlit UI**: Web-based user interface that provides:
  - Chat-style Q&A interface
  - Chat history management
  - Informative sidebar
  - Error handling and user feedback

### Processing Layer
- **PDF Processing**:
  - DirectoryLoader: Loads PDFs from specified directory
  - RecursiveCharacterTextSplitter: Splits documents into chunks (1000 chars, 200 overlap)

- **RAG Engine**:
  - Primary Agent: Orchestrates the RAG process
  - Reasoning Agent: Generates responses using context
  - Async processing with event loop management

- **AI Models**:
  - Ollama LLM: Local language model for response generation
  - HuggingFace Embeddings: sentence-transformers/all-mpnet-base-v2 for vector embeddings

### Storage Layer
- **ChromaDB**: Vector database storing:
  - Document chunks
  - Vector embeddings
  - Metadata
- **PDF Files**: Source documents in PDF format

### Key Features
1. **Asynchronous Processing**: Uses asyncio for non-blocking operations
2. **Error Handling**: Comprehensive error management and logging
3. **Modular Design**: Clear separation of concerns between components
4. **Scalable Architecture**: Components can be upgraded independently
5. **Local Processing**: Uses local LLM through Ollama for privacy and speed

### Data Flow
1. **Document Processing**:
   ```
   PDF → Text Chunks → Embeddings → Vector Store
   ```

2. **Query Processing**:
   ```
   Query → Vector Search → Context Retrieval → LLM Processing → Response
