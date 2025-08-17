# InsureSense 360 - Complete Process Map

This document provides a comprehensive overview of the entire InsureSense 360 workflow, from policy processing to vector store integration and API usage.

## Overall Architecture

```mermaid
graph TD
    A[Insurance Policies] --> B[Document Processing]
    B --> C[Policy Extraction]
    C --> D[Structured Data]
    D --> E[Vector Store]
    E --> F[Semantic Search]
    E --> G[RAG Chatbot]
    F --> H[API Endpoints]
    G --> H
    H --> I[User Interface]
    
    subgraph "Data Processing Pipeline"
        A
        B
        C
        D
    end
    
    subgraph "AI/ML Engine"
        E
        F
        G
    end
    
    subgraph "Application Layer"
        H
        I
    end
```

## 1. Data Processing Pipeline

### 1.1 Policy Ingestion
```mermaid
graph LR
    A[PDF Policies] --> B[Document Processor]
    B --> C[Text Extraction]
    C --> D[Section Identification]
    D --> E[Policy Structuring]
    E --> F[JSON Output]
    
    subgraph "Policy Sources"
        A
    end
    
    subgraph "Processing Steps"
        B
        C
        D
        E
    end
    
    subgraph "Output"
        F
    end
```

### 1.2 Enhanced Information Extraction
```mermaid
graph LR
    A[Policy JSON] --> B[Enhanced Extractor]
    B --> C[Benefits Analysis]
    B --> D[Exclusions Analysis]
    B --> E[Premium Information]
    B --> F[Terms & Conditions]
    B --> G[Renewal Terms]
    B --> H[Claims Process]
    C --> I[Structured Data]
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
    
    subgraph "Input"
        A
    end
    
    subgraph "Extraction Process"
        B
        C
        D
        E
        F
        G
        H
    end
    
    subgraph "Output"
        I
    end
```

## 2. Vector Store Integration

### 2.1 Data to Embeddings Flow
```mermaid
graph LR
    A[Policy Sections] --> B[Text Splitting]
    B --> C[Embedding Model]
    C --> D[Vector Representations]
    D --> E[FAISS/Chroma Store]
    
    subgraph "Preprocessing"
        A
        B
    end
    
    subgraph "Embedding Generation"
        C
        D
    end
    
    subgraph "Storage"
        E
    end
```

### 2.2 Search and Retrieval
```mermaid
graph LR
    A[User Query] --> B[Query Embedding]
    B --> C[Similarity Search]
    C --> D[Relevant Documents]
    D --> E[Confidence Scoring]
    E --> F[Ranked Results]
    
    subgraph "Query Processing"
        A
        B
    end
    
    subgraph "Vector Search"
        C
        D
        E
    end
    
    subgraph "Output"
        F
    end
```

## 3. RAG Chatbot System

### 3.1 Chat Flow
```mermaid
graph LR
    A[User Message] --> B[Query Analysis]
    B --> C[Vector Store Search]
    C --> D[Context Retrieval]
    D --> E[LLM Response Generation]
    E --> F[Answer + Sources]
    
    subgraph "Input"
        A
    end
    
    subgraph "Processing"
        B
        C
        D
        E
    end
    
    subgraph "Output"
        F
    end
```

### 3.2 Conversation Memory
```mermaid
graph LR
    A[Chat History] --> B[Memory Buffer]
    B --> C[Context Preservation]
    C --> D[Follow-up Tracking]
    D --> E[Personalized Responses]
    
    subgraph "Memory Components"
        A
        B
        C
        D
    end
    
    subgraph "Benefits"
        E
    end
```

## 4. API Endpoints Architecture

### 4.1 Core Endpoints
```mermaid
graph LR
    A[API Gateway] --> B[Health Check]
    A --> C[Policy Upload]
    A --> D[Policy Processing]
    A --> E[Semantic Search]
    A --> F[Chat Interface]
    A --> G[Policy Recommendations]
    A --> H[Risk Assessment]
    A --> I[Policy Summarization]
    
    subgraph "API Endpoints"
        B
        C
        D
        E
        F
        G
        H
        I
    end
    
    subgraph "Entry Point"
        A
    end
```

### 4.2 Data Flow Through API
```mermaid
graph LR
    A[HTTP Request] --> B[Request Validation]
    B --> C[Business Logic]
    C --> D[Vector Store Access]
    D --> E[Data Processing]
    E --> F[Response Formatting]
    F --> G[HTTP Response]
    
    subgraph "Request Handling"
        A
        B
        C
    end
    
    subgraph "Processing"
        D
        E
    end
    
    subgraph "Response"
        F
        G
    end
```

## 5. Complete End-to-End Workflow

### 5.1 Policy Processing Workflow
```mermaid
graph TD
    A[Raw PDF Policies] --> B[Document Processing]
    B --> C[Text Extraction]
    C --> D[Section Analysis]
    D --> E[Policy Structuring]
    E --> F[JSON Storage]
    F --> G[CSV Export]
    G --> H[Vector Store Indexing]
    
    subgraph "Ingestion Phase"
        A
        B
        C
    end
    
    subgraph "Processing Phase"
        D
        E
        F
    end
    
    subgraph "Export Phase"
        G
        H
    end
```

### 5.2 User Interaction Workflow
```mermaid
graph TD
    A[User Query] --> B[API Endpoint]
    B --> C[Vector Store Search]
    C --> D[Relevant Policy Sections]
    D --> E[RAG Processing]
    E --> F[LLM Response]
    F --> G[Formatted Answer]
    G --> H[User Interface]
    
    subgraph "Input"
        A
    end
    
    subgraph "Processing"
        B
        C
        D
        E
        F
    end
    
    subgraph "Output"
        G
        H
    end
```

## 6. System Components Overview

### 6.1 Core Modules
```mermaid
graph LR
    A[Document Processor] --> B[Policy Objects]
    C[Vector Store] --> D[Embeddings]
    E[RAG Chatbot] --> F[Conversational AI]
    G[API Endpoints] --> H[REST Interface]
    
    subgraph "Data Processing"
        A
        B
    end
    
    subgraph "AI/ML Engine"
        C
        D
        E
        F
    end
    
    subgraph "Application Layer"
        G
        H
    end
```

### 6.2 Data Flow Architecture
```mermaid
graph LR
    A[Policy PDFs] --> B[Processing Pipeline]
    B --> C[Structured Data]
    C --> D[Vector Store]
    D --> E[Search Engine]
    D --> F[Chat Engine]
    E --> G[API Responses]
    F --> G
    G --> H[User Applications]
    
    subgraph "Data Sources"
        A
    end
    
    subgraph "Processing Engine"
        B
        C
        D
    end
    
    subgraph "AI Services"
        E
        F
    end
    
    subgraph "Output"
        G
        H
    end
```

## 7. Current Implementation Status

### 7.1 Completed Components
```mermaid
graph LR
    A[✓ Document Processor] --> B[✓ Policy Extraction]
    B --> C[✓ JSON Storage]
    C --> D[✓ CSV Export]
    D --> E[Vector Store Ready]
    E --> F[API Endpoints]
    F --> G[User Interface Ready]
    
    style A fill:#90EE90
    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#FFA500
    style F fill:#FFA500
    style G fill:#FFA500
```

### 7.2 Next Steps
```mermaid
graph LR
    A[Add Policies to Vector Store] --> B[Enable Semantic Search]
    B --> C[Activate RAG Chatbot]
    C --> D[Full API Integration]
    D --> E[User Testing]
    
    style A fill:#FFA500
    style B fill:#FFA500
    style C fill:#FFA500
    style D fill:#FFA500
    style E fill:#FFA500
```

## 8. Data Flow Summary

### 8.1 Policy Processing Flow
1. **Input**: PDF policy documents from `lic_policies/` directory
2. **Processing**: 
   - Document parsing using LangChain loaders
   - Text extraction and section identification
   - Policy structuring into InsurancePolicy objects
3. **Storage**: 
   - Individual JSON files in `data/processed/`
   - Enhanced CSV in `data/extracted/enhanced_lic_policies.csv`
4. **Vectorization**: 
   - Policy sections converted to embeddings
   - Stored in FAISS/Chroma vector store
5. **Usage**: 
   - Semantic search via API endpoints
   - RAG chatbot context retrieval

### 8.2 User Query Flow
1. **Input**: User question through API or chat interface
2. **Processing**:
   - Query embedding generation
   - Similarity search in vector store
   - Context retrieval from relevant policies
3. **Response**:
   - LLM response generation with policy context
   - Source citations and confidence scoring
   - Formatted output for user consumption

## 9. System Capabilities

### 9.1 Current Features
- ✅ Process 10+ LIC policy PDFs
- ✅ Extract detailed policy information (benefits, exclusions, premiums, etc.)
- ✅ Generate structured JSON and CSV outputs
- ✅ Vector store ready for semantic search
- ✅ API endpoints for all core functionality
- ✅ RAG chatbot framework implemented

### 9.2 Upcoming Enhancements
- 🔄 Add processed policies to vector store
- 🔄 Enable full semantic search capabilities
- 🔄 Activate advanced RAG chatbot features
- 🔄 Implement conversation memory persistence
- 🔄 Add multi-language support
- 🔄 Integrate with Azure services

This comprehensive process map shows how all components of InsureSense 360 work together to provide an intelligent insurance policy analysis platform.