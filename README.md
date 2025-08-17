# InsureSense 360 - AI-Powered Insurance Platform

InsureSense 360 is a comprehensive AI-powered insurance platform that combines advanced document processing, semantic search, and Retrieval-Augmented Generation (RAG) to provide intelligent insurance policy analysis and customer support.

## ✨ Features

### 🔍 **Document Processing**
- **Multi-format Support**: PDF, TXT, HTML, Markdown
- **LangChain Integration**: Advanced document loaders and text splitters
- **Intelligent Sectioning**: Automatic identification of policy sections (coverage, exclusions, terms, etc.)
- **IRDAI Integration**: Download and process insurance policies from regulatory sources

### 🗄️ **Vector Database**
- **Dual Backend Support**: FAISS (fast) and Chroma (persistent)
- **Semantic Search**: High-quality embeddings using sentence-transformers or OpenAI
- **Advanced Retrieval**: LangChain retrievers with MMR and similarity search
- **Batch Processing**: Efficient handling of large document collections

### 🤖 **RAG Chatbot**
- **LangChain RAG Chains**: ConversationalRetrievalChain and RetrievalQA
- **Conversation Memory**: Buffer and summary memory for context continuity
- **Multi-source Answers**: Grounded responses with policy citations
- **Follow-up Suggestions**: Intelligent question recommendations
- **Confidence Scoring**: Reliability metrics for responses

### 🌐 **API & Integration**
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **Real-time Processing**: Async document processing and search
- **Comprehensive Endpoints**: Upload, search, chat, and system management
- **CORS Support**: Frontend integration ready

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector        │    │   RAG           │
│   Processor     │───▶│   Store         │───▶│   Chatbot       │
│   (LangChain)   │    │   (FAISS/Chroma)│    │   (LangChain)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   OpenAI        │    │   Conversation  │
│   Backend       │    │   Integration   │    │   Memory        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd insurance-app

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_template.txt .env
# Edit .env with your OpenAI API key (optional)
```

### 2. **Environment Configuration**

```bash
# Required for enhanced RAG responses
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Vector database settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DIMENSION=384
FAISS_INDEX_PATH=data/faiss_index

# File storage paths
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
IRDAI_POLICIES_PATH=data/irdai_policies
```

### 3. **Testing Setup**

**IMPORTANT: Run tests before deployment to ensure everything works correctly.**

```bash
# Navigate to tests folder
cd tests

# Run all tests in order (see tests/README.md for details)
python test_system.py
python test_llm.py
python test_direct_llm.py
python test_together_llm.py
python test_api.py
python test_api_integration.py

# Or run all tests at once
python -m pytest tests/ -v
```

**Expected Results**: All tests should pass ✅

### 4. **Run the System**

```bash
# Start the server
python start_server.py

# Access the API
# - API Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
```

## 📚 API Endpoints

### **Document Management**
- `POST /upload/policy` - Upload and process insurance documents
- `POST /process/irdai` - Download and process IRDAI policies
- `POST /system/rebuild-index` - Rebuild vector index

### **Search & Retrieval**
- `POST /search/policies` - Semantic search across policies
- `GET /system/stats` - System statistics and health

### **RAG Chatbot**
- `POST /chat` - Chat with the AI assistant
- `GET /chat/suggestions` - Get suggested questions
- `GET /chat/history/{id}` - Get conversation history
- `DELETE /chat/history/{id}` - Clear conversation

### **System Management**
- `GET /health` - Health check
- `POST /system/export-embeddings` - Export embeddings for analysis

## 🔧 Configuration

### **LangChain Settings**
```python
# Document processing
chunk_size = 1000          # Text chunk size
chunk_overlap = 200        # Overlap between chunks

# Vector store
vector_store_type = "faiss"  # "faiss" or "chroma"
embedding_model = "all-MiniLM-L6-v2"

# RAG chatbot
use_memory = True          # Enable conversation memory
memory_type = "buffer"     # "buffer" or "summary"
```

### **Advanced Features**
- **Multi-vector Retrieval**: For complex queries requiring multiple perspectives
- **MMR Search**: Maximum Marginal Relevance for diverse results
- **Conversation Summary**: Long-term memory for extended conversations
- **Export Capabilities**: Embeddings and metadata export for analysis

## 📖 Usage Examples

### **1. Upload and Process Documents**

```python
import requests

# Upload a policy document
with open('policy.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload/policy', files=files)
    print(response.json())
```

### **2. Search Policies**

```python
# Search for coverage information
search_data = {
    "query": "What is covered under health insurance?",
    "top_k": 5,
    "policy_types": ["Health Insurance"]
}

response = requests.post('http://localhost:8000/search/policies', json=search_data)
results = response.json()
```

### **3. Chat with RAG Bot**

```python
# Ask questions about policies
chat_data = {
    "message": "What are the main exclusions in this policy?",
    "max_sources": 3
}

response = requests.post('http://localhost:8000/chat', json=chat_data)
answer = response.json()
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence_score']}")
```

## 🧪 Testing

### **Run Complete Test Suite**
```bash
python test_system.py
```

The test suite covers:
- ✅ Document processing with LangChain
- ✅ Vector store operations (FAISS/Chroma)
- ✅ RAG chatbot functionality
- ✅ Conversation memory
- ✅ Advanced LangChain features

### **Test Individual Components**
```python
# Test document processing
from src.document_processor import DocumentProcessor
processor = DocumentProcessor()
policies = processor.process_directory("data/irdai_policies")

# Test vector store
from src.vector_store import VectorStore
vector_store = VectorStore()
vector_store.add_policies_batch(policies)

# Test RAG chatbot
from src.rag_chatbot import RAGChatbot
chatbot = RAGChatbot(vector_store)
response = chatbot.chat("What does this policy cover?")
```

## 🔍 Troubleshooting

### **Common Issues**

1. **LangChain Import Errors**
   ```bash
   pip install --upgrade langchain langchain-community
   ```

2. **FAISS Installation Issues**
   ```bash
   # For CPU
   pip install faiss-cpu
   
   # For GPU (CUDA)
   pip install faiss-gpu
   ```

3. **OpenAI API Errors**
   - Verify API key in `.env`
   - Check API quota and billing
   - Use fallback HuggingFace embeddings

4. **Memory Issues**
   - Reduce chunk size
   - Use Chroma instead of FAISS for large datasets
   - Enable conversation summary memory

### **Performance Optimization**

- **Chunk Size**: Adjust based on document complexity (800-1500 characters)
- **Batch Processing**: Use `add_policies_batch()` for multiple documents
- **Vector Store**: FAISS for speed, Chroma for persistence
- **Embedding Model**: Balance between quality and speed

## 🚀 Azure Migration Path

### **Phase 1: Core Services**
- **Azure OpenAI**: Replace OpenAI API calls
- **Azure Cognitive Search**: Enhanced vector search capabilities
- **Azure Blob Storage**: Document storage and management

### **Phase 2: Advanced Features**
- **Azure ML Workspace**: Custom embedding models
- **Azure Functions**: Serverless processing
- **Azure Static Web Apps**: Frontend deployment

### **Phase 3: Enterprise Features**
- **Azure Form Recognizer**: Advanced document processing
- **Azure Translator**: Multilingual support
- **Azure Monitor**: Comprehensive logging and monitoring

## 🔒 Security & Best Practices

### **Data Protection**
- Secure API endpoints with authentication
- Encrypt sensitive policy data
- Implement rate limiting
- Regular security audits

### **Responsible AI**
- Bias detection in responses
- Content moderation
- Transparency in AI decisions
- User consent management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt

# Run linting
black src/
flake8 src/

# Run all tests at once (recommended)
python run_tests.py

# Or run tests individually (see tests/README.md for details)
cd tests
python test_system.py
python test_llm.py
python test_direct_llm.py
python test_together_llm.py
python test_api.py
python test_api_integration.py
```

## 🧪 Testing

### **Comprehensive Testing Suite**

The project includes a complete testing suite in the `tests/` folder:

- **System Tests**: Basic setup and dependency verification
- **LLM Tests**: Together AI integration and response generation
- **API Tests**: End-to-end API functionality testing
- **Integration Tests**: Complete workflow validation

### **Running Tests**

```bash
# Navigate to tests folder
cd tests

# Run all tests at once (recommended)
python run_tests.py

# Or run tests individually
python test_system.py
python test_llm.py
python test_direct_llm.py
python test_together_llm.py
python test_api.py
python test_api_integration.py

# Or run with pytest
python -m pytest tests/ -v
```

### **Test Results**

- ✅ **All tests pass**: System ready for deployment
- ⚠️ **Some tests fail**: Check configuration and dependencies
- ❌ **Multiple failures**: Review setup process

**See `tests/README.md` for detailed testing instructions and troubleshooting.**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@insuresense360.com

## 🗺️ Roadmap

### **Q1 2024**
- [x] LangChain integration
- [x] Advanced RAG chains
- [x] Conversation memory
- [x] Multi-vector retrieval

### **Q2 2024**
- [ ] Risk scoring models
- [ ] Policy matching algorithms
- [ ] Frontend dashboard
- [ ] Azure migration

### **Q3 2024**
- [ ] Multilingual support
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Enterprise features

---

**Built with ❤️ using LangChain, FastAPI, and modern AI technologies**

*InsureSense 360 - Making Insurance Intelligent*
