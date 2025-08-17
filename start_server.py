#!/usr/bin/env python3
"""
Start script for InsureSense 360 with LangChain integration
Creates necessary directories and starts the FastAPI server
"""

import uvicorn
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

def main():
    """Start the InsureSense 360 server"""
    print("🚀 Starting InsureSense 360 with LangChain Integration...")
    print("=" * 60)
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        template_file = Path("env_template.txt")
        if template_file.exists():
            with open(template_file, 'r') as f:
                template_content = f.read()
            with open(env_file, 'w') as f:
                f.write(template_content)
            print("✓ Created .env file from template")
            print("📝 Please edit .env file with your configuration")
            print("   - Add your OpenAI API key for enhanced RAG responses")
            print("   - Configure embedding models and vector store settings")
        else:
            print("⚠️  env_template.txt not found. Using default configuration.")
            print("📝 You can create a .env file manually with your settings")
    
    # Create necessary directories
    directories = [
        "data/uploads",           # For uploaded documents
        "data/processed",         # For processed policy JSONs
        "data/irdai_policies",    # For IRDAI policy downloads
        "data/faiss_index",       # For FAISS vector index
        "data/chroma_db",         # For Chroma vector database
        "data/embeddings"         # For exported embeddings
    ]
    
    print("\n📁 Creating necessary directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Check for LangChain installation
    try:
        import langchain
        print(f"\n🔗 LangChain version: {langchain.__version__}")
    except ImportError:
        print("\n❌ LangChain not found. Please install with:")
        print("   pip install -r requirements.txt")
        return
    
    # Check for OpenAI API key
  
    # Check for Together API key
    if os.getenv("TOGETHER_API_KEY"):
        print("✅ Together API key configured")
    else:
        print("⚠️  Together API key not configured")
        print("   - Add TOGETHER_API_KEY to .env for enhanced responses")
    
    # Check for embedding models
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence transformers available")
    except ImportError:
        print("⚠️  Sentence transformers not available")
        print("   - Install with: pip install sentence-transformers")
    
    # Check for vector store backends
    try:
        import faiss
        print("✅ FAISS vector store available")
    except ImportError:
        print("⚠️  FAISS not available")
        print("   - Install with: pip install faiss-cpu")
    
    try:
        import chromadb
        print("✅ Chroma vector store available")
    except ImportError:
        print("⚠️  Chroma not available")
        print("   - Install with: pip install chromadb")
    
    print("\n" + "=" * 60)
    print("🌐 Starting FastAPI server...")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("   - Interactive API: http://localhost:8000/redoc")
    print("\n💡 Quick Start:")
    print("   1. Upload a policy: POST /upload/policy")
    print("   2. Process IRDAI policies: POST /process/irdai")
    print("   3. Search policies: POST /search/policies")
    print("   4. Chat with RAG bot: POST /chat")
    print("   5. View system stats: GET /system/stats")
    print("\n🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check if port 8000 is available")
        print("   2. Verify all dependencies are installed")
        print("   3. Check .env configuration")
        print("   4. Run: python test_system.py")

if __name__ == "__main__":
    main()
