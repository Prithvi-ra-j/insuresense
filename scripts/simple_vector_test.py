#!/usr/bin/env python3
"""
Simple Vector Store Test
Test basic vector store functionality without adding policies
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_vector_store():
    """Test basic vector store functionality"""
    print("Testing Basic Vector Store Functionality")
    print("=" * 50)
    
    try:
        # Initialize vector store
        print("[+] Initializing vector store...")
        vector_store = VectorStore(vector_store_type="faiss")
        print("[+] Vector store initialized successfully")
        
        # Check statistics
        stats = vector_store.get_statistics()
        print(f"[+] Statistics:")
        print(f"    Vector Store Type: {stats.get('vector_store_type', 'Unknown')}")
        print(f"    Total Vectors: {stats.get('total_vectors', 0)}")
        print(f"    Total Policies: {stats.get('total_policies', 0)}")
        print(f"    Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        
        # Test simple search
        print("[+] Testing simple search...")
        results = vector_store.search("test query", top_k=1)
        print(f"[+] Search returned {len(results)} results")
        
        print("\n[+] Basic vector store test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[-] Error in basic vector store test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("InsureSense 360 - Simple Vector Store Test")
    print("=" * 50)
    
    success = test_basic_vector_store()
    
    if success:
        print("\n[+] Basic vector store test completed successfully!")
        return 0
    else:
        print("\n[-] Basic vector store test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())