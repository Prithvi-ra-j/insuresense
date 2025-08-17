#!/usr/bin/env python3
"""
Test Vector Store with Processed Policies
Simple test to verify vector store functionality
"""

import sys
import json
import logging
from pathlib import Path
from typing import List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processor import InsurancePolicy, PolicySection
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_store():
    """Test vector store functionality"""
    print("Testing Vector Store with Processed Policies")
    print("=" * 50)
    
    try:
        # Initialize vector store
        print("[+] Initializing vector store...")
        vector_store = VectorStore(vector_store_type="faiss")
        print("[+] Vector store initialized successfully")
        
        # Check statistics
        stats = vector_store.get_statistics()
        print(f"[+] Initial statistics:")
        print(f"    Total vectors: {stats.get('total_vectors', 0)}")
        print(f"    Total policies: {stats.get('total_policies', 0)}")
        
        # Test search with a simple query
        print("[+] Testing search functionality...")
        results = vector_store.search("What is covered under this policy?", top_k=3)
        print(f"[+] Search returned {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"[-] Error testing vector store: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("InsureSense 360 - Vector Store Test")
    print("=" * 50)
    
    success = test_vector_store()
    
    if success:
        print("\n[+] Vector store test completed successfully!")
        return 0
    else:
        print("\n[-] Vector store test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())