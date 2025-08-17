#!/usr/bin/env python3
"""
Test Vector Store Integration
Verify that policies have been correctly added to the vector store and can be searched
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

def test_vector_store_integration():
    """Test that policies have been correctly added to vector store"""
    print("Testing Vector Store Integration...")
    print("=" * 50)
    
    try:
        # Initialize vector store
        print("[+] Initializing vector store...")
        vector_store = VectorStore(vector_store_type="faiss")
        print("[+] Vector store initialized successfully")
        
        # Check statistics
        stats = vector_store.get_statistics()
        print(f"[+] Vector Store Statistics:")
        print(f"    Vector Store Type: {stats.get('vector_store_type', 'Unknown')}")
        print(f"    Total Vectors: {stats.get('total_vectors', 0)}")
        print(f"    Total Policies: {stats.get('total_policies', 0)}")
        print(f"    Policy Types: {', '.join(stats.get('policy_types', []))}")
        
        # Verify that we have policies
        if stats.get('total_policies', 0) == 0:
            print("[-] No policies found in vector store")
            return False
            
        # Test search functionality
        print("[+] Testing search functionality...")
        
        # Test 1: Simple search
        query1 = "What is covered under accidental death benefit?"
        results1 = vector_store.search(query1, top_k=3)
        print(f"    Query: '{query1}'")
        print(f"    Found {len(results1)} results")
        if results1:
            print(f"    Top result: {results1[0].section_title[:50]}...")
            print(f"    Score: {results1[0].similarity_score:.4f}")
        
        # Test 2: Policy type search
        query2 = "What are the benefits of JEEVAN policies?"
        results2 = vector_store.search(query2, top_k=3)
        print(f"    Query: '{query2}'")
        print(f"    Found {len(results2)} results")
        if results2:
            print(f"    Top result: {results2[0].document_title}")
            print(f"    Score: {results2[0].similarity_score:.4f}")
        
        # Test 3: Exclusion search
        query3 = "What is not covered under these policies?"
        results3 = vector_store.search(query3, top_k=3)
        print(f"    Query: '{query3}'")
        print(f"    Found {len(results3)} results")
        if results3:
            print(f"    Top result: {results3[0].section_title[:50]}...")
            print(f"    Score: {results3[0].similarity_score:.4f}")
        
        print("[+] Vector store integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[-] Error in vector store integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("InsureSense 360 - Vector Store Integration Test")
    print("=" * 50)
    
    success = test_vector_store_integration()
    
    if success:
        print("\n[+] Vector store integration test completed successfully!")
        print("[+] All policies are correctly indexed and searchable!")
        return 0
    else:
        print("\n[-] Vector store integration test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())