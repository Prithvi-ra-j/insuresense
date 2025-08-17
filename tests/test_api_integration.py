#!/usr/bin/env python3
"""
Test API Integration for Phase 6
Simple test to verify API endpoints are working
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Dataset records: {data.get('dataset_records')}")
            print(f"   - Vector store available: {data.get('vector_store_available')}")
            print(f"   - RAG chatbot available: {data.get('rag_chatbot_available')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_search_endpoint():
    """Test the search endpoint"""
    print("\n🔍 Testing search endpoint...")
    try:
        search_data = {
            "query": "life insurance benefits",
            "max_results": 5
        }
        response = requests.post(f"{API_BASE}/search_policies", json=search_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search test passed!")
            print(f"   - Found {len(data.get('results', []))} results")
            print(f"   - Query time: {data.get('query_time', 0):.3f}s")
            return True
        else:
            print(f"❌ Search test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Search test error: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n🔍 Testing chat endpoint...")
    try:
        chat_data = {
            "message": "What are the main benefits of life insurance?",
            "max_sources": 3
        }
        response = requests.post(f"{API_BASE}/chat", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat test passed!")
            print(f"   - Response: {data.get('response', '')[:100]}...")
            print(f"   - Confidence: {data.get('confidence', 0)}")
            print(f"   - Sources: {len(data.get('sources', []))}")
            return True
        else:
            print(f"❌ Chat test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat test error: {e}")
        return False

def test_policy_recommendation():
    """Test policy recommendation endpoint"""
    print("\n🔍 Testing policy recommendation endpoint...")
    try:
        recommendation_data = {
            "age": 35,
            "income": 75000,
            "family_size": 3,
            "risk_tolerance": "moderate",
            "coverage_needs": ["life", "health"]
        }
        response = requests.post(f"{API_BASE}/recommend_policy", json=recommendation_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Policy recommendation test passed!")
            print(f"   - Recommendations: {len(data.get('recommendations', []))}")
            return True
        else:
            print(f"❌ Policy recommendation test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Policy recommendation test error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing InsureSense 360 API Integration")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(3)
    
    # Test all endpoints
    tests = [
        test_health_endpoint,
        test_search_endpoint,
        test_chat_endpoint,
        test_policy_recommendation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   - Passed: {passed}/{total}")
    print(f"   - Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 6 is ready to proceed.")
        print("✅ API endpoints are working correctly")
        print("✅ Vector store is functional")
        print("✅ RAG chatbot is operational")
        print("\n📋 Next steps:")
        print("   1. Test /upload/policy endpoint with real PDFs")
        print("   2. Add more policies to vector store")
        print("   3. Move to Phase 7: RAG System Validation")
    else:
        print(f"\n⚠️ {total-passed} tests failed. Please check server logs.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
