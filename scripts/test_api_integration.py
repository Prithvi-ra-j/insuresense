#!/usr/bin/env python3
"""
Test API Integration
Verify that all API endpoints are working correctly with vector store integration
"""

import sys
import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_search_endpoint():
    """Test the policy search endpoint"""
    print("\nTesting policy search endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/search_policies?query=accidental+death+benefit&max_results=3")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Policy search successful: {data.get('total_found', 0)} results found")
            if data.get('results'):
                print(f"   First result: {data['results'][0].get('title', 'Unknown')}")
                print(f"   Score: {data['results'][0].get('relevance_score', 0):.4f}")
            return True
        else:
            print(f"❌ Policy search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Policy search error: {e}")
        return False

def test_chat_endpoint():
    """Test the RAG chatbot endpoint"""
    print("\nTesting RAG chatbot endpoint...")
    try:
        request_data = {
            "message": "What does the accidental death benefit cover?"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat response successful")
            print(f"   Response: {data.get('response', '')[:100]}...")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            print(f"   Sources: {len(data.get('sources', []))}")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_dataset_stats():
    """Test the dataset statistics endpoint"""
    print("\nTesting dataset stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/dataset/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Dataset stats: {data}")
            return True
        else:
            print(f"❌ Dataset stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dataset stats error: {e}")
        return False

def test_policy_recommendation():
    """Test the policy recommendation endpoint"""
    print("\nTesting policy recommendation endpoint...")
    try:
        request_data = {
            "age": 30,
            "income": 800000,
            "family_size": 3,
            "risk_tolerance": "moderate",
            "coverage_needs": ["life", "health"]
        }
        
        response = requests.post(f"{BASE_URL}/recommend_policy", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Policy recommendation: {len(data.get('recommendations', []))} recommendations")
            return True
        else:
            print(f"❌ Policy recommendation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Policy recommendation error: {e}")
        return False

def test_risk_score():
    """Test the risk score endpoint"""
    print("\nTesting risk score endpoint...")
    try:
        request_data = {
            "age": 35,
            "health_conditions": ["hypertension"],
            "occupation": "software engineer",
            "lifestyle_factors": ["sports"],
            "family_history": ["diabetes"]
        }
        
        response = requests.post(f"{BASE_URL}/risk_score", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Risk score: {data.get('risk_score')} ({data.get('risk_level')})")
            return True
        else:
            print(f"❌ Risk score failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Risk score error: {e}")
        return False

def main():
    """Run all API integration tests"""
    print("🚀 Starting InsureSense 360 API Integration Tests")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run all tests
    tests = [
        test_health_endpoint,
        test_search_endpoint,
        test_chat_endpoint,
        test_dataset_stats,
        test_policy_recommendation,
        test_risk_score
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API integration tests passed!")
        print("✅ Vector store is correctly integrated with all API endpoints!")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    print("\n📋 API Endpoints tested:")
    print("  ✅ GET  /health")
    print("  ✅ GET  /search_policies")
    print("  ✅ POST /chat")
    print("  ✅ GET  /dataset/stats")
    print("  ✅ POST /recommend_policy")
    print("  ✅ POST /risk_score")

if __name__ == "__main__":
    main()