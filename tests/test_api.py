#!/usr/bin/env python3
"""
Test script for InsureSense 360 API
Tests all core endpoints to ensure they work correctly
"""

import requests
import json
import time
from pathlib import Path

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

def test_search_policies():
    """Test the policy search endpoint"""
    print("\nTesting policy search endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/search_policies?query=life insurance&max_results=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Policy search: {data.get('total_found', 0)} results found")
            return True
        else:
            print(f"❌ Policy search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Policy search error: {e}")
        return False

def test_chat():
    """Test the RAG chatbot endpoint"""
    print("\nTesting RAG chatbot endpoint...")
    try:
        request_data = {
            "message": "Can you recommend a life insurance policy for me?",
            "conversation_id": None
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat response: {data.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_policy_summary():
    """Test the policy summary endpoint"""
    print("\nTesting policy summary endpoint...")
    try:
        # Test with policy text
        request_data = {
            "policy_text": "This is a sample life insurance policy that provides comprehensive coverage for individuals aged 18-65. The policy includes death benefits, maturity benefits, and additional riders for critical illness and accidental death.",
            "summary_type": "comprehensive"
        }
        
        response = requests.post(f"{BASE_URL}/summarize_policy", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Policy summary: {len(data.get('summary', ''))} characters")
            return True
        else:
            print(f"❌ Policy summary failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Policy summary error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting InsureSense 360 API Tests")
    print("=" * 50)
    
    # Check if server is running
    print("Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding. Please start the server first:")
            print("   python src/main.py")
            return
    except:
        print("❌ Cannot connect to server. Please start the server first:")
        print("   python src/main.py")
        return
    
    print("✅ Server is running!")
    
    # Run all tests
    tests = [
        test_health_endpoint,
        test_dataset_stats,
        test_policy_recommendation,
        test_risk_score,
        test_search_policies,
        test_chat,
        test_policy_summary
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    print("\n📋 API Endpoints tested:")
    print("  ✅ GET  /health")
    print("  ✅ GET  /dataset/stats")
    print("  ✅ POST /recommend_policy")
    print("  ✅ POST /risk_score")
    print("  ✅ GET  /search_policies")
    print("  ✅ POST /chat")
    print("  ✅ POST /summarize_policy")

if __name__ == "__main__":
    main()
