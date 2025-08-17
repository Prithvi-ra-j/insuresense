#!/usr/bin/env python3
"""
Test script to verify LLM configuration and API key
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.rag_chatbot import RAGChatbot
from src.vector_store import VectorStore

def test_llm_config():
    """Test LLM configuration"""
    print("🔍 Testing LLM Configuration...")
    print("=" * 50)
    
    # Check environment variables
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Together API Key: {'✅ Set' if settings.together_api_key else '❌ Not set'}")
    print(f"Together Model: {settings.together_model}")
    print(f"Together Base URL: {settings.together_base_url}")
    
    if settings.together_api_key:
        print(f"API Key (first 10 chars): {settings.together_api_key[:10]}...")
    else:
        print("❌ No Together AI API key found!")
        return False
    
    # Test LLM initialization
    try:
        print("\n🧠 Testing LLM Initialization...")
        vector_store = VectorStore()
        chatbot = RAGChatbot(vector_store)
        
        if chatbot.llm:
            print("✅ LLM initialized successfully!")
            print(f"Model: {chatbot.llm.model_name}")
            return True
        else:
            print("❌ LLM initialization failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return False

def test_simple_query():
    """Test a simple query with the LLM"""
    print("\n🤖 Testing Simple Query...")
    print("=" * 50)
    
    try:
        vector_store = VectorStore()
        chatbot = RAGChatbot(vector_store)
        
        if not chatbot.llm:
            print("❌ No LLM available for testing")
            return False
        
        # Test a simple query
        response = chatbot.chat("What is insurance?")
        print(f"Response: {response.answer}")
        print(f"Confidence: {response.confidence_score}")
        print("✅ Query test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing query: {e}")
        return False

if __name__ == "__main__":
    print("🚀 InsureSense 360 - LLM Configuration Test")
    print("=" * 60)
    
    # Test configuration
    config_ok = test_llm_config()
    
    if config_ok:
        # Test query
        query_ok = test_simple_query()
        
        if query_ok:
            print("\n🎉 All tests passed! LLM is working correctly.")
        else:
            print("\n⚠️ Configuration is correct but query failed.")
    else:
        print("\n❌ LLM configuration failed. Please check your API key.")
    
    print("\n" + "=" * 60) 