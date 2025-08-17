#!/usr/bin/env python3
"""
Test Together AI with Llama for InsureSense 360
Verify that the LLM integration is working correctly
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.rag_chatbot import RAGChatbot
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_together_ai_configuration():
    """Test Together AI configuration"""
    print("🔍 Testing Together AI Configuration...")
    print("=" * 50)
    
    # Check environment variables
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Together API Key: {'✅ Configured' if settings.together_api_key else '❌ Not configured'}")
    print(f"Together Model: {settings.together_model}")
    print(f"Together Base URL: {settings.together_base_url}")
    
    if not settings.together_api_key:
        print("\n❌ Together API key not found!")
        print("Please set TOGETHER_API_KEY in your .env file")
        return False
    
    return True

def test_llm_initialization():
    """Test LLM initialization"""
    print("\n🔍 Testing LLM Initialization...")
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        # Initialize RAG chatbot
        chatbot = RAGChatbot(vector_store)
        print("✅ RAG chatbot initialized")
        
        if chatbot.llm:
            print(f"✅ LLM initialized: {type(chatbot.llm).__name__}")
            return True
        else:
            print("❌ LLM not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return False

def test_llm_response():
    """Test LLM response generation"""
    print("\n🔍 Testing LLM Response Generation...")
    
    try:
        from langchain_together import Together as LangChainTogether
        
        # Initialize Together AI LLM
        llm = LangChainTogether(
            together_api_key=settings.together_api_key,
            model=settings.together_model,
            temperature=0.1,
            max_tokens=500,
        )
        
        # Test simple response
        test_prompt = "What is insurance? Please provide a brief explanation."
        print(f"Testing prompt: {test_prompt}")
        
        response = llm.invoke(test_prompt)
        print(f"✅ LLM Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LLM response: {e}")
        return False

def test_rag_chatbot():
    """Test RAG chatbot with Together AI"""
    print("\n🔍 Testing RAG Chatbot with Together AI...")
    
    try:
        # Initialize components
        vector_store = VectorStore()
        chatbot = RAGChatbot(vector_store)
        
        if not chatbot.llm:
            print("❌ LLM not available for RAG chatbot")
            return False
        
        # Test chat response with fallback method
        test_message = "What are the main benefits of life insurance?"
        print(f"Testing chat message: {test_message}")
        
        response = chatbot.chat(test_message, use_langchain=False)  # Use fallback method
        print(f"✅ Chat Response: {response.answer[:200]}...")
        print(f"✅ Confidence Score: {response.confidence_score}")
        print(f"✅ Sources Found: {len(response.sources)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG chatbot: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Together AI with Llama for InsureSense 360")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Configuration", test_together_ai_configuration),
        ("LLM Initialization", test_llm_initialization),
        ("LLM Response", test_llm_response),
        ("RAG Chatbot", test_rag_chatbot)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ FAILED: {test_name} - Exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   - Passed: {passed}/{total}")
    print(f"   - Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Together AI with Llama is working correctly.")
        print("✅ Configuration is correct")
        print("✅ LLM is initialized")
        print("✅ Response generation is working")
        print("✅ RAG chatbot is functional")
        print("\n📋 Ready to proceed with Phase 7!")
    else:
        print(f"\n⚠️ {total-passed} tests failed. Please check configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 