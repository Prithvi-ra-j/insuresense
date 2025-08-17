#!/usr/bin/env python3
"""
Direct LLM test with Together AI
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from langchain_openai import ChatOpenAI

def test_direct_llm():
    """Test LLM directly"""
    print("🧠 Testing Direct LLM with Together AI...")
    print("=" * 50)
    
    try:
        # Initialize LLM directly
        llm = ChatOpenAI(
            openai_api_key=settings.together_api_key,
            model_name=settings.together_model,
            openai_api_base=settings.together_base_url,
            temperature=0.1,
            max_tokens=500
        )
        
        print("✅ LLM initialized successfully!")
        print(f"Model: {llm.model_name}")
        
        # Test a simple query
        print("\n🤖 Testing simple query...")
        response = llm.invoke("What is insurance in one sentence?")
        print(f"Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Direct LLM Test with Together AI")
    print("=" * 60)
    
    success = test_direct_llm()
    
    if success:
        print("\n🎉 Direct LLM test passed!")
    else:
        print("\n❌ Direct LLM test failed!")
    
    print("=" * 60) 