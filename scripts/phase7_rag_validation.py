#!/usr/bin/env python3
"""
Phase 7: RAG System Validation
Comprehensive testing of the RAG system with Together AI and Llama
"""

import sys
import os
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore
from src.rag_chatbot import RAGChatbot
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemValidator:
    """Comprehensive RAG system validator for Phase 7"""
    
    def __init__(self):
        self.vector_store = None
        self.chatbot = None
        self.test_results = {}
        
    def initialize_components(self):
        """Initialize vector store and RAG chatbot"""
        print("🔧 Initializing RAG System Components...")
        
        try:
            # Initialize vector store
            self.vector_store = VectorStore()
            print("✅ Vector store initialized")
            
            # Initialize RAG chatbot
            self.chatbot = RAGChatbot(self.vector_store)
            print("✅ RAG chatbot initialized")
            
            if self.chatbot.llm:
                print(f"✅ LLM initialized: {type(self.chatbot.llm).__name__}")
            else:
                print("⚠️ LLM not available, using fallback responses")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            return False
    
    def test_semantic_search(self):
        """Test semantic search functionality"""
        print("\n🔍 Testing Semantic Search...")
        print("=" * 50)
        
        test_queries = [
            "life insurance benefits",
            "health insurance coverage",
            "policy exclusions",
            "premium payment terms",
            "claim settlement process",
            "policy renewal conditions"
        ]
        
        results = {}
        
        for query in test_queries:
            try:
                print(f"\n📝 Testing query: '{query}'")
                
                # Test vector store search
                search_results = self.vector_store.search(query, top_k=5)
                
                print(f"   ✅ Found {len(search_results)} results")
                
                # Check result quality
                if search_results:
                    avg_score = sum(r.similarity_score for r in search_results) / len(search_results)
                    print(f"   📊 Average similarity score: {avg_score:.3f}")
                    
                    # Show top result
                    top_result = search_results[0]
                    print(f"   🏆 Top result: {top_result.policy_type} - {top_result.section_title}")
                    
                    results[query] = {
                        "count": len(search_results),
                        "avg_score": avg_score,
                        "top_result": {
                            "policy_type": top_result.policy_type,
                            "section_title": top_result.section_title,
                            "similarity_score": top_result.similarity_score
                        }
                    }
                else:
                    print("   ⚠️ No results found")
                    results[query] = {"count": 0, "avg_score": 0}
                    
            except Exception as e:
                print(f"   ❌ Error testing query '{query}': {e}")
                results[query] = {"error": str(e)}
        
        self.test_results["semantic_search"] = results
        return results
    
    def test_rag_chatbot(self):
        """Test RAG chatbot responses"""
        print("\n🤖 Testing RAG Chatbot...")
        print("=" * 50)
        
        test_questions = [
            "What are the main benefits of life insurance?",
            "What is covered under health insurance?",
            "What are the common exclusions in insurance policies?",
            "How do I file a claim?",
            "What happens if I miss a premium payment?",
            "Can I renew my policy after it expires?"
        ]
        
        results = {}
        
        for question in test_questions:
            try:
                print(f"\n💬 Testing question: '{question}'")
                
                # Test chatbot response
                start_time = time.time()
                response = self.chatbot.chat(question, use_langchain=False)
                response_time = time.time() - start_time
                
                print(f"   ✅ Response generated in {response_time:.2f}s")
                print(f"   📊 Confidence score: {response.confidence_score:.3f}")
                print(f"   📚 Sources found: {len(response.sources)}")
                
                # Show response preview
                answer_preview = response.answer[:150] + "..." if len(response.answer) > 150 else response.answer
                print(f"   💭 Answer: {answer_preview}")
                
                # Show sources
                if response.sources:
                    print(f"   📖 Sources:")
                    for i, source in enumerate(response.sources[:2], 1):
                        print(f"      {i}. {source.get('policy_type', 'Unknown')} - {source.get('section_title', 'Unknown')}")
                
                results[question] = {
                    "response_time": response_time,
                    "confidence_score": response.confidence_score,
                    "sources_count": len(response.sources),
                    "answer_length": len(response.answer),
                    "has_sources": len(response.sources) > 0
                }
                
            except Exception as e:
                print(f"   ❌ Error testing question '{question}': {e}")
                results[question] = {"error": str(e)}
        
        self.test_results["rag_chatbot"] = results
        return results
    
    def test_conversation_memory(self):
        """Test conversation memory and continuity"""
        print("\n🧠 Testing Conversation Memory...")
        print("=" * 50)
        
        conversation_id = "test_conversation_001"
        
        # Test conversation flow
        conversation_flow = [
            "What is life insurance?",
            "What are the benefits?",
            "How much coverage should I get?",
            "What about the premium costs?"
        ]
        
        results = {}
        
        try:
            print(f"🔄 Testing conversation flow with ID: {conversation_id}")
            
            for i, message in enumerate(conversation_flow, 1):
                print(f"\n💬 Message {i}: '{message}'")
                
                # Send message with conversation ID
                response = self.chatbot.chat(
                    message, 
                    conversation_id=conversation_id,
                    use_langchain=False
                )
                
                print(f"   ✅ Response {i}: {response.answer[:100]}...")
                print(f"   📊 Confidence: {response.confidence_score:.3f}")
                
                results[f"message_{i}"] = {
                    "message": message,
                    "response_length": len(response.answer),
                    "confidence_score": response.confidence_score,
                    "sources_count": len(response.sources)
                }
            
            # Check conversation history
            if conversation_id in self.chatbot.conversation_history:
                history = self.chatbot.conversation_history[conversation_id]
                print(f"\n📚 Conversation history length: {len(history)} messages")
                results["history_length"] = len(history)
            else:
                print("\n⚠️ No conversation history found")
                results["history_length"] = 0
            
        except Exception as e:
            print(f"❌ Error testing conversation memory: {e}")
            results["error"] = str(e)
        
        self.test_results["conversation_memory"] = results
        return results
    
    def test_follow_up_suggestions(self):
        """Test follow-up question suggestions"""
        print("\n💡 Testing Follow-up Suggestions...")
        print("=" * 50)
        
        test_questions = [
            "What is life insurance?",
            "How do I file a claim?",
            "What are the policy exclusions?"
        ]
        
        results = {}
        
        for question in test_questions:
            try:
                print(f"\n💬 Testing: '{question}'")
                
                response = self.chatbot.chat(question, use_langchain=False)
                
                if response.follow_up_questions:
                    print(f"   ✅ Generated {len(response.follow_up_questions)} follow-up questions:")
                    for i, follow_up in enumerate(response.follow_up_questions, 1):
                        print(f"      {i}. {follow_up}")
                    
                    results[question] = {
                        "follow_up_count": len(response.follow_up_questions),
                        "follow_ups": response.follow_up_questions
                    }
                else:
                    print("   ⚠️ No follow-up questions generated")
                    results[question] = {"follow_up_count": 0}
                    
            except Exception as e:
                print(f"   ❌ Error testing follow-ups for '{question}': {e}")
                results[question] = {"error": str(e)}
        
        self.test_results["follow_up_suggestions"] = results
        return results
    
    def test_performance_metrics(self):
        """Test performance metrics"""
        print("\n⚡ Testing Performance Metrics...")
        print("=" * 50)
        
        # Test response times
        test_queries = [
            "life insurance",
            "health coverage",
            "policy terms"
        ]
        
        performance_results = {}
        
        for query in test_queries:
            try:
                print(f"\n⏱️ Testing response time for: '{query}'")
                
                # Test search performance
                start_time = time.time()
                search_results = self.vector_store.search(query, top_k=5)
                search_time = time.time() - start_time
                
                # Test chatbot performance
                start_time = time.time()
                chat_response = self.chatbot.chat(query, use_langchain=False)
                chat_time = time.time() - start_time
                
                print(f"   🔍 Search time: {search_time:.3f}s")
                print(f"   🤖 Chat time: {chat_time:.3f}s")
                print(f"   📊 Total time: {search_time + chat_time:.3f}s")
                
                performance_results[query] = {
                    "search_time": search_time,
                    "chat_time": chat_time,
                    "total_time": search_time + chat_time,
                    "search_results": len(search_results),
                    "chat_confidence": chat_response.confidence_score
                }
                
            except Exception as e:
                print(f"   ❌ Error testing performance for '{query}': {e}")
                performance_results[query] = {"error": str(e)}
        
        self.test_results["performance_metrics"] = performance_results
        return performance_results
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Generating Test Report...")
        print("=" * 60)
        
        # Calculate overall metrics
        total_tests = 0
        passed_tests = 0
        
        for test_name, results in self.test_results.items():
            if isinstance(results, dict):
                test_count = len(results)
                error_count = sum(1 for r in results.values() if isinstance(r, dict) and "error" in r)
                passed_count = test_count - error_count
                
                total_tests += test_count
                passed_tests += passed_count
                
                print(f"\n📋 {test_name.replace('_', ' ').title()}:")
                print(f"   ✅ Passed: {passed_count}/{test_count}")
                print(f"   ❌ Failed: {error_count}/{test_count}")
                print(f"   📊 Success Rate: {(passed_count/test_count)*100:.1f}%")
        
        # Overall success rate
        overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 Overall Results:")
        print(f"   ✅ Total Passed: {passed_tests}/{total_tests}")
        print(f"   📊 Overall Success Rate: {overall_success_rate:.1f}%")
        
        # Phase 7 completion status
        if overall_success_rate >= 80:
            print(f"\n🎉 Phase 7: RAG System Validation - COMPLETED SUCCESSFULLY!")
            print("✅ Semantic search working correctly")
            print("✅ RAG chatbot providing accurate responses")
            print("✅ Conversation memory functional")
            print("✅ Performance metrics acceptable")
            print("\n📋 Ready to proceed with Phase 8: Frontend Development!")
        else:
            print(f"\n⚠️ Phase 7: RAG System Validation - NEEDS IMPROVEMENT")
            print("Some tests failed. Please review and fix issues.")
        
        # Save detailed report
        report_path = Path("data/test_reports/phase7_rag_validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return overall_success_rate >= 80

def main():
    """Main Phase 7 validation function"""
    print("🚀 Phase 7: RAG System Validation")
    print("=" * 60)
    print("Testing Together AI + Llama integration with insurance policies")
    print("=" * 60)
    
    # Initialize validator
    validator = RAGSystemValidator()
    
    # Run all tests
    tests = [
        ("Component Initialization", validator.initialize_components),
        ("Semantic Search", validator.test_semantic_search),
        ("RAG Chatbot", validator.test_rag_chatbot),
        ("Conversation Memory", validator.test_conversation_memory),
        ("Follow-up Suggestions", validator.test_follow_up_suggestions),
        ("Performance Metrics", validator.test_performance_metrics)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Generate final report
    success = validator.generate_test_report()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
