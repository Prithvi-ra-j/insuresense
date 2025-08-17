"""
RAG Chatbot for Insurance Policy Q&A
Combines vector search with AI generation
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.retrievers import BaseRetriever
from langchain_community.callbacks.manager import get_openai_callback
from langchain_together import Together as LangChainTogether
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Together AI client
from together import Together

from .vector_store import VectorStore, SearchResult
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None

@dataclass
class ChatResponse:
    """Represents a chat response from the RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    search_query: str
    conversation_id: str = None
    follow_up_questions: List[str] = None

class RAGChatbot:
    """Enhanced RAG chatbot using LangChain components"""
    
    def __init__(self, vector_store: VectorStore = None, 
                 use_memory: bool = True,
                 memory_type: str = "buffer"):
        self.vector_store = vector_store
        self.use_memory = use_memory
        self.memory_type = memory_type
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize memory
        self.memory = self._initialize_memory() if use_memory else None
        
        # Initialize RAG chain
        self.rag_chain = self._initialize_rag_chain()
        
        # Conversation tracking
        self.conversation_history = {}
        self.conversation_counter = 0
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if settings.together_api_key:
                os.environ["TOGETHER_API_KEY"] = settings.together_api_key

            if settings.llm_provider == "together" and settings.together_api_key:
                logger.info("Using Together AI via LangChain integration")
                return LangChainTogether(
                    together_api_key=settings.together_api_key,
                    model=settings.together_model,
                    temperature=0.1,
                    max_tokens=1000,
                )
            elif settings.llm_provider == "openai" and settings.openai_api_key:
                logger.info("Using OpenAI Chat model")
                return ChatOpenAI(
                    openai_api_key=settings.openai_api_key,
                    model_name=settings.openai_model,
                    temperature=0.1,
                    max_tokens=1000
                )
            else:
                logger.warning("No API key found. Using fallback responses.")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _initialize_memory(self):
        """Initialize conversation memory"""
        try:
            if self.memory_type == "buffer":
                return ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            elif self.memory_type == "summary":
                return ConversationSummaryMemory(
                    llm=self.llm,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            else:
                logger.warning(f"Unknown memory type: {self.memory_type}. Using buffer.")
                return ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            return None
    
    def _initialize_rag_chain(self):
        """Initialize the RAG chain using the new LangChain pattern with input normalization."""
        try:
            if not self.llm or not self.vector_store:
                logger.warning("No LLM or vector store available. RAG chain not initialized.")
                return None

            retriever = self.vector_store.get_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            # History-aware retriever
            history_aware_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert insurance assistant. Use the chat history and question to formulate a search query for the insurance policy database."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            history_aware_retriever = create_history_aware_retriever(
                self.llm,
                retriever,
                history_aware_prompt
            )

            # RAG chain
            rag_chain = create_retrieval_chain(
                history_aware_retriever,
                self.llm
            )

            # --- Input normalization ---
            from langchain_core.runnables import RunnableLambda

            def format_input(user_message_and_meta):
                """
                Normalize input so rag_chain always receives:
                {"input": <str>, "chat_history": <list>}
                """
                if isinstance(user_message_and_meta, str):
                    return {"input": user_message_and_meta, "chat_history": []}

                # Dict case
                user_message = user_message_and_meta.get("message", "")
                conversation_id = user_message_and_meta.get("conversation_id")

                chat_history = []
                if conversation_id and conversation_id in self.conversation_history:
                    chat_history = [
                        {"role": msg.role, "content": msg.content}
                        for msg in self.conversation_history[conversation_id]
                    ]

                return {"input": user_message, "chat_history": chat_history}

            # Ensure rag_chain only ever receives dicts
            return RunnableLambda(format_input) | rag_chain

        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            return None
    
    def _create_qa_prompt(self) -> PromptTemplate:
        """Create the QA prompt template"""
        template = """You are an expert insurance advisor assistant. You help users understand insurance policies by providing clear, accurate information based on the provided policy documents.

Context information from insurance policies:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Use simple, clear language that non-experts can understand
4. Always cite the specific policy sections you're referencing
5. If you're unsure about something, acknowledge the uncertainty

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def chat(
        self,
        user_message: str,
        conversation_id: str = None,
        policy_types: List[str] = None,
        insurers: List[str] = None,
        max_sources: int = 3,
        use_langchain: bool = True
    ) -> ChatResponse:
        """
        Handle a chat message and generate a response.
        Maintains conversation history if conversation_id is provided.
        """

        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{self.conversation_counter}"
            self.conversation_counter += 1

        # Ensure conversation history is initialized
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        # Append user message to history
        self.conversation_history[conversation_id].append(
            ChatMessage(
                role="user",
                content=user_message,
                timestamp=self._get_timestamp()
            )
        )

        # Generate response
        if use_langchain:
            response = self._generate_langchain_response(
                user_message=user_message,
                policy_types=policy_types,
                insurers=insurers,
                max_sources=max_sources,
                conversation_id=conversation_id   # ✅ Always passed
            )
        else:
            response = self._generate_fallback_response(
                user_message=user_message,
                policy_types=policy_types,
                insurers=insurers,
                max_sources=max_sources
            )

        # Append assistant response to history
        self.conversation_history[conversation_id].append(
            ChatMessage(
                role="assistant",
                content=response.answer,
                timestamp=self._get_timestamp(),
                metadata={
                    'sources': response.sources,
                    'confidence': response.confidence_score
                }
            )
        )

        # Add conversation ID to response
        response.conversation_id = conversation_id

        # Generate follow-up questions
        response.follow_up_questions = self._generate_follow_up_questions(
            user_message, response.answer
        )

        return response
    
    def _generate_langchain_response(self, user_message: str,
                                   policy_types: List[str] = None,
                                   insurers: List[str] = None,
                                   max_sources: int = 3,
                                   conversation_id: str = None) -> ChatResponse:
        """Generate response using LangChain RAG chain (new pattern)"""
        try:
            # Get relevant documents
            search_results = self._retrieve_relevant_content(
                user_message, max_sources, policy_types, insurers
            )

            if not search_results:
                return ChatResponse(
                    answer="I couldn't find any relevant information in the policy documents to answer your question. Please try rephrasing your question or ask about a different topic.",
                    sources=[],
                    confidence_score=0.0,
                    search_query=user_message
                )

            # Prepare chat history for the chain
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in self.conversation_history.get(conversation_id, [])
            ]

            # Prepare chat history as a list of BaseMessages
            messages = [
                HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
                for msg in self.conversation_history.get(conversation_id, [])
            ]
            messages.append(HumanMessage(content=user_message))

            if self.rag_chain is None:
                logger.warning("RAG chain is None. Using fallback response.")
                return self._generate_fallback_response(user_message, policy_types, insurers, max_sources)
            
            response = self.rag_chain.invoke(messages)
            
            # Handle different response formats
            if isinstance(response, dict) and "answer" in response:
                answer = response["answer"]
            elif isinstance(response, str):
                answer = response
            else:
                # If response is a list or other format, try to extract the answer
                answer = str(response) if response else "I couldn't generate a response."

            # Format sources
            sources = self._format_sources(search_results)

            # Calculate confidence
            confidence = self._calculate_confidence(search_results)

            return ChatResponse(
                answer=answer,
                sources=sources,
                confidence_score=confidence,
                search_query=user_message
            )

        except Exception as e:
            logger.error(f"Error in LangChain response generation: {e}")
            raise
    
    def _generate_fallback_response(self, user_message: str,
                                  policy_types: List[str] = None,
                                  insurers: List[str] = None,
                                  max_sources: int = 3) -> ChatResponse:
        """Generate response using fallback method"""
        # Retrieve relevant content
        search_results = self._retrieve_relevant_content(
            user_message, max_sources, policy_types, insurers
        )
        
        if not search_results:
            return ChatResponse(
                answer="I couldn't find any relevant information in the policy documents to answer your question. Please try rephrasing your question or ask about a different topic.",
                sources=[],
                confidence_score=0.0,
                search_query=user_message
            )
        
        # Generate response using OpenAI if available
        if self.llm:
            answer = self._generate_openai_response(user_message, search_results)
        else:
            answer = self._generate_fallback_answer(user_message, search_results)
        
        # Format sources
        sources = self._format_sources(search_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(search_results)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence_score=confidence,
            search_query=user_message
        )
    
    def _retrieve_relevant_content(self, query: str, max_sources: int,
                                 policy_types: List[str] = None,
                                 insurers: List[str] = None) -> List[SearchResult]:
        """Retrieve relevant content from vector store"""
        if not self.vector_store:
            return []
            
        return self.vector_store.search(
            query, 
            top_k=max_sources * 2,  # Get more for filtering
            policy_types=policy_types,
            insurers=insurers
        )[:max_sources]
    
    def _prepare_context_for_langchain(self, search_results: List[SearchResult]) -> List:
        """Prepare context documents for LangChain"""
        from langchain_core.documents import Document
        
        context_docs = []
        for result in search_results:
            doc = Document(
                page_content=result.section_content,
                metadata={
                    'policy_id': result.policy_id,
                    'insurer_name': result.insurer_name,
                    'policy_type': result.policy_type,
                    'section_title': result.section_title,
                    'section_type': result.section_type,
                    'page_number': result.page_number
                }
            )
            context_docs.append(doc)
        
        return context_docs
    
    def _generate_openai_response(self, user_message: str,
                                search_results: List[SearchResult]) -> str:
        """Generate response using Together AI or OpenAI"""
        try:
            # Prepare context
            context = self._prepare_context_for_openai(search_results)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Generate response based on LLM type
            if isinstance(self.llm, Together):
                # Use Together AI client directly
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
                ]
                
                response = self.llm.chat.completions.create(
                    model=settings.together_model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.choices[0].message.content
                
            elif isinstance(self.llm, LangChainTogether):
                # Use LangChainTogether (handled by LangChain chain)
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_message}\n\nAnswer:"
                response = self.llm.invoke(prompt)
                return response
            elif hasattr(self.llm, 'predict'):
                # For base LLM
                prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_message}\n\nAnswer:"
                response = self.llm.predict(prompt)
                return response
            else:
                # For chat model
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
                ]
                response = self.llm.predict_messages(messages)
                response = response.content if hasattr(response, 'content') else str(response)
                return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_answer(user_message, search_results)
    
    def _prepare_context_for_openai(self, search_results: List[SearchResult]) -> str:
        """Prepare context string for OpenAI"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_part = f"""
Source {i} - {result.section_title} (Policy: {result.policy_id}, Insurer: {result.insurer_name}):
{result.section_content[:500]}{'...' if len(result.section_content) > 500 else ''}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the AI assistant"""
        return """You are an expert insurance advisor assistant. Your role is to help users understand insurance policies by providing clear, accurate information based on the provided policy documents.

Key responsibilities:
1. Answer questions based ONLY on the provided context
2. Explain complex insurance terms in simple language
3. Provide accurate information about coverage, exclusions, and terms
4. Cite specific policy sections when answering
5. Acknowledge when information is not available in the context
6. Help users make informed decisions about insurance

Guidelines:
- Be helpful and professional
- Use clear, concise language
- Always reference the source documents
- If unsure, say so clearly
- Focus on factual information from the policies"""
    
    def _generate_fallback_answer(self, user_message: str,
                                search_results: List[SearchResult]) -> str:
        """Generate a basic fallback answer without LLM"""
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Create a simple answer based on search results
        answer_parts = [f"Based on the policy documents, I found the following relevant information:"]
        
        for i, result in enumerate(search_results, 1):
            answer_parts.append(f"\n{i}. From {result.section_title} (Policy: {result.policy_id}):")
            answer_parts.append(f"   {result.section_content[:200]}{'...' if len(result.section_content) > 200 else ''}")
        
        answer_parts.append("\nFor more detailed information, please refer to the specific policy documents mentioned above.")
        
        return "".join(answer_parts)
    
    def _format_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format search results as source citations"""
        sources = []
        
        for result in search_results:
            source = {
                'policy_id': result.policy_id,
                'insurer_name': result.insurer_name,
                'policy_type': result.policy_type,
                'section_title': result.section_title,
                'section_type': result.section_type,
                'page_number': result.page_number,
                'similarity_score': result.similarity_score,
                'content_preview': result.section_content[:200] + '...' if len(result.section_content) > 200 else result.section_content
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score based on search results"""
        if not search_results:
            return 0.0
        
        # Calculate average similarity score
        avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
        
        # Boost confidence based on number of sources
        source_boost = min(len(search_results) / 3.0, 0.3)
        
        # Boost confidence based on high similarity scores
        high_similarity_boost = sum(1 for result in search_results if result.similarity_score > 0.8) * 0.1
        
        confidence = min(avg_similarity + source_boost + high_similarity_boost, 1.0)
        
        return round(confidence, 2)
    
    def _generate_follow_up_questions(self, user_message: str, answer: str) -> List[str]:
        """Generate follow-up questions based on the conversation"""
        follow_up_questions = [
            "Would you like me to explain any specific terms mentioned in the answer?",
            "Do you have questions about other aspects of this policy?",
            "Would you like me to search for similar policies from other insurers?",
            "Is there anything specific about coverage or exclusions you'd like to know more about?"
        ]
        
        # Add context-specific questions
        if "coverage" in user_message.lower() or "coverage" in answer.lower():
            follow_up_questions.append("Would you like me to explain what is NOT covered under this policy?")
        
        if "premium" in user_message.lower() or "cost" in user_message.lower():
            follow_up_questions.append("Would you like me to explain the payment terms and conditions?")
        
        if "exclusion" in user_message.lower() or "exclusion" in answer.lower():
            follow_up_questions.append("Would you like me to explain what IS covered under this policy?")
        
        return follow_up_questions[:3]  # Return top 3 questions
    
    def _create_error_response(self, conversation_id: str, error_message: str) -> ChatResponse:
        """Create an error response"""
        return ChatResponse(
            answer=f"I apologize, but I encountered an error while processing your request: {error_message}. Please try again or rephrase your question.",
            sources=[],
            confidence_score=0.0,
            search_query="",
            conversation_id=conversation_id
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def suggest_questions(self, topic: str = None) -> List[str]:
        """Suggest questions for users to ask"""
        if topic:
            # Topic-specific suggestions
            if "coverage" in topic.lower():
                return [
                    "What does this policy cover?",
                    "Are there any limitations on coverage?",
                    "What is the maximum coverage amount?",
                    "Does this cover pre-existing conditions?"
                ]
            elif "premium" in topic.lower():
                return [
                    "How much does this policy cost?",
                    "What are the payment options?",
                    "Is there a grace period for payments?",
                    "Can I change my premium amount?"
                ]
            elif "exclusion" in topic.lower():
                return [
                    "What is NOT covered under this policy?",
                    "Are there any waiting periods?",
                    "What conditions are excluded?",
                    "Are there any age restrictions?"
                ]
        
        # General suggestions
        return [
            "What does this insurance policy cover?",
            "How much does this policy cost?",
            "What are the main exclusions?",
            "What is the claims process?",
            "Can you explain the terms and conditions?",
            "What are the renewal terms?",
            "Is there a waiting period?",
            "What documents do I need for claims?"
        ]
    
    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """Get conversation history for a specific conversation"""
        return self.conversation_history.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str):
        """Clear a specific conversation"""
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
            logger.info(f"Cleared conversation: {conversation_id}")
    
    def clear_all_conversations(self):
        """Clear all conversation history"""
        self.conversation_history.clear()
        logger.info("Cleared all conversation history")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversations"""
        total_messages = sum(len(conv) for conv in self.conversation_history.values())
        total_conversations = len(self.conversation_history)
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "active_conversations": len([conv for conv in self.conversation_history.values() if conv]),
            "memory_type": self.memory_type if self.memory else "none",
            "llm_available": self.llm is not None,
            "rag_chain_available": self.rag_chain is not None
        }
