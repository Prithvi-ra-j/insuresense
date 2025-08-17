#!/usr/bin/env python3
"""
InsureSense 360 - FastAPI Core Endpoints
Comprehensive API for insurance policy analysis, recommendations, and RAG chatbot
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import uuid

# Import our core modules
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_chatbot import RAGChatbot
from src.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="InsureSense 360 API",
    description="AI-powered insurance platform for policy analysis and recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings and core components
settings = Settings()
document_processor = DocumentProcessor()

# Initialize vector store and chatbot with error handling
try:
    vector_store = VectorStore()
    logger.info("Initialized vector store successfully")
except Exception as e:
    logger.warning(f"Could not initialize vector store: {e}")
    vector_store = None

try:
    rag_chatbot = RAGChatbot(vector_store)
    logger.info("Initialized RAG chatbot successfully")
except Exception as e:
    logger.warning(f"Could not initialize RAG chatbot: {e}")
    rag_chatbot = None

# Load the final dataset
try:
    dataset_path = Path("data/extracted/final_dataset.csv")
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} records and {len(df.columns)} columns")
    else:
        df = pd.DataFrame()
        logger.warning("Final dataset not found, using empty DataFrame")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    df = pd.DataFrame()

# Pydantic models for request/response
class PolicySummaryRequest(BaseModel):
    policy_id: Optional[str] = None
    policy_text: Optional[str] = None
    summary_type: str = Field(default="comprehensive", description="Type of summary: brief, comprehensive, or detailed")

class PolicyRecommendationRequest(BaseModel):
    age: int = Field(..., ge=18, le=80, description="Age of the person")
    income: float = Field(..., ge=0, description="Annual income")
    family_size: int = Field(default=1, ge=1, le=10, description="Family size")
    existing_policies: List[str] = Field(default=[], description="List of existing policy types")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance: low, moderate, high")
    coverage_needs: List[str] = Field(default=[], description="Coverage needs: life, health, accident, etc.")

class RiskScoreRequest(BaseModel):
    age: int = Field(..., ge=18, le=80)
    health_conditions: List[str] = Field(default=[], description="List of health conditions")
    occupation: str = Field(default="", description="Occupation type")
    lifestyle_factors: List[str] = Field(default=[], description="Lifestyle factors: smoking, sports, etc.")
    family_history: List[str] = Field(default=[], description="Family medical history")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    search_type: str = Field(default="semantic", description="Search type: semantic, keyword, or hybrid")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for continuity")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class PolicySummaryResponse(BaseModel):
    policy_id: str
    summary: str
    key_points: List[str]
    coverage_details: Dict[str, Any]
    premium_info: Dict[str, Any]
    exclusions: List[str]
    recommendations: List[str]

class PolicyRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    reasoning: str
    risk_assessment: Dict[str, Any]
    cost_analysis: Dict[str, Any]

class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_level: str
    factors: List[Dict[str, Any]]
    recommendations: List[str]

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    query_time: float

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    confidence: float
    sources: List[Dict[str, Any]]
    follow_up_questions: List[str]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "dataset_loaded": len(df) > 0,
        "dataset_records": len(df),
        "dataset_columns": len(df.columns) if len(df) > 0 else 0,
        "vector_store_available": vector_store is not None,
        "rag_chatbot_available": rag_chatbot is not None and rag_chatbot.llm is not None
    }

# Policy summarization endpoint
@app.post("/summarize_policy", response_model=PolicySummaryResponse)
async def summarize_policy(request: PolicySummaryRequest):
    """Summarize insurance policy documents"""
    try:
        if request.policy_id and len(df) > 0:
            # Find policy in dataset
            policy_data = df[df['policy_id'] == request.policy_id]
            if policy_data.empty:
                raise HTTPException(status_code=404, detail="Policy not found")
            
            policy = policy_data.iloc[0]
            
            # Extract key information
            summary = f"Policy: {policy.get('document_title', 'Unknown')}\n"
            summary += f"Type: {policy.get('policy_type', 'Unknown')}\n"
            summary += f"Insurer: {policy.get('insurer_name', 'Unknown')}\n\n"
            
            # Add benefits summary
            if 'benefits_summary' in policy and pd.notna(policy['benefits_summary']):
                summary += f"Benefits: {policy['benefits_summary']}\n\n"
            
            # Add exclusions summary
            if 'exclusions_summary' in policy and pd.notna(policy['exclusions_summary']):
                summary += f"Exclusions: {policy['exclusions_summary']}\n\n"
            
            # Add premium summary
            if 'premium_summary' in policy and pd.notna(policy['premium_summary']):
                summary += f"Premium: {policy['premium_summary']}\n\n"
            
            key_points = []
            if 'benefits_text' in policy and pd.notna(policy['benefits_text']):
                key_points.append("Comprehensive benefits coverage")
            if 'exclusions_text' in policy and pd.notna(policy['exclusions_text']):
                key_points.append("Standard exclusions apply")
            
            coverage_details = {
                "policy_type": policy.get('policy_type', 'Unknown'),
                "sections_count": policy.get('sections_count', 0),
                "total_pages": policy.get('total_pages', 0)
            }
            
            premium_info = {
                "details": policy.get('premium_summary', 'Not available'),
                "payment_frequency": "As per policy terms"
            }
            
            exclusions = []
            if 'exclusions_text' in policy and pd.notna(policy['exclusions_text']):
                exclusions = ["Standard policy exclusions apply"]
            
            recommendations = [
                "Review policy terms carefully",
                "Consult with insurance advisor",
                "Compare with similar policies"
            ]
            
        elif request.policy_text:
            # Process uploaded text
            summary = f"Policy Summary (Generated from text)\n\n"
            summary += f"Text length: {len(request.policy_text)} characters\n"
            summary += "Key sections identified and analyzed.\n\n"
            
            # Simple text analysis
            sentences = request.policy_text.split('.')
            key_points = [f"Document contains {len(sentences)} sentences"]
            
            coverage_details = {
                "text_length": len(request.policy_text),
                "sentences": len(sentences),
                "words": len(request.policy_text.split())
            }
            
            premium_info = {"details": "Extracted from policy text"}
            exclusions = ["Standard exclusions may apply"]
            recommendations = ["Review complete policy document"]
            
        else:
            raise HTTPException(status_code=400, detail="Either policy_id or policy_text must be provided")
        
        return PolicySummaryResponse(
            policy_id=request.policy_id or str(uuid.uuid4()),
            summary=summary,
            key_points=key_points,
            coverage_details=coverage_details,
            premium_info=premium_info,
            exclusions=exclusions,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in summarize_policy: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing policy: {str(e)}")

# Policy recommendation endpoint
@app.post("/recommend_policy", response_model=PolicyRecommendationResponse)
async def recommend_policy(request: PolicyRecommendationRequest):
    """Recommend insurance policies based on user criteria"""
    try:
        recommendations = []
        
        # Basic recommendation logic based on age and income
        if request.age < 30:
            if request.income < 500000:
                recommendations.append({
                    "policy_type": "Term Life Insurance",
                    "reason": "Affordable protection for young individuals",
                    "estimated_premium": "₹2,000-5,000/year",
                    "coverage_amount": "₹10-25 Lakhs",
                    "priority": "High"
                })
            else:
                recommendations.append({
                    "policy_type": "Endowment Plan",
                    "reason": "Good savings and protection combination",
                    "estimated_premium": "₹15,000-30,000/year",
                    "coverage_amount": "₹25-50 Lakhs",
                    "priority": "High"
                })
        
        elif request.age < 50:
            recommendations.append({
                "policy_type": "Whole Life Insurance",
                "reason": "Lifetime protection with cash value",
                "estimated_premium": "₹20,000-40,000/year",
                "coverage_amount": "₹30-75 Lakhs",
                "priority": "High"
            })
            
            if request.family_size > 2:
                recommendations.append({
                    "policy_type": "Family Floater Health Insurance",
                    "reason": "Comprehensive health coverage for family",
                    "estimated_premium": "₹8,000-15,000/year",
                    "coverage_amount": "₹5-10 Lakhs",
                    "priority": "Medium"
                })
        
        else:
            recommendations.append({
                "policy_type": "Senior Citizen Health Insurance",
                "reason": "Specialized coverage for senior citizens",
                "estimated_premium": "₹15,000-25,000/year",
                "coverage_amount": "₹3-7 Lakhs",
                "priority": "High"
            })
        
        # Add risk-based recommendations
        if request.risk_tolerance == "high":
            recommendations.append({
                "policy_type": "ULIP (Unit Linked Insurance Plan)",
                "reason": "Investment-linked insurance for high risk tolerance",
                "estimated_premium": "₹25,000-50,000/year",
                "coverage_amount": "₹50-100 Lakhs",
                "priority": "Medium"
            })
        
        reasoning = f"Based on your age ({request.age}), income (₹{request.income:,.0f}), and family size ({request.family_size}), we recommend these policies to provide comprehensive protection."
        
        risk_assessment = {
            "age_risk": "Low" if request.age < 40 else "Medium" if request.age < 60 else "High",
            "income_adequacy": "Good" if request.income > 1000000 else "Adequate" if request.income > 500000 else "Needs improvement",
            "family_protection": "High" if request.family_size > 2 else "Medium"
        }
        
        total_annual_premium = sum([
            float(rec["estimated_premium"].split("-")[0].replace("₹", "").replace(",", ""))
            for rec in recommendations
        ])
        
        cost_analysis = {
            "total_annual_premium": f"₹{total_annual_premium:,.0f}",
            "premium_to_income_ratio": f"{(total_annual_premium / request.income * 100):.1f}%",
            "affordability": "Good" if (total_annual_premium / request.income) < 0.1 else "Moderate"
        }
        
        return PolicyRecommendationResponse(
            recommendations=recommendations,
            reasoning=reasoning,
            risk_assessment=risk_assessment,
            cost_analysis=cost_analysis
        )
        
    except Exception as e:
        logger.error(f"Error in recommend_policy: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Risk scoring endpoint
@app.post("/risk_score", response_model=RiskScoreResponse)
async def calculate_risk_score(request: RiskScoreRequest):
    """Calculate risk score based on various factors"""
    try:
        base_score = 50  # Base risk score
        
        # Age factor
        if request.age < 30:
            base_score -= 10
        elif request.age < 50:
            base_score += 5
        else:
            base_score += 20
        
        # Health conditions
        health_risk = len(request.health_conditions) * 10
        base_score += health_risk
        
        # Occupation risk
        high_risk_occupations = ["pilot", "firefighter", "police", "construction", "mining"]
        if any(occ in request.occupation.lower() for occ in high_risk_occupations):
            base_score += 15
        
        # Lifestyle factors
        if "smoking" in request.lifestyle_factors:
            base_score += 20
        if "sports" in request.lifestyle_factors:
            base_score += 5
        
        # Family history
        family_risk = len(request.family_history) * 5
        base_score += family_risk
        
        # Normalize score to 0-100
        risk_score = max(0, min(100, base_score))
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        factors = []
        if request.age >= 50:
            factors.append({"factor": "Age", "impact": "High", "description": "Age-related risk factors"})
        if request.health_conditions:
            factors.append({"factor": "Health Conditions", "impact": "High", "description": f"{len(request.health_conditions)} health conditions identified"})
        if "smoking" in request.lifestyle_factors:
            factors.append({"factor": "Smoking", "impact": "High", "description": "Smoking increases health risks"})
        
        recommendations = []
        if risk_score > 60:
            recommendations.append("Consider comprehensive health insurance")
            recommendations.append("Regular health checkups recommended")
        if "smoking" in request.lifestyle_factors:
            recommendations.append("Consider smoking cessation programs")
        if request.age >= 50:
            recommendations.append("Consider senior citizen specific policies")
        
        return RiskScoreResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            factors=factors,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in calculate_risk_score: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating risk score: {str(e)}")

# Semantic search endpoint - Updated to use vector store
@app.get("/search_policies", response_model=SearchResponse)
async def search_policies(
    query: str,
    max_results: int = 10,
    search_type: str = "semantic"
):
    """Search policies using semantic search with vector store"""
    try:
        import time
        start_time = time.time()
        
        # Use vector store for semantic search if available
        if vector_store is not None and search_type == "semantic":
            logger.info(f"Performing semantic search: {query}")
            search_results = vector_store.search(query, top_k=max_results)
            
            # Convert to API response format
            results = []
            for result in search_results:
                results.append({
                    "policy_id": result.policy_id,
                    "title": result.document_title,
                    "type": result.policy_type,
                    "insurer": result.insurer_name,
                    "relevance_score": result.similarity_score,
                    "summary": result.section_content[:200] + "..." if len(result.section_content) > 200 else result.section_content,
                    "section_title": result.section_title,
                    "page_number": result.page_number
                })
            
            query_time = time.time() - start_time
            
            return SearchResponse(
                results=results,
                total_found=len(results),
                query_time=query_time
            )
        elif len(df) > 0:
            # Fallback to simple text-based search
            logger.info(f"Performing text-based search: {query}")
            results = []
            query_lower = query.lower()
            
            # Search in policy titles and descriptions
            for idx, row in df.iterrows():
                score = 0
                searchable_text = ""
                
                # Combine searchable fields
                for field in ['document_title', 'policy_type', 'insurer_name', 'benefits_summary', 'exclusions_summary']:
                    if field in row and pd.notna(row[field]):
                        searchable_text += f" {str(row[field]).lower()}"
                
                # Simple keyword matching
                for word in query_lower.split():
                    if word in searchable_text:
                        score += 1
                
                if score > 0:
                    results.append({
                        "policy_id": row.get('policy_id', f"policy_{idx}"),
                        "title": row.get('document_title', 'Unknown'),
                        "type": row.get('policy_type', 'Unknown'),
                        "insurer": row.get('insurer_name', 'Unknown'),
                        "relevance_score": score,
                        "summary": row.get('benefits_summary', '')[:200] + "..." if pd.notna(row.get('benefits_summary')) else "",
                        "sections_count": row.get('sections_count', 0)
                    })
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            results = results[:max_results]
            
            query_time = time.time() - start_time
            
            return SearchResponse(
                results=results,
                total_found=len(results),
                query_time=query_time
            )
        else:
            return SearchResponse(
                results=[],
                total_found=0,
                query_time=0.0
            )
            
    except Exception as e:
        logger.error(f"Error in search_policies: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching policies: {str(e)}")

# RAG chatbot endpoint - Updated to use actual RAG chatbot
@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """Chat with RAG-powered insurance assistant"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Check if RAG chatbot is available
        if rag_chatbot is not None and rag_chatbot.llm is not None:
            # Use actual RAG chatbot
            logger.info(f"Processing chat message: {request.message}")
            response = rag_chatbot.chat(
                user_message=request.message,
                conversation_id=conversation_id
            )
            
            # Convert to API response format
            return ChatResponse(
                response=response.answer,
                conversation_id=response.conversation_id or conversation_id,
                confidence=response.confidence_score,
                sources=response.sources,
                follow_up_questions=response.follow_up_questions or []
            )
        elif rag_chatbot is not None:
            # Use RAG chatbot without LLM (fallback to manual RAG)
            logger.info(f"Processing chat message with fallback RAG: {request.message}")
            response = rag_chatbot.chat(
                user_message=request.message,
                conversation_id=conversation_id
            )
            
            # Convert to API response format
            return ChatResponse(
                response=response.answer,
                conversation_id=response.conversation_id or conversation_id,
                confidence=response.confidence_score,
                sources=response.sources,
                follow_up_questions=response.follow_up_questions or []
            )
        else:
            # Fallback to simple response generation
            user_message = request.message.lower()
            
            if "policy" in user_message and "recommend" in user_message:
                response = "I can help you find the right insurance policy! Please provide your age, income, and family size for personalized recommendations."
                confidence = 0.9
                sources = [{"type": "recommendation_engine", "content": "Policy recommendation system"}]
                follow_up = ["What's your age?", "What's your annual income?", "How many family members do you have?"]
            
            elif "risk" in user_message and "score" in user_message:
                response = "I can calculate your risk score based on various factors like age, health conditions, and lifestyle. Would you like to provide these details?"
                confidence = 0.85
                sources = [{"type": "risk_assessment", "content": "Risk scoring algorithm"}]
                follow_up = ["What's your age?", "Do you have any health conditions?", "What's your occupation?"]
            
            elif "premium" in user_message or "cost" in user_message:
                response = "Premium costs vary based on policy type, coverage amount, age, and health factors. I can help you estimate costs for specific policies."
                confidence = 0.8
                sources = [{"type": "premium_calculator", "content": "Premium calculation system"}]
                follow_up = ["What type of policy are you interested in?", "What's your age?", "What coverage amount do you need?"]
            
            elif "claim" in user_message or "benefits" in user_message:
                response = "I can help you understand policy benefits and claims processes. Which specific policy or benefit would you like to know more about?"
                confidence = 0.9
                sources = [{"type": "policy_database", "content": "Policy benefits database"}]
                follow_up = ["Which policy are you asking about?", "What type of claim?", "What benefits are you looking for?"]
            
            else:
                response = "I'm your insurance assistant! I can help you with policy recommendations, risk assessment, premium calculations, and understanding policy benefits. What would you like to know?"
                confidence = 0.7
                sources = [{"type": "general_knowledge", "content": "Insurance domain knowledge"}]
                follow_up = ["Tell me about policy recommendations", "Calculate my risk score", "Help me understand premiums"]
            
            return ChatResponse(
                response=response,
                conversation_id=conversation_id,
                confidence=confidence,
                sources=sources,
                follow_up_questions=follow_up
            )
        
    except Exception as e:
        logger.error(f"Error in chat_with_rag: {e}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

# Dataset statistics endpoint
@app.get("/dataset/stats")
async def get_dataset_stats():
    """Get dataset statistics"""
    try:
        if len(df) == 0:
            return {"message": "No dataset loaded"}
        
        stats = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_records": df.duplicated().sum(),
            "data_types": df.dtypes.value_counts().to_dict(),
            "sample_columns": list(df.columns[:10])
        }
        
        # Add policy-specific stats
        if 'policy_type' in df.columns:
            stats['policy_types'] = df['policy_type'].value_counts().to_dict()
        
        if 'insurer_name' in df.columns:
            stats['insurers'] = df['insurer_name'].value_counts().to_dict()
        
        # Add vector store stats if available
        if vector_store is not None:
            try:
                vector_stats = vector_store.get_statistics()
                stats['vector_store_stats'] = vector_stats
            except Exception as e:
                logger.warning(f"Could not get vector store stats: {e}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_dataset_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dataset stats: {str(e)}")

# Policy upload endpoint
@app.post("/upload/policy")
async def upload_policy(file: UploadFile = File(...)):
    """Upload and process insurance policy document"""
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        policy = document_processor.process_document(str(file_path))
        
        # Add to vector store if available
        if vector_store is not None:
            try:
                vector_store.add_policy(policy)
                logger.info(f"Added policy {policy.policy_id} to vector store")
            except Exception as e:
                logger.warning(f"Could not add policy to vector store: {e}")
        
        return {
            "message": "Policy uploaded and processed successfully",
            "policy_id": policy.policy_id,
            "document_title": policy.document_title,
            "policy_type": policy.policy_type,
            "sections_count": len(policy.sections),
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error in upload_policy: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading policy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
