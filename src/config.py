"""
Configuration settings for InsureSense 360
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "InsureSense 360"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Together AI Configuration
    together_api_key: Optional[str] = None
    together_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # Better Llama model for insurance Q&A
    together_base_url: str = "https://api.together.xyz"
    
    # LLM Provider Selection
    llm_provider: str = "together"  # "openai", "together", or "local"
    
    # Vector Database Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    faiss_index_path: str = "data/faiss_index"
    
    # File Storage
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    
    # IRDAI Configuration
    irdai_base_url: str = "https://www.irdai.gov.in"
    irdai_policies_path: str = "data/irdai_policies"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
