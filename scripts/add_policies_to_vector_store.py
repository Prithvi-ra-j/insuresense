#!/usr/bin/env python3
"""
Phase 6: Data Population Script
Add processed policies to vector store for semantic search and RAG functionality
"""

import sys
import os
from pathlib import Path
import json
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_policies():
    """Load processed policies from JSON files"""
    processed_dir = Path("data/processed")
    policies = []
    
    if not processed_dir.exists():
        logger.warning("No processed policies directory found")
        return policies
    
    for json_file in processed_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
                policies.append(policy_data)
                logger.info(f"Loaded policy: {policy_data.get('policy_id', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
    
    return policies

def load_final_dataset():
    """Load the final dataset CSV"""
    dataset_path = Path("data/extracted/final_dataset.csv")
    if dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded final dataset: {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load final dataset: {e}")
            return None
    else:
        logger.warning("Final dataset not found")
        return None

def add_policies_to_vector_store():
    """Add policies to vector store for semantic search"""
    logger.info("🚀 Starting Phase 6: Data Population")
    logger.info("=" * 60)
    
    # Initialize vector store
    try:
        vector_store = VectorStore()
        logger.info("✅ Vector store initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize vector store: {e}")
        return False
    
    # Load processed policies
    logger.info("\n📄 Loading processed policies...")
    policies = load_processed_policies()
    
    if not policies:
        logger.warning("No processed policies found. Creating sample policies...")
        # Create sample policies if none exist
        processor = DocumentProcessor()
        policies = processor.process_directory("data/irdai_policies")
    
    logger.info(f"✅ Loaded {len(policies)} policies")
    
    # Add policies to vector store
    logger.info("\n🗄️ Adding policies to vector store...")
    try:
        for i, policy in enumerate(policies, 1):
            logger.info(f"Processing policy {i}/{len(policies)}: {policy.get('policy_id', 'Unknown')}")
            
            # Extract text content for vectorization
            texts = []
            metadatas = []
            
            # Add policy sections
            for section in policy.get('sections', []):
                section_text = section.get('content', '')
                if section_text.strip():
                    texts.append(section_text)
                    metadatas.append({
                        'policy_id': policy.get('policy_id', ''),
                        'section_type': section.get('type', ''),
                        'section_title': section.get('title', ''),
                        'policy_type': policy.get('policy_type', ''),
                        'insurer': policy.get('insurer', ''),
                        'source': 'processed_policy'
                    })
            
            # Add policy metadata as a separate document
            if policy.get('summary'):
                texts.append(policy['summary'])
                metadatas.append({
                    'policy_id': policy.get('policy_id', ''),
                    'section_type': 'summary',
                    'section_title': 'Policy Summary',
                    'policy_type': policy.get('policy_type', ''),
                    'insurer': policy.get('insurer', ''),
                    'source': 'processed_policy'
                })
            
            # Add to vector store
            if texts:
                vector_store.add_texts(texts, metadatas)
                logger.info(f"  ✓ Added {len(texts)} text chunks")
        
        # Save vector store
        vector_store.save()
        logger.info("✅ Vector store saved successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to add policies to vector store: {e}")
        return False
    
    # Load final dataset and add to vector store
    logger.info("\n📊 Adding final dataset to vector store...")
    df = load_final_dataset()
    
    if df is not None and len(df) > 0:
        try:
            # Add structured data to vector store
            structured_texts = []
            structured_metadatas = []
            
            for _, row in df.iterrows():
                # Create text representation of the record
                text_parts = []
                if pd.notna(row.get('document_title')):
                    text_parts.append(f"Policy: {row['document_title']}")
                if pd.notna(row.get('policy_type')):
                    text_parts.append(f"Type: {row['policy_type']}")
                if pd.notna(row.get('benefits_summary')):
                    text_parts.append(f"Benefits: {row['benefits_summary']}")
                if pd.notna(row.get('exclusions_summary')):
                    text_parts.append(f"Exclusions: {row['exclusions_summary']}")
                if pd.notna(row.get('premium_summary')):
                    text_parts.append(f"Premium: {row['premium_summary']}")
                
                if text_parts:
                    text = " | ".join(text_parts)
                    structured_texts.append(text)
                    structured_metadatas.append({
                        'policy_id': str(row.get('policy_id', '')),
                        'section_type': 'structured_data',
                        'section_title': 'Policy Information',
                        'policy_type': str(row.get('policy_type', '')),
                        'insurer': str(row.get('insurer_name', '')),
                        'source': 'final_dataset',
                        'record_index': str(row.name)
                    })
            
            if structured_texts:
                vector_store.add_texts(structured_texts, structured_metadatas)
                logger.info(f"✅ Added {len(structured_texts)} structured data records")
                vector_store.save()
        
        except Exception as e:
            logger.error(f"❌ Failed to add structured data: {e}")
    
    # Test vector store functionality
    logger.info("\n🧪 Testing vector store functionality...")
    try:
        # Test search
        test_query = "life insurance benefits"
        results = vector_store.search(test_query, k=3)
        logger.info(f"✅ Search test successful: Found {len(results)} results for '{test_query}'")
        
        # Test similarity search
        similar_results = vector_store.similarity_search(test_query, k=2)
        logger.info(f"✅ Similarity search test successful: Found {len(similar_results)} similar documents")
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Phase 6: Data Population COMPLETED!")
    logger.info("✅ Policies added to vector store")
    logger.info("✅ Semantic search ready")
    logger.info("✅ RAG chatbot ready")
    logger.info("\n📋 Next steps:")
    logger.info("   1. Test /upload/policy endpoint")
    logger.info("   2. Test /search/policies endpoint")
    logger.info("   3. Test /chat endpoint")
    logger.info("   4. Move to Phase 7: RAG System Validation")
    
    return True

if __name__ == "__main__":
    success = add_policies_to_vector_store()
    if success:
        print("\n🚀 Ready to proceed with Phase 7!")
    else:
        print("\n❌ Phase 6 failed. Please check logs and try again.")