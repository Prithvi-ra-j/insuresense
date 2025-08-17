"""
Vector store for insurance policy embeddings
Handles text embedding and FAISS-based similarity search
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# LangChain imports
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .document_processor import InsurancePolicy, PolicySection
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from the vector store"""
    policy_id: str
    insurer_name: str
    policy_type: str
    document_title: str
    section_title: str
    section_content: str
    section_type: str
    similarity_score: float
    page_number: int
    metadata: Dict[str, Any] = None

class VectorStore:
    """Enhanced vector store using LangChain implementations"""
    
    def __init__(self, model_name: str = None, vector_store_type: str = "faiss"):
        self.model_name = model_name or settings.embedding_model
        self.vector_store_type = vector_store_type
        self.embedding_model = None
        self.vector_store = None
        self.metadata_store = {}
        self._initialize_embeddings()
        self._load_or_create_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            # Try OpenAI embeddings first if API key is available
            if settings.openai_api_key:
                logger.info("Using OpenAI embeddings")
                self.embedding_model = OpenAIEmbeddings(
                    openai_api_key=settings.openai_api_key,
                    model="text-embedding-ada-002"
                )
            else:
                # Fallback to HuggingFace embeddings
                logger.info(f"Using HuggingFace embeddings: {self.model_name}")
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create new one"""
        if self.vector_store_type == "faiss":
            self._load_or_create_faiss()
        elif self.vector_store_type == "chroma":
            self._load_or_create_chroma()
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
    
    def _load_or_create_faiss(self):
        """Load or create FAISS vector store"""
        index_path = Path(settings.faiss_index_path)
        
        if index_path.exists() and (index_path / "index.faiss").exists():
            try:
                logger.info("Loading existing FAISS index")
                self.vector_store = FAISS.load_local(
                    str(index_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                self._load_metadata()
                logger.info("Successfully loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._create_new_faiss()
        else:
            self._create_new_faiss()
    
    def _create_new_faiss(self):
        """Create new FAISS vector store"""
        logger.info("Creating new FAISS vector store")
        # Create with a minimal document to avoid empty list error
        self.vector_store = FAISS.from_texts(
            ["Initial document"],
            self.embedding_model
        )
        # Remove the initial document immediately
        try:
            self.vector_store.delete(self.vector_store.index_to_docstore_id[0])
        except:
            pass  # Ignore if deletion fails
        self._save_faiss()
    
    def _load_or_create_chroma(self):
        """Load or create Chroma vector store"""
        chroma_path = Path("data/chroma_db")
        try:
            if chroma_path.exists():
                logger.info("Loading existing Chroma database")
                self.vector_store = Chroma(
                    persist_directory=str(chroma_path),
                    embedding_function=self.embedding_model
                )
                self._load_metadata()
            else:
                logger.info("Creating new Chroma database")
                self.vector_store = Chroma(
                    persist_directory=str(chroma_path),
                    embedding_function=self.embedding_model
                )
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise
    
    def _load_metadata(self):
        """Load metadata from file"""
        metadata_path = Path(settings.faiss_index_path) / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata_store)} documents")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.metadata_store = {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        metadata_path = Path(settings.faiss_index_path) / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, indent=2, ensure_ascii=False)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def add_policy(self, policy: InsurancePolicy):
        """Add a single policy to the vector store"""
        logger.info(f"Adding policy {policy.policy_id} to vector store")
        
        # Convert policy sections to LangChain documents
        documents = []
        for section in policy.sections:
            doc = Document(
                page_content=section.content,
                metadata={
                    'policy_id': policy.policy_id,
                    'insurer_name': policy.insurer_name,
                    'policy_type': policy.policy_type,
                    'document_title': policy.document_title,
                    'section_title': section.title,
                    'section_type': section.section_type,
                    'page_number': section.page_number,
                    'file_path': policy.file_path,
                    'created_at': policy.created_at
                }
            )
            documents.append(doc)
        
        # Add to vector store
        if self.vector_store_type == "faiss":
            self.vector_store.add_documents(documents)
            self._save_faiss()
        else:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
        
        # Store metadata
        self.metadata_store[policy.policy_id] = {
            'insurer_name': policy.insurer_name,
            'policy_type': policy.policy_type,
            'document_title': policy.document_title,
            'sections_count': len(policy.sections),
            'file_path': policy.file_path,
            'created_at': policy.created_at
        }
        self._save_metadata()
        
        logger.info(f"Successfully added policy {policy.policy_id} with {len(documents)} sections")
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add text documents to the vector store"""
        logger.info(f"Adding {len(texts)} text documents to vector store")
        
        # Create documents from texts
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Add to vector store
        if self.vector_store_type == "faiss":
            self.vector_store.add_documents(documents)
            self._save_faiss()
        else:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
        
        logger.info(f"Successfully added {len(documents)} text documents")
    
    def add_policies_batch(self, policies: List[InsurancePolicy]):
        """Add multiple policies to the vector store efficiently"""
        logger.info(f"Adding {len(policies)} policies to vector store in batch")
        
        all_documents = []
        
        for policy in policies:
            # Convert policy sections to LangChain documents
            for section in policy.sections:
                doc = Document(
                    page_content=section.content,
                    metadata={
                        'policy_id': policy.policy_id,
                        'insurer_name': policy.insurer_name,
                        'policy_type': policy.policy_type,
                        'document_title': policy.document_title,
                        'section_title': section.title,
                        'section_type': section.section_type,
                        'page_number': section.page_number,
                        'file_path': policy.file_path,
                        'created_at': policy.created_at
                    }
                )
                all_documents.append(doc)
            
            # Store metadata
            self.metadata_store[policy.policy_id] = {
                'insurer_name': policy.insurer_name,
                'policy_type': policy.policy_type,
                'document_title': policy.document_title,
                'sections_count': len(policy.sections),
                'file_path': policy.file_path,
                'created_at': policy.created_at
            }
        
        # Add all documents to vector store
        if self.vector_store_type == "faiss":
            self.vector_store.add_documents(all_documents)
            self._save_faiss()
        else:
            self.vector_store.add_documents(all_documents)
            self.vector_store.persist()
        
        self._save_metadata()
        logger.info(f"Successfully added {len(policies)} policies with {len(all_documents)} total sections")
    
    def search(self, query: str, top_k: int = 5,
               policy_types: List[str] = None,
               insurers: List[str] = None) -> List[SearchResult]:
        """Search for relevant policy sections"""
        logger.info(f"Searching for: {query}")
        
        try:
            # Perform similarity search
            if self.vector_store_type == "faiss":
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query, k=top_k * 2  # Get more results for filtering
                )
            else:
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query, k=top_k * 2
                )
            
            # Convert to SearchResult objects
            results = []
            for doc, score in docs_and_scores:
                # Apply filters
                if policy_types and doc.metadata.get('policy_type') not in policy_types:
                    continue
                if insurers and doc.metadata.get('insurer_name') not in insurers:
                    continue
                
                result = SearchResult(
                    policy_id=doc.metadata.get('policy_id', 'Unknown'),
                    insurer_name=doc.metadata.get('insurer_name', 'Unknown'),
                    policy_type=doc.metadata.get('policy_type', 'Unknown'),
                    document_title=doc.metadata.get('document_title', 'Unknown'),
                    section_title=doc.metadata.get('section_title', 'Unknown'),
                    section_content=doc.page_content,
                    section_type=doc.metadata.get('section_type', 'Unknown'),
                    similarity_score=float(score),
                    page_number=doc.metadata.get('page_number', 0),
                    metadata=doc.metadata
                )
                results.append(result)
            
            # Sort by similarity score and return top_k
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", **kwargs):
        """Get a LangChain retriever for advanced retrieval operations"""
        if search_type == "similarity":
            return self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=kwargs
            )
        elif search_type == "mmr":
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs=kwargs
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
    
    def _save_faiss(self):
        """Save FAISS index to disk"""
        if self.vector_store_type == "faiss":
            try:
                index_path = Path(settings.faiss_index_path)
                index_path.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(index_path))
                logger.info("FAISS index saved successfully")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
    
    def clear_index(self):
        """Clear the entire vector store"""
        logger.warning("Clearing entire vector store")
        
        if self.vector_store_type == "faiss":
            self._create_new_faiss()
        else:
            # For Chroma, we'd need to delete the database
            chroma_path = Path("data/chroma_db")
            if chroma_path.exists():
                import shutil
                shutil.rmtree(chroma_path)
                logger.info("Chroma database deleted")
        
        self.metadata_store = {}
        self._save_metadata()
        logger.info("Vector store cleared successfully")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            if self.vector_store_type == "faiss":
                total_vectors = len(self.vector_store.index_to_docstore_id)
            else:
                # For Chroma, get collection count
                total_vectors = self.vector_store._collection.count()
            
            return {
                "vector_store_type": self.vector_store_type,
                "total_vectors": total_vectors,
                "total_policies": len(self.metadata_store),
                "embedding_model": self.model_name,
                "policy_types": list(set(
                    meta.get('policy_type', 'Unknown') 
                    for meta in self.metadata_store.values()
                )),
                "insurers": list(set(
                    meta.get('insurer_name', 'Unknown') 
                    for meta in self.metadata_store.values()
                )),
                "last_updated": max(
                    (meta.get('created_at', '') for meta in self.metadata_store.values()),
                    default=''
                )
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "error": str(e),
                "vector_store_type": self.vector_store_type,
                "embedding_model": self.model_name
            }
    
    def export_embeddings(self, output_path: str = "data/embeddings.json"):
        """Export embeddings and metadata for analysis"""
        logger.info("Exporting embeddings and metadata")
        
        try:
            if self.vector_store_type == "faiss":
                # Get all documents and their embeddings
                docs = []
                for doc_id in self.vector_store.index_to_docstore_id.values():
                    doc = self.vector_store.docstore._dict.get(doc_id)
                    if doc:
                        docs.append({
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        })
                
                export_data = {
                    'embeddings_count': len(docs),
                    'documents': docs,
                    'metadata_store': self.metadata_store,
                    'export_date': str(Path().cwd()),
                    'vector_store_type': self.vector_store_type
                }
                
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Exported embeddings to {output_path}")
                return str(output_path)
            else:
                logger.warning("Export not implemented for Chroma yet")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export embeddings: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore()
    
    # Print statistics
    stats = vector_store.get_statistics()
    print("Vector Store Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Example search
    query = "What are the exclusions for health insurance?"
    results = vector_store.search(query, top_k=3)
    
    print(f"\nSearch Results for: {query}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.section_title}")
        print(f"   Policy: {result.document_title}")
        print(f"   Insurer: {result.insurer_name}")
        print(f"   Score: {result.similarity_score:.4f}")
        print(f"   Content: {result.section_content[:100]}...")
