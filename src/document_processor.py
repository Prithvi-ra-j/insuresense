"""
Document processing pipeline for insurance policies
Handles PDF parsing, text extraction, and structuring
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from urllib.parse import urljoin, urlparse

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredFileLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PolicySection:
    """Represents a section of an insurance policy"""
    title: str
    content: str
    page_number: int
    section_type: str
    metadata: Dict[str, Any] = None

@dataclass
class InsurancePolicy:
    """Represents a complete insurance policy document"""
    policy_id: str
    insurer_name: str
    policy_type: str
    document_title: str
    total_pages: int
    sections: List[PolicySection]
    extracted_text: str
    metadata: Dict[str, Any]
    created_at: str
    file_path: str

class DocumentProcessor:
    """Enhanced document processor using LangChain components"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Section type keywords for classification
        self.section_keywords = {
            'coverage': ['coverage', 'benefits', 'what is covered', 'policy benefits', 'sum assured'],
            'exclusion': ['exclusions', 'what is not covered', 'limitations', 'restrictions'],
            'premium': ['premium', 'payment', 'cost', 'fees', 'charges'],
            'terms': ['terms and conditions', 'conditions', 'definitions', 'clauses'],
            'general': ['general', 'introduction', 'overview', 'contact', 'disclaimer']
        }
    
    def download_irdai_policies(self, max_policies: int = 10) -> List[str]:
        """Download insurance policies from IRDAI website"""
        logger.info(f"Downloading up to {max_policies} policies from IRDAI")
        
        # For now, create sample policies since actual IRDAI scraping requires more complex setup
        sample_policies = self._create_sample_policies(max_policies)
        
        # In production, you would implement actual IRDAI scraping here
        # Example structure:
        # irdai_urls = [
        #     "https://www.irdai.gov.in/...",
        #     "https://www.irdai.gov.in/..."
        # ]
        # return self._download_from_urls(irdai_urls)
        
        return sample_policies
    
    def _create_sample_policies(self, count: int) -> List[str]:
        """Create sample policy documents for testing"""
        sample_policies = []
        
        for i in range(count):
            policy_content = f"""
            INSURANCE POLICY DOCUMENT - SAMPLE {i+1}
            
            POLICY DETAILS
            Policy Number: SAMPLE-{i+1:04d}
            Insurer: Sample Insurance Company {i+1}
            Policy Type: {'Life Insurance' if i % 2 == 0 else 'Health Insurance'}
            Issue Date: {datetime.now().strftime('%Y-%m-%d')}
            
            COVERAGE DETAILS
            This policy provides comprehensive coverage for various risks including:
            - Death benefit up to Rs. 10,00,000
            - Critical illness coverage
            - Accidental death benefit
            - Disability coverage
            
            EXCLUSIONS
            The following are not covered under this policy:
            - Pre-existing conditions
            - Self-inflicted injuries
            - War and terrorism
            - Nuclear incidents
            
            PREMIUM DETAILS
            Annual Premium: Rs. {5000 + (i * 500)}
            Payment Frequency: Annual
            Grace Period: 30 days
            
            TERMS AND CONDITIONS
            1. The policyholder must disclose all material facts
            2. Premium must be paid on time
            3. Claims must be reported within 30 days
            4. Policy can be renewed annually
            
            CONTACT INFORMATION
            Customer Service: 1800-SAMPLE-{i+1}
            Email: support@sample{i+1}.com
            Address: Sample Street, Sample City, Sample State
            """
            
            # Save sample policy to file
            file_path = f"data/irdai_policies/sample_policy_{i+1}.txt"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(policy_content)
            
            sample_policies.append(file_path)
            logger.info(f"Created sample policy: {file_path}")
        
        return sample_policies
    
    def _download_from_urls(self, urls: List[str]) -> List[str]:
        """Download documents from URLs (placeholder for actual implementation)"""
        downloaded_files = []
        
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Determine file type and save
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or f"downloaded_{len(downloaded_files)}.pdf"
                file_path = f"data/irdai_policies/{filename}"
                
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(file_path)
                logger.info(f"Downloaded: {url} -> {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
        
        return downloaded_files
    
    def process_document(self, file_path: str) -> InsurancePolicy:
        """Process a document using LangChain loaders and return structured policy"""
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Load document using appropriate LangChain loader
            document = self._load_document(file_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Extract sections from chunks
            sections = self._extract_sections_from_chunks(chunks)
            
            # Create policy object
            policy = self._create_policy_object(file_path, document.page_content, sections, len(chunks))
            
            logger.info(f"Successfully processed {file_path}: {len(sections)} sections")
            return policy
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
    
    def _load_document(self, file_path: str) -> Document:
        """Load document using appropriate LangChain loader"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                # Try PyPDFLoader first, fallback to UnstructuredFileLoader
                try:
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                    if documents:
                        return documents[0]
                except Exception as e:
                    logger.warning(f"PyPDFLoader failed, trying UnstructuredFileLoader: {e}")
                
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                if documents:
                    return documents[0]
                else:
                    raise ValueError("No content extracted from PDF")
                    
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                return documents[0]
                
            elif file_extension in ['.html', '.htm']:
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                return documents[0]
                
            else:
                # Try UnstructuredFileLoader for other formats
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                return documents[0]
                
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise
    
    def _extract_sections_from_chunks(self, chunks: List[Document]) -> List[PolicySection]:
        """Extract policy sections from document chunks"""
        sections = []
        
        for i, chunk in enumerate(chunks):
            # Try to identify section title from chunk content
            section_title = self._extract_section_title(chunk.page_content)
            section_type = self._classify_section_type(section_title)
            
            section = PolicySection(
                title=section_title,
                content=chunk.page_content,
                page_number=i + 1,
                section_type=section_type,
                metadata=chunk.metadata
            )
            sections.append(section)
        
        return sections
    
    def _extract_section_title(self, content: str) -> str:
        """Extract section title from chunk content"""
        lines = content.strip().split('\n')
        
        # Look for lines that might be headers (all caps, short, ends with colon)
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            if (len(line) < 100 and 
                (line.isupper() or line.endswith(':') or 
                 any(keyword in line.lower() for keyword in ['coverage', 'exclusion', 'premium', 'terms']))):
                return line
        
        # Fallback: use first meaningful line
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                return line[:50] + "..." if len(line) > 50 else line
        
        return "Unknown Section"
    
    def _classify_section_type(self, section_title: str) -> str:
        """Classify section type based on title and content"""
        title_lower = section_title.lower()
        
        for section_type, keywords in self.section_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type
        
        return 'general'
    
    def _create_policy_object(self, file_path: str, text: str, sections: List[PolicySection], total_pages: int) -> InsurancePolicy:
        """Create InsurancePolicy object from processed data"""
        # Extract basic information from text
        insurer_name = self._extract_insurer_name(text)
        policy_type = self._extract_policy_type(text)
        document_title = Path(file_path).stem
        
        # Generate unique policy ID
        policy_id = f"POL-{datetime.now().strftime('%Y%m%d')}-{hash(file_path) % 10000:04d}"
        
        return InsurancePolicy(
            policy_id=policy_id,
            insurer_name=insurer_name,
            policy_type=policy_type,
            document_title=document_title,
            total_pages=total_pages,
            sections=sections,
            extracted_text=text,
            metadata={
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'processing_date': datetime.now().isoformat(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            created_at=datetime.now().isoformat(),
            file_path=file_path
        )
    
    def _extract_insurer_name(self, text: str) -> str:
        """Extract insurer name from text"""
        # Look for common patterns
        patterns = [
            r'insurer[:\s]+([A-Z][A-Za-z\s&]+)',
            r'company[:\s]+([A-Z][A-Za-z\s&]+)',
            r'([A-Z][A-Za-z\s&]+)\s+insurance',
            r'([A-Z][A-Za-z\s&]+)\s+life',
            r'([A-Z][A-Za-z\s&]+)\s+health'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Insurer"
    
    def _extract_policy_type(self, text: str) -> str:
        """Extract policy type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['life insurance', 'life policy']):
            return 'Life Insurance'
        elif any(word in text_lower for word in ['health insurance', 'health policy', 'medical']):
            return 'Health Insurance'
        elif any(word in text_lower for word in ['motor insurance', 'auto', 'car']):
            return 'Motor Insurance'
        elif any(word in text_lower for word in ['property insurance', 'home', 'building']):
            return 'Property Insurance'
        else:
            return 'General Insurance'
    
    def save_processed_policy(self, policy: InsurancePolicy) -> str:
        """Save processed policy to JSON file"""
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{policy.policy_id}.json"
        output_path = output_dir / filename
        
        # Convert to dict for JSON serialization
        policy_dict = asdict(policy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(policy_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed policy to: {output_path}")
        return str(output_path)
    
    def process_directory(self, directory_path: str) -> List[InsurancePolicy]:
        """Process all documents in a directory"""
        policies = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return policies
        
        # Get all supported file types
        supported_extensions = {'.pdf', '.txt', '.md', '.html', '.htm'}
        files = [f for f in directory.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_extensions]
        
        logger.info(f"Found {len(files)} files to process in {directory_path}")
        
        for file_path in files:
            try:
                policy = self.process_document(str(file_path))
                policies.append(policy)
                
                # Save processed policy
                self.save_processed_policy(policy)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(policies)} out of {len(files)} files")
        return policies
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            return {"total_policies": 0, "policy_types": {}, "insurers": {}}
        
        policies = []
        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    policy_data = json.load(f)
                    policies.append(policy_data)
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")
                continue
        
        # Calculate statistics
        policy_types = {}
        insurers = {}
        
        for policy in policies:
            policy_type = policy.get('policy_type', 'Unknown')
            insurer = policy.get('insurer_name', 'Unknown')
            
            policy_types[policy_type] = policy_types.get(policy_type, 0) + 1
            insurers[insurer] = insurers.get(insurer, 0) + 1
        
        return {
            "total_policies": len(policies),
            "policy_types": policy_types,
            "insurers": insurers,
            "last_updated": datetime.now().isoformat()
        }
