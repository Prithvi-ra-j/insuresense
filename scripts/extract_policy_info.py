#!/usr/bin/env python3
"""
Extract Policy Information from LIC PDFs
Extracts Benefits, Exclusions, Renewal Terms from PDFs and saves to CSV
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
import re
from datetime import datetime

# Import our document processor
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyExtractor:
    """Extract key information from insurance policy PDFs"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        
        # Keywords for different sections
        self.section_keywords = {
            'benefits': [
                'benefits', 'coverage', 'what is covered', 'policy benefits', 
                'sum assured', 'death benefit', 'maturity benefit', 'survival benefit',
                'accidental benefit', 'critical illness benefit', 'disability benefit'
            ],
            'exclusions': [
                'exclusions', 'what is not covered', 'limitations', 'restrictions',
                'not covered', 'excluded', 'exclusion clause', 'limitations and exclusions'
            ],
            'renewal': [
                'renewal', 'renewal terms', 'renewal conditions', 'renewal premium',
                'renewal process', 'renewal date', 'renewal period', 'renewal option'
            ],
            'premium': [
                'premium', 'premium amount', 'premium payment', 'premium rates',
                'premium calculation', 'premium terms', 'payment frequency'
            ],
            'terms': [
                'terms and conditions', 'general conditions', 'policy conditions',
                'definitions', 'clauses', 'policy terms'
            ]
        }
    
    def extract_from_pdfs(self, pdf_dir: str, output_file: str) -> pd.DataFrame:
        """Extract information from all PDFs in directory"""
        pdf_dir = Path(pdf_dir)
        extracted_data = []
        
        # Get all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                
                # Process the PDF
                policy = self.document_processor.process_document(str(pdf_file))
                
                # Extract key information
                extracted_info = self._extract_key_information(policy)
                
                # Add file metadata
                extracted_info['file_name'] = pdf_file.name
                extracted_info['file_size'] = pdf_file.stat().st_size
                extracted_info['extraction_date'] = datetime.now().isoformat()
                
                extracted_data.append(extracted_info)
                logger.info(f"Successfully extracted info from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(extracted_data)
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Extracted data saved to {output_path}")
        logger.info(f"Total policies processed: {len(df)}")
        
        return df
    
    def _extract_key_information(self, policy) -> Dict[str, Any]:
        """Extract key information from a policy object"""
        extracted_info = {
            'policy_id': policy.policy_id,
            'insurer_name': policy.insurer_name,
            'policy_type': policy.policy_type,
            'document_title': policy.document_title,
            'total_pages': policy.total_pages,
            'sections_count': len(policy.sections)
        }
        
        # Extract text content
        full_text = policy.extracted_text.lower()
        
        # Extract benefits
        benefits = self._extract_section_content(policy, 'benefits')
        extracted_info['benefits_text'] = benefits
        extracted_info['benefits_summary'] = self._summarize_section(benefits)
        
        # Extract exclusions
        exclusions = self._extract_section_content(policy, 'exclusions')
        extracted_info['exclusions_text'] = exclusions
        extracted_info['exclusions_summary'] = self._summarize_section(exclusions)
        
        # Extract renewal terms
        renewal = self._extract_section_content(policy, 'renewal')
        extracted_info['renewal_text'] = renewal
        extracted_info['renewal_summary'] = self._summarize_section(renewal)
        
        # Extract premium information
        premium = self._extract_section_content(policy, 'premium')
        extracted_info['premium_text'] = premium
        extracted_info['premium_summary'] = self._summarize_section(premium)
        
        # Extract general terms
        terms = self._extract_section_content(policy, 'terms')
        extracted_info['terms_text'] = terms
        extracted_info['terms_summary'] = self._summarize_section(terms)
        
        # Extract key metrics
        extracted_info.update(self._extract_key_metrics(full_text))
        
        return extracted_info
    
    def _extract_section_content(self, policy, section_type: str) -> str:
        """Extract content for a specific section type"""
        content_parts = []
        
        for section in policy.sections:
            if section.section_type == section_type:
                content_parts.append(section.content)
            elif any(keyword in section.title.lower() for keyword in self.section_keywords[section_type]):
                content_parts.append(section.content)
        
        return "\n\n".join(content_parts) if content_parts else ""
    
    def _summarize_section(self, text: str, max_length: int = 500) -> str:
        """Create a summary of section content"""
        if not text:
            return ""
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Take first few sentences
        sentences = text.split('.')
        summary = '. '.join(sentences[:3]) + '.'
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def _extract_key_metrics(self, text: str) -> Dict[str, Any]:
        """Extract key metrics from text"""
        metrics = {}
        
        # Extract sum assured amounts
        sum_assured_patterns = [
            r'rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|lac|lakhs|lacs)?',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh|lac|lakhs|lacs)',
            r'sum assured[:\s]*rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in sum_assured_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['sum_assured_amounts'] = matches[:3]  # Take first 3 matches
                break
        
        # Extract premium amounts
        premium_patterns = [
            r'premium[:\s]*rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*premium',
        ]
        
        for pattern in premium_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['premium_amounts'] = matches[:3]
                break
        
        # Extract policy terms
        term_patterns = [
            r'(\d+)\s*(?:years?|yrs?)',
            r'term[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'policy term[:\s]*(\d+)\s*(?:years?|yrs?)',
        ]
        
        for pattern in term_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Filter reasonable policy terms (1-50 years)
                valid_terms = [int(m) for m in matches if 1 <= int(m) <= 50]
                if valid_terms:
                    metrics['policy_terms'] = valid_terms[:3]
                break
        
        # Extract age limits
        age_patterns = [
            r'age[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*age',
            r'minimum age[:\s]*(\d+)',
            r'maximum age[:\s]*(\d+)',
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Filter reasonable ages (1-100)
                valid_ages = [int(m) for m in matches if 1 <= int(m) <= 100]
                if valid_ages:
                    metrics['age_limits'] = valid_ages[:5]
                break
        
        return metrics

def main():
    """Main function to extract policy information"""
    print("Starting Policy Information Extraction...")
    
    # Initialize extractor
    extractor = PolicyExtractor()
    
    # Define paths
    pdf_dir = "data/lic_policies"
    output_file = "data/extracted/lic_policies_extracted.csv"
    
    # Extract information
    df = extractor.extract_from_pdfs(pdf_dir, output_file)
    
    # Print summary
    print(f"\nExtraction completed!")
    print(f"Total policies processed: {len(df)}")
    print(f"Output saved to: {output_file}")
    
    # Show sample data
    if not df.empty:
        print(f"\nSample extracted data:")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample policy: {df.iloc[0]['document_title']}")
        print(f"Policy type: {df.iloc[0]['policy_type']}")
        print(f"Sections found: {df.iloc[0]['sections_count']}")

if __name__ == "__main__":
    main()
