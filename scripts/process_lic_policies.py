#!/usr/bin/env python3
"""
Process Existing LIC Policies
Enhanced processing of LIC policy PDFs with better section extraction
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any
import re
from datetime import datetime
import sys
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPolicyExtractor:
    """Enhanced extractor for LIC policy information"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        
        # Enhanced keywords for different sections
        self.section_keywords = {
            'benefits': [
                'benefits', 'coverage', 'what is covered', 'policy benefits', 
                'sum assured', 'death benefit', 'maturity benefit', 'survival benefit',
                'accidental benefit', 'critical illness benefit', 'disability benefit',
                'amount payable', 'benefit payable', 'payout'
            ],
            'exclusions': [
                'exclusions', 'what is not covered', 'limitations', 'restrictions',
                'not covered', 'excluded', 'exclusion clause', 'limitations and exclusions',
                'we shall not pay', 'not payable', 'void'
            ],
            'premium': [
                'premium', 'premium amount', 'premium payment', 'premium rates',
                'premium calculation', 'premium terms', 'payment frequency',
                'how to pay', 'mode of payment', 'due date'
            ],
            'terms': [
                'terms and conditions', 'general conditions', 'policy conditions',
                'definitions', 'clauses', 'policy terms', 'conditions'
            ],
            'renewal': [
                'renewal', 'renewal terms', 'renewal conditions', 'renewal premium',
                'renewal process', 'renewal date', 'renewal period', 'renewal option',
                'continuation', 'revival'
            ],
            'claim': [
                'claim', 'claim procedure', 'how to claim', 'claim process',
                'documents required', 'claim settlement', 'intimation of claim'
            ]
        }
    
    def process_lic_policies(self, pdf_dir: str, output_file: str, processed_dir: str = "insuresense/data/processed") -> pd.DataFrame:
        """Process all LIC policy PDFs in directory"""
        pdf_dir = Path(pdf_dir)
        processed_data = []
        
        # Get all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} LIC policy PDF files to process")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing ({i}/{len(pdf_files)}): {pdf_file.name}")
                
                # Process the PDF
                policy = self.document_processor.process_document(str(pdf_file))
                
                # Save individual policy JSON
                self._save_policy_json(policy, processed_dir)
                
                # Extract enhanced information
                extracted_info = self._extract_enhanced_information(policy)
                
                # Add file metadata
                extracted_info['file_name'] = pdf_file.name
                extracted_info['file_size'] = pdf_file.stat().st_size
                extracted_info['processing_date'] = datetime.now().isoformat()
                
                processed_data.append(extracted_info)
                logger.info(f"Successfully processed {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                # Add error entry
                error_info = {
                    'file_name': pdf_file.name,
                    'error': str(e),
                    'processing_date': datetime.now().isoformat()
                }
                processed_data.append(error_info)
                continue
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Processed data saved to {output_path}")
        logger.info(f"Total policies processed: {len(df)}")
        
        return df
    
    def _save_policy_json(self, policy, processed_dir: str):
        """Save policy object as JSON file"""
        # Create processed directory if it doesn't exist
        processed_path = Path(processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename from policy ID
        filename = f"{getattr(policy, 'policy_id', 'unknown')}.json"
        file_path = processed_path / filename
        
        # Convert policy to dictionary
        policy_dict = {}
        for attr in dir(policy):
            if not attr.startswith('_'):
                value = getattr(policy, attr)
                # Convert non-serializable objects to string representations
                if isinstance(value, (str, int, float, bool, type(None))):
                    policy_dict[attr] = value
                elif isinstance(value, list):
                    policy_dict[attr] = [self._serialize_section(s) for s in value]
                else:
                    policy_dict[attr] = str(value)
        
        # Save to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(policy_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved policy JSON: {file_path}")
    
    def _serialize_section(self, section):
        """Serialize a policy section to a dictionary"""
        if hasattr(section, '__dict__'):
            return {k: v for k, v in section.__dict__.items() if isinstance(v, (str, int, float, bool, type(None), list))}
        else:
            return str(section)
    
    def _extract_enhanced_information(self, policy) -> Dict[str, Any]:
        """Extract enhanced information from a policy object"""
        extracted_info = {
            'policy_id': getattr(policy, 'policy_id', ''),
            'insurer_name': getattr(policy, 'insurer_name', ''),
            'policy_type': getattr(policy, 'policy_type', ''),
            'document_title': getattr(policy, 'document_title', ''),
            'total_pages': getattr(policy, 'total_pages', 0),
            'sections_count': len(getattr(policy, 'sections', []))
        }
        
        # Extract text content by section type
        for section_type in self.section_keywords.keys():
            content = self._extract_section_content(policy, section_type)
            extracted_info[f'{section_type}_text'] = content
            extracted_info[f'{section_type}_summary'] = self._summarize_section(content)
        
        # Extract key metrics from full text
        full_text = getattr(policy, 'extracted_text', '').lower()
        extracted_info.update(self._extract_key_metrics(full_text))
        
        return extracted_info
    
    def _extract_section_content(self, policy, section_type: str) -> str:
        """Extract content for a specific section type with enhanced matching"""
        content_parts = []
        
        # Check if policy has sections attribute
        if not hasattr(policy, 'sections'):
            return ""
        
        for section in policy.sections:
            # Direct section type match
            if hasattr(section, 'section_type') and section.section_type == section_type:
                content_parts.append(getattr(section, 'content', ''))
            # Title-based matching
            elif hasattr(section, 'title'):
                section_title = section.title.lower()
                # Check if any keyword for this section type is in the title
                if any(keyword in section_title for keyword in self.section_keywords[section_type]):
                    content_parts.append(getattr(section, 'content', ''))
                # Additional check for content-based matching
                elif hasattr(section, 'content'):
                    section_content = section.content.lower()
                    if any(keyword in section_content for keyword in self.section_keywords[section_type][:3]):
                        content_parts.append(section.content)
        
        return "\n\n".join(content_parts) if content_parts else ""
    
    def _summarize_section(self, text: str, max_length: int = 500) -> str:
        """Create a summary of section content"""
        if not text:
            return ""
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # If text is short enough, return as is
        if len(text) <= max_length:
            return text
        
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
                metrics['sum_assured_amounts'] = matches[:5]  # Take first 5 matches
                break
        
        # Extract premium amounts
        premium_patterns = [
            r'premium[:\s]*rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*premium',
            r'annual premium[:\s]*rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in premium_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['premium_amounts'] = matches[:5]
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
                    metrics['policy_terms'] = valid_terms[:5]
                break
        
        # Extract age limits
        age_patterns = [
            r'age[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*age',
            r'minimum age[:\s]*(\d+)',
            r'maximum age[:\s]*(\d+)',
            r'entry age[:\s]*(\d+)',
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
    """Main function to process LIC policies"""
    print("Starting Enhanced LIC Policy Processing...")
    print("=" * 50)
    
    # Initialize extractor
    extractor = EnhancedPolicyExtractor()
    
    # Define paths
    pdf_dir = "insuresense/lic_policies"  # Correct path to PDF directory
    output_file = "insuresense/data/extracted/enhanced_lic_policies.csv"
    processed_dir = "insuresense/data/processed"
    
    print(f"PDF Directory: {pdf_dir}")
    print(f"Output File: {output_file}")
    print(f"Processed JSON Directory: {processed_dir}")
    
    # Process policies
    try:
        df = extractor.process_lic_policies(pdf_dir, output_file, processed_dir)
        
        # Print summary
        print(f"\nProcessing completed successfully!")
        print(f"Total policies processed: {len(df)}")
        print(f"Output saved to: {output_file}")
        
        # Show sample data
        if not df.empty:
            print(f"\nSample extracted data:")
            print(f"Columns: {list(df.columns)}")
            if 'document_title' in df.columns and not df['document_title'].empty:
                print(f"Sample policy: {df.iloc[0]['document_title']}")
            if 'policy_type' in df.columns and not df['policy_type'].empty:
                print(f"Policy type: {df.iloc[0]['policy_type']}")
            if 'sections_count' in df.columns and not df['sections_count'].empty:
                print(f"Sections found: {df.iloc[0]['sections_count']}")
            
            # Show some statistics
            successful_count = len(df) - len(df[df.get('error', '').notna()]) if 'error' in df.columns else len(df)
            print(f"Successfully processed: {successful_count}/{len(df)}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        logger.error(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())