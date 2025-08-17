#!/usr/bin/env python3
"""
Merge Datasets
Combines extracted policy information, cleaned structured data, and claims data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetMerger:
    """Merge different datasets into a comprehensive final dataset"""
    
    def __init__(self):
        self.merged_data = None
    
    def load_extracted_policies(self, file_path: str) -> pd.DataFrame:
        """Load extracted policy information"""
        logger.info(f"Loading extracted policies: {file_path}")
        
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} extracted policies")
            return df
        else:
            logger.warning(f"Extracted policies file not found: {file_path}")
            return pd.DataFrame()
    
    def load_cleaned_structured_data(self, dir_path: str) -> Dict[str, pd.DataFrame]:
        """Load all cleaned structured datasets"""
        logger.info(f"Loading cleaned structured data from: {dir_path}")
        
        datasets = {}
        dir_path = Path(dir_path)
        
        for file_path in dir_path.glob("cleaned_*.csv"):
            try:
                df = pd.read_csv(file_path)
                dataset_name = file_path.stem.replace('cleaned_', '')
                datasets[dataset_name] = df
                logger.info(f"Loaded {dataset_name}: {len(df)} records")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
        
        return datasets
    
    def merge_datasets(self, extracted_policies: pd.DataFrame, 
                      structured_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a comprehensive final dataset"""
        logger.info("Starting dataset merge...")
        
        # Start with extracted policies as base
        if not extracted_policies.empty:
            merged_df = extracted_policies.copy()
            logger.info(f"Base dataset: {len(merged_df)} extracted policies")
        else:
            # Create empty base if no extracted policies
            merged_df = pd.DataFrame()
            logger.info("No extracted policies found, creating empty base")
        
        # Add structured data as additional columns
        for dataset_name, df in structured_data.items():
            if not df.empty:
                # Add prefix to column names to avoid conflicts
                df_prefixed = df.copy()
                df_prefixed.columns = [f"{dataset_name}_{col}" for col in df_prefixed.columns]
                
                # Merge with base dataset
                if merged_df.empty:
                    merged_df = df_prefixed
                else:
                    # Use outer merge to keep all records
                    merged_df = pd.merge(merged_df, df_prefixed, 
                                       how='outer', left_index=True, right_index=True)
                
                logger.info(f"Added {dataset_name}: {len(df)} records")
        
        # Add metadata columns
        merged_df['data_source'] = 'merged_dataset'
        merged_df['merge_date'] = pd.Timestamp.now().isoformat()
        merged_df['total_records'] = len(merged_df)
        
        logger.info(f"Final merged dataset: {len(merged_df)} records, {len(merged_df.columns)} columns")
        return merged_df
    
    def create_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for the merged dataset"""
        logger.info("Creating summary statistics...")
        
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
            'column_names': list(df.columns),
            'sample_data': df.head(3).to_dict('records') if not df.empty else []
        }
        
        # Add dataset-specific statistics
        if 'policy_type' in df.columns:
            stats['policy_types'] = df['policy_type'].value_counts().to_dict()
        
        if 'insurer_name' in df.columns:
            stats['insurers'] = df['insurer_name'].value_counts().to_dict()
        
        return stats
    
    def save_final_dataset(self, df: pd.DataFrame, output_file: str):
        """Save the final merged dataset"""
        logger.info(f"Saving final dataset to: {output_file}")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Save summary statistics
        stats = self.create_summary_statistics(df)
        stats_file = output_path.parent / "dataset_summary.json"
        
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Dataset and summary saved successfully")
        return stats
    
    def process_all_datasets(self, extracted_dir: str, structured_dir: str, 
                           output_file: str) -> Dict[str, Any]:
        """Process all datasets and create final merged dataset"""
        
        # Load extracted policies
        extracted_file = Path(extracted_dir) / "lic_policies_extracted.csv"
        extracted_policies = self.load_extracted_policies(str(extracted_file))
        
        # Load cleaned structured data
        structured_data = self.load_cleaned_structured_data(structured_dir)
        
        # Merge datasets
        merged_df = self.merge_datasets(extracted_policies, structured_data)
        
        # Save final dataset
        stats = self.save_final_dataset(merged_df, output_file)
        
        return stats

def main():
    """Main function to merge datasets"""
    print("Starting Dataset Merge...")
    
    # Initialize merger
    merger = DatasetMerger()
    
    # Define paths
    extracted_dir = "data/extracted"
    structured_dir = "data/extracted"  # Cleaned files are in extracted dir
    output_file = "data/extracted/final_dataset.csv"
    
    # Process all datasets
    stats = merger.process_all_datasets(extracted_dir, structured_dir, output_file)
    
    # Print summary
    print(f"\nDataset merge completed!")
    print(f"Final dataset statistics:")
    print(f"  - Total records: {stats['total_records']}")
    print(f"  - Total columns: {stats['total_columns']}")
    print(f"  - Missing values: {stats['missing_values']}")
    print(f"  - Duplicate records: {stats['duplicate_records']}")
    print(f"Output saved to: {output_file}")
    
    # Show data types
    print(f"\nData types:")
    for dtype, count in stats['data_types'].items():
        print(f"  - {dtype}: {count} columns")
    
    # Show sample columns
    print(f"\nSample columns:")
    for col in stats['column_names'][:10]:
        print(f"  - {col}")
    if len(stats['column_names']) > 10:
        print(f"  ... and {len(stats['column_names']) - 10} more columns")

if __name__ == "__main__":
    main()
