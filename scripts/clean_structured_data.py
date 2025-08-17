#!/usr/bin/env python3
"""
Clean and Normalize Structured Datasets
Removes duplicates, fixes column names, and normalizes data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and normalize structured datasets"""
    
    def __init__(self):
        self.cleaned_data = {}
    
    def clean_life_insurance_data(self, file_path: str) -> pd.DataFrame:
        """Clean the Life Insurance Policy Data CSV"""
        logger.info(f"Cleaning Life Insurance Policy Data: {file_path}")
        
        # Read the data
        df = pd.read_csv(file_path)
        logger.info(f"Original shape: {df.shape}")
        
        # Clean column names
        df.columns = self._clean_column_names(df.columns)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize data types
        df = self._normalize_data_types(df)
        
        # Clean text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).apply(self._clean_text)
        
        logger.info(f"Final cleaned shape: {df.shape}")
        return df
    
    def clean_car_insurance_claims(self, file_path: str) -> pd.DataFrame:
        """Clean the Car Insurance Claims CSV"""
        logger.info(f"Cleaning Car Insurance Claims Data: {file_path}")
        
        # Read the data
        df = pd.read_csv(file_path)
        logger.info(f"Original shape: {df.shape}")
        
        # Clean column names
        df.columns = self._clean_column_names(df.columns)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize data types
        df = self._normalize_data_types(df)
        
        # Clean text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).apply(self._clean_text)
        
        logger.info(f"Final cleaned shape: {df.shape}")
        return df
    
    def clean_bajaj_allianz_data(self, file_path: str) -> pd.DataFrame:
        """Clean the Bajaj Allianz Excel file"""
        logger.info(f"Cleaning Bajaj Allianz Data: {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        logger.info(f"Original shape: {df.shape}")
        
        # Clean column names
        df.columns = self._clean_column_names(df.columns)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize data types
        df = self._normalize_data_types(df)
        
        # Clean text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).apply(self._clean_text)
        
        logger.info(f"Final cleaned shape: {df.shape}")
        return df
    
    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """Clean column names"""
        cleaned_columns = []
        
        for col in columns:
            # Convert to string
            col_str = str(col)
            
            # Remove special characters and replace with underscore
            col_clean = re.sub(r'[^a-zA-Z0-9\s]', '_', col_str)
            
            # Replace multiple spaces/underscores with single underscore
            col_clean = re.sub(r'[\s_]+', '_', col_clean)
            
            # Remove leading/trailing underscores
            col_clean = col_clean.strip('_')
            
            # Convert to lowercase
            col_clean = col_clean.lower()
            
            # Handle empty column names
            if not col_clean:
                col_clean = f'column_{len(cleaned_columns)}'
            
            # Handle duplicate column names
            if col_clean in cleaned_columns:
                col_clean = f'{col_clean}_{cleaned_columns.count(col_clean) + 1}'
            
            cleaned_columns.append(col_clean)
        
        return cleaned_columns
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _normalize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data types"""
        # Convert date-like columns to datetime
        date_patterns = ['date', 'time', 'created', 'updated', 'start', 'end']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Convert numeric strings to numbers
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text) or text == 'nan':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\-\_\&\(\)]', '', text)
        
        return text
    
    def process_all_datasets(self, input_dir: str, output_dir: str):
        """Process all datasets in the input directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_files = []
        
        # Process CSV files
        for csv_file in input_path.glob("*.csv"):
            try:
                if "life_insurance" in csv_file.name.lower():
                    df = self.clean_life_insurance_data(str(csv_file))
                elif "car_insurance" in csv_file.name.lower():
                    df = self.clean_car_insurance_claims(str(csv_file))
                else:
                    df = self.clean_life_insurance_data(str(csv_file))  # Default
                
                # Save cleaned data
                output_file = output_path / f"cleaned_{csv_file.name}"
                df.to_csv(output_file, index=False, encoding='utf-8')
                processed_files.append(output_file)
                
                logger.info(f"Saved cleaned data to: {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to process {csv_file.name}: {e}")
        
        # Process Excel files
        for excel_file in input_path.glob("*.xlsx"):
            try:
                df = self.clean_bajaj_allianz_data(str(excel_file))
                
                # Save cleaned data
                output_file = output_path / f"cleaned_{excel_file.stem}.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
                processed_files.append(output_file)
                
                logger.info(f"Saved cleaned data to: {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to process {excel_file.name}: {e}")
        
        return processed_files

def main():
    """Main function to clean structured datasets"""
    print("Starting Data Cleaning and Normalization...")
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Define paths
    input_dir = "data/structured_data"
    output_dir = "data/extracted"
    
    # Process all datasets
    processed_files = cleaner.process_all_datasets(input_dir, output_dir)
    
    # Print summary
    print(f"\nData cleaning completed!")
    print(f"Total files processed: {len(processed_files)}")
    print(f"Cleaned files saved to: {output_dir}")
    
    for file in processed_files:
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
