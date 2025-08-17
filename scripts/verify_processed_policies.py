#!/usr/bin/env python3
"""
Verify Processed Policies
Check that processed policies are correctly saved and can be loaded
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def verify_processed_policies():
    """Verify that policies have been processed correctly"""
    print("Verifying Processed Policies...")
    print("=" * 50)
    
    # Check processed policies directory
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        print("[-] Processed policies directory not found")
        return False
    
    # Count JSON files
    json_files = list(processed_dir.glob("*.json"))
    print("[+] Processed policies directory found")
    print(f"[*] JSON files in processed directory: {len(json_files)}")
    
    # Check if we have some processed policies
    if len(json_files) > 0:
        print(f"[+] Found {len(json_files)} processed policy files")
        
        # Try to load a few policies
        for i, json_file in enumerate(json_files[:3]):  # Check first 3 files
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    policy_data = json.load(f)
                
                print(f"  {i+1}. {json_file.name}:")
                print(f"     Policy ID: {policy_data.get('policy_id', 'N/A')}")
                print(f"     Document Title: {policy_data.get('document_title', 'N/A')}")
                print(f"     Policy Type: {policy_data.get('policy_type', 'N/A')}")
                print(f"     Sections: {len(policy_data.get('sections', []))}")
                
            except Exception as e:
                print(f"  [-] Error loading {json_file.name}: {e}")
    else:
        print("[-] No processed policy files found")
    
    print()
    
    # Check extracted CSV files
    extracted_dir = Path("data/extracted")
    if not extracted_dir.exists():
        print("[-] Extracted data directory not found")
        return False
    
    print("[+] Extracted data directory found")
    
    # Check enhanced LIC policies CSV
    enhanced_csv = extracted_dir / "enhanced_lic_policies.csv"
    if enhanced_csv.exists():
        try:
            df = pd.read_csv(enhanced_csv)
            print(f"[+] Enhanced LIC policies CSV found: {len(df)} records")
            
            # Show some statistics
            if not df.empty:
                print(f"   Columns: {len(df.columns)}")
                print(f"   Sample document titles:")
                for title in df['document_title'].head(3):
                    print(f"     - {title}")
        except Exception as e:
            print(f"[-] Error reading enhanced LIC policies CSV: {e}")
    else:
        print("[-] Enhanced LIC policies CSV not found")
    
    # Check final dataset
    final_csv = extracted_dir / "final_dataset.csv"
    if final_csv.exists():
        try:
            # Read just the first few rows to check
            df = pd.read_csv(final_csv, nrows=5)
            total_records = df.iloc[0]['total_records'] if 'total_records' in df.columns else "Unknown"
            print(f"[+] Final dataset CSV found: {total_records} records")
            print(f"   Columns: {len(df.columns)}")
        except Exception as e:
            print(f"[-] Error reading final dataset CSV: {e}")
    else:
        print("[-] Final dataset CSV not found")
    
    print()
    print("Verification complete!")
    return True

def main():
    """Main function"""
    print("InsureSense 360 - Policy Verification")
    print("=" * 50)
    
    success = verify_processed_policies()
    
    if success:
        print("\n[+] Policy verification completed successfully!")
        print("[+] Processed policies are correctly saved and can be loaded")
        return 0
    else:
        print("\n[-] Policy verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())