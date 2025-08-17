#!/usr/bin/env python3
"""
Master Data Preparation Script
Runs all data preparation steps in sequence
"""

import subprocess
import sys
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"📁 Running: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, f"scripts/{script_name}"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"⚠️  Warnings/Errors: {result.stderr}")
        
        # Check success
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"✅ {description} completed successfully in {elapsed_time:.2f}s")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main function to run all data preparation steps"""
    print("Starting Complete Data Preparation Pipeline")
    print("This will run all data preparation steps in sequence")
    
    # Define the steps
    steps = [
        {
            'script': 'extract_policy_info.py',
            'description': 'Extracting Policy Information from PDFs'
        },
        {
            'script': 'clean_structured_data.py', 
            'description': 'Cleaning and Normalizing Structured Data'
        },
        {
            'script': 'merge_datasets.py',
            'description': 'Merging All Datasets into Final Dataset'
        }
    ]
    
    # Track results
    successful_steps = []
    failed_steps = []
    
    # Run each step
    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}/{len(steps)}: {step['description']}")
        
        success = run_script(step['script'], step['description'])
        
        if success:
            successful_steps.append(step['description'])
        else:
            failed_steps.append(step['description'])
            print(f"Continuing with next step...")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"📊 DATA PREPARATION PIPELINE COMPLETED")
    print(f"{'='*60}")
    
    print(f"Successful steps ({len(successful_steps)}/{len(steps)}):")
    for step in successful_steps:
        print(f"  - {step}")
    
    if failed_steps:
        print(f"\nFailed steps ({len(failed_steps)}/{len(steps)}):")
        for step in failed_steps:
            print(f"  - {step}")
    
    # Check final output
    final_dataset = Path("data/extracted/final_dataset.csv")
    if final_dataset.exists():
        print(f"\nSUCCESS: Final dataset created at {final_dataset}")
        print(f"You can now proceed to Phase 4: Local API Execution")
    else:
        print(f"\nWARNING: Final dataset not found. Some steps may have failed.")
    
    print(f"\nNext steps:")
    print(f"  1. Review the extracted data in data/extracted/")
    print(f"  2. Check dataset_summary.json for statistics")
    print(f"  3. Proceed to Phase 4: Local API Execution")

if __name__ == "__main__":
    main()
