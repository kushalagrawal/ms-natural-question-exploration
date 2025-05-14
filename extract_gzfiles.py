#!/usr/bin/env python3

"""
Script to recursively extract .gz files while preserving directory structure.
Creates a mirrored directory structure under 'unzipped' directory and extracts all .gz files.
"""

import os
import subprocess
from pathlib import Path
import sys
import shutil

def extract_gz_file(gz_file_path, output_file_path):
    """Extract a single .gz file to the specified output path using gunzip command."""
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary copy of the .gz file in the target directory
        temp_gz = output_file_path.with_suffix('.gz')
        shutil.copy2(gz_file_path, temp_gz)
        
        # Use gunzip command to extract
        result = subprocess.run(['gunzip', '-f', str(temp_gz)], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            print(f"Error extracting {gz_file_path}: {result.stderr}")
            # Clean up temporary file if it exists
            if temp_gz.exists():
                temp_gz.unlink()
            return False
            
        return True
    except Exception as e:
        print(f"Error processing {gz_file_path}: {str(e)}")
        # Clean up temporary file if it exists
        if temp_gz.exists():
            temp_gz.unlink()
        return False

def process_directory(base_dir='.', output_base_dir='unzipped'):
    """Recursively process directory to find and extract .gz files."""
    base_path = Path(base_dir).resolve()
    output_base_path = Path(output_base_dir).resolve()
    output_base_path.mkdir(exist_ok=True)
    
    total_files = 0
    successful_extractions = 0
    
    gz_files = list(base_path.glob('**/*.gz'))
    total_gz_files = len(gz_files)
    
    if total_gz_files == 0:
        print(f"No .gz files found in {base_path}")
        return 0, 0
    
    print(f"Found {total_gz_files} .gz files to extract")
    
    for idx, gz_file_path in enumerate(gz_files, 1):
        relative_path = gz_file_path.relative_to(base_path)
        output_file_path = output_base_path / relative_path.with_suffix('')
        
        print(f"[{idx}/{total_gz_files}] Extracting: {relative_path}")
        print(f"    -> {output_file_path.name}")
        
        total_files += 1
        if extract_gz_file(gz_file_path, output_file_path):
            successful_extractions += 1
    
    return total_files, successful_extractions

def main():
    """Main function to handle command-line arguments and execute extraction."""
    base_dir = '.'
    output_dir = 'unzipped'
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"Starting extraction process from '{base_dir}' to '{output_dir}'")
    
    total, successful = process_directory(base_dir, output_dir)
    
    print(f"\nExtraction complete!")
    print(f"Files processed: {total}")
    print(f"Files successfully extracted: {successful}")
    
    if total > 0:
        success_rate = (successful / total) * 100
        print(f"Success rate: {success_rate:.2f}%")
    
    if total > 0 and successful == total:
        print("All files extracted successfully!")
        return 0
    elif successful > 0:
        print("Some files were extracted, but there were errors. Check the output for details.")
        return 1
    else:
        print("No files were extracted successfully.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
