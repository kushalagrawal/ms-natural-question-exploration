#!/bin/bash

# Create base output directory
mkdir -p unzipped

# Find all .gz files and process them
find . -name "*.gz" | while read -r file; do
    # Get the relative path and filename
    relpath=$(dirname "$file")
    filename=$(basename "$file" .gz)
    
    # Create target directory
    mkdir -p "unzipped/$relpath"
    
    # Extract file
    echo "Extracting: $file -> unzipped/$relpath/$filename"
    gunzip -c "$file" > "unzipped/$relpath/$filename"
done
