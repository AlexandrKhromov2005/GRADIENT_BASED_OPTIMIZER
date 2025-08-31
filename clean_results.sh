#!/bin/bash
# Script to clean all result files

echo "Cleaning all result files..."

# Remove all results_*.txt files
rm -f results_*.txt

# Count how many files were removed
removed_count=$(find . -name "results_*.txt" -type f | wc -l)
echo "Cleaned result files."

# List remaining result files if any
remaining=$(find . -name "results_*.txt" -type f)
if [ -n "$remaining" ]; then
    echo "Warning: Some result files remain:"
    echo "$remaining"
else
    echo "All result files have been cleaned."
fi