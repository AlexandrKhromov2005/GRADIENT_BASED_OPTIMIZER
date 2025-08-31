#!/bin/bash
# Script to remove generated images and keep only original source images

echo "Cleaning generated images, keeping only original source images..."

cd images/

# Original source images that should be kept
original_images=(
    "airplane.png"
    "baboon.png" 
    "boat.png"
    "bridge.png"
    "earth_from_space.png"
    "lake.png"
    "lenna.png"
    "pepper.png"
    "watermark.png"
)

# Count files before cleaning
total_before=$(find . -name "*.png" -type f | wc -l)

# Remove all PNG files except the original ones
for file in *.png; do
    if [[ ! " ${original_images[@]} " =~ " ${file} " ]]; then
        rm -f "$file"
        echo "Removed: $file"
    fi
done

# Count files after cleaning  
total_after=$(find . -name "*.png" -type f | wc -l)
removed_count=$((total_before - total_after))

echo ""
echo "Image cleanup completed:"
echo "  Files before: $total_before"
echo "  Files after: $total_after" 
echo "  Files removed: $removed_count"
echo ""
echo "Remaining original images:"
ls -la *.png 2>/dev/null || echo "No PNG files found"