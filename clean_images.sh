#!/bin/bash
# Script to remove generated images and keep only original source images

echo "Cleaning generated images, keeping only original source images..."

cd images/

# Original source images that should be kept
original_images=(
    "aerial.png"
    "affine.png"
    "airplane.png"
    "apc.png"
    "baboon.png"
    "big_bird.png"
    "boat.png"
    "bridge.png"
    "butterfly.png"
    "car_and_apcs.png"
    "corrall.png"
    "cross_road.png"
    "dve_chaika.png"
    "earth_from_space.png"
    "fish.png"
    "lake.png"
    "lenna.png"
    "meow.png"
    "mokey.png"
    "mountain.png"
    "old_cycle.png"
    "old_pair.png"
    "owl.png"
    "pepper.png"
    "single_tower_twin.png"
    "splash.png"
    "stream_and_bridge.png"
    "tank.png"
    "truck_and_apcs.png"
    "watermark.png"
    "wild_west_building.png"
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