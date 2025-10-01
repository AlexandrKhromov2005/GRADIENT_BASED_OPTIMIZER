#!/bin/bash

echo "🚀 Running quick experiment with final_model_torchscript.pt classifier"
echo "📊 Processing only first 5 images for faster results..."

# Backup original images directory
if [ ! -d "images_backup_temp" ]; then
    cp -r images images_backup_temp
fi

# Create temporary images directory with just 5 images for quick test
mkdir -p images_quick
cp images/aerial.png images_quick/
cp images/baboon.png images_quick/
cp images/boat.png images_quick/
cp images/bridge.png images_quick/
cp images/lenna.png images_quick/
cp images/watermark.png images_quick/

# Replace images directory
mv images images_full
mv images_quick images

echo "🎯 Starting classifier experiment with 5 images..."
timeout 600 ./build/gradient_based_optimizer --dataset-classifier

# Restore original images
mv images images_quick  
mv images_full images

echo "✅ Quick experiment completed!"
echo "📂 Results should be in dataset_classifier/ directory"

# Show summary if results exist
if [ -d "dataset_classifier" ]; then
    echo ""
    echo "📊 Quick Results Summary:"
    echo "Scheme2 selected: $(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l) blocks"
    echo "Scheme3 selected: $(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l) blocks"  
    echo "Extraction correct: $(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l) blocks"
    echo "Extraction incorrect: $(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l) blocks"
    
    correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
    incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)
    total=$((correct + incorrect))
    
    if [ $total -gt 0 ]; then
        accuracy=$(echo "scale=2; $correct * 100 / $total" | bc -l)
        echo "🎯 Overall accuracy: $accuracy%"
    fi
fi