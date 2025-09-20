#include "embedding_with_classifier.h"
#include "image_processing_custom.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "🚀 Testing Classifier Integration with Embedding System" << std::endl;
    
    try {
        // Step 1: Initialize classifier
        std::vector<std::string> model_paths = {
            "best_scheme_classifier_torchscript.pt",
            "ensemble_model_1_torchscript.pt"
        };
        std::vector<float> thresholds = {0.510f, 0.510f};
        
        std::cout << "📦 Initializing classifier..." << std::endl;
        if (!EmbeddingWithClassifier::initializeClassifier(model_paths, thresholds, true)) {
            std::cerr << "❌ Failed to initialize classifier. Exiting." << std::endl;
            return -1;
        }
        
        // Step 2: Load test image
        std::string test_image_path = "images/lenna.png";  // Adjust path as needed
        cv::Mat test_image = readImage(test_image_path);
        
        if (test_image.empty()) {
            std::cerr << "❌ Could not load test image: " << test_image_path << std::endl;
            return -1;
        }
        
        std::cout << "📸 Loaded test image: " << test_image_path 
                  << " (" << test_image.rows << "x" << test_image.cols << ")" << std::endl;
        
        // Step 3: Split image into 8x8 blocks
        auto blocks = splitInto8x8Blocks(test_image);
        std::cout << "🔪 Split into " << blocks.size() << " blocks of 8x8" << std::endl;
        
        // Step 4: Test embedding and extraction with classifier
        std::vector<cv::Mat> embedded_blocks;
        std::vector<uchar> original_bits;
        std::vector<uchar> extracted_bits;
        
        int test_blocks_count = std::min(10, (int)blocks.size()); // Test first 10 blocks
        std::cout << "🧪 Testing " << test_blocks_count << " blocks..." << std::endl;
        
        for (int i = 0; i < test_blocks_count; i++) {
            std::cout << "\n--- Block " << i << " ---" << std::endl;
            
            // Generate random bit to embed
            uchar bit_to_embed = rand() % 2;
            original_bits.push_back(bit_to_embed);
            
            auto start_embed = std::chrono::high_resolution_clock::now();
            
            // Embed bit with automatic scheme selection
            cv::Mat embedded_block = EmbeddingWithClassifier::embedBitWithSchemeSelection(
                blocks[i], bit_to_embed);
            
            auto end_embed = std::chrono::high_resolution_clock::now();
            auto embed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_embed - start_embed).count();
            
            embedded_blocks.push_back(embedded_block);
            
            auto start_extract = std::chrono::high_resolution_clock::now();
            
            // Extract bit with automatic scheme prediction
            uchar extracted_bit = EmbeddingWithClassifier::extractBitWithSchemePrediction(
                embedded_block);
            
            auto end_extract = std::chrono::high_resolution_clock::now();
            auto extract_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_extract - start_extract).count();
            
            extracted_bits.push_back(extracted_bit);
            
            // Check correctness
            bool correct = (bit_to_embed == extracted_bit);
            std::cout << (correct ? "✅" : "❌") << " Block " << i 
                      << ": embedded=" << (int)bit_to_embed 
                      << ", extracted=" << (int)extracted_bit
                      << " (embed: " << embed_time << "ms, extract: " << extract_time << "ms)" << std::endl;
        }
        
        // Step 5: Calculate accuracy
        int correct_count = 0;
        for (int i = 0; i < test_blocks_count; i++) {
            if (original_bits[i] == extracted_bits[i]) {
                correct_count++;
            }
        }
        
        double accuracy = (double)correct_count / test_blocks_count * 100.0;
        std::cout << "\n📊 Results Summary:" << std::endl;
        std::cout << "   Correct: " << correct_count << "/" << test_blocks_count << std::endl;
        std::cout << "   Accuracy: " << accuracy << "%" << std::endl;
        
        // Step 6: Save result (optional)
        if (!embedded_blocks.empty()) {
            // Replace first blocks with embedded ones
            for (int i = 0; i < test_blocks_count; i++) {
                blocks[i] = embedded_blocks[i];
            }
            
            cv::Mat result_image = merge8x8Blocks(blocks, test_image.rows, test_image.cols);
            
            std::string output_path = "result_with_classifier.png";
            if (writeImage(output_path, result_image)) {
                std::cout << "💾 Result saved to: " << output_path << std::endl;
            }
        }
        
        std::cout << "\n🎉 Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception occurred: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}