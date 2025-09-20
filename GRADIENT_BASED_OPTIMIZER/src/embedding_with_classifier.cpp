#include "embedding_with_classifier.h"
#include "block_metrics.h"
#include <iostream>

bool EmbeddingWithClassifier::classifier_ready_ = false;

bool EmbeddingWithClassifier::initializeClassifier(const std::vector<std::string>& model_paths,
                                                   const std::vector<float>& thresholds,
                                                   bool use_cuda) {
    auto& manager = EmbeddingSchemeManager::getInstance();
    
    // Load embedding schemes first
    if (!manager.loadSchemes()) {
        std::cerr << "❌ Failed to load embedding schemes" << std::endl;
        return false;
    }
    
    // Initialize the classifier
    bool success = manager.initializeClassifier(model_paths, thresholds, use_cuda);
    classifier_ready_ = success;
    
    if (success) {
        std::cout << "🚀 EmbeddingWithClassifier initialized successfully" << std::endl;
    } else {
        std::cerr << "❌ Failed to initialize EmbeddingWithClassifier" << std::endl;
    }
    
    return success;
}

cv::Mat EmbeddingWithClassifier::embedBitWithSchemeSelection(const cv::Mat& block_8x8, uchar bit_to_embed) {
    if (block_8x8.rows != 8 || block_8x8.cols != 8) {
        std::cerr << "❌ Input block must be 8x8" << std::endl;
        return block_8x8.clone();
    }
    
    auto& manager = EmbeddingSchemeManager::getInstance();
    
    // Select scheme using classifier
    std::string selected_scheme = manager.selectSchemeForEmbedding(block_8x8);
    
    // Set the selected scheme
    std::string original_scheme = manager.getCurrentScheme() ? manager.getCurrentScheme()->name : "unknown";
    manager.setCurrentScheme(selected_scheme);
    
    std::cout << "🎯 Embedding bit " << (int)bit_to_embed 
              << " using scheme: " << selected_scheme << std::endl;
    
    // Create a copy of the block for embedding
    cv::Mat block_copy = block_8x8.clone();
    
    // Perform embedding using GBO
    try {
        GBO gbo(bit_to_embed, block_copy);
        gbo.main_loop();
        
        std::cout << "✅ Bit embedded successfully with " << selected_scheme << std::endl;
        return block_copy;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Embedding failed: " << e.what() << std::endl;
        return block_8x8.clone(); // return original block on failure
    }
}

uchar EmbeddingWithClassifier::extractBitWithSchemePrediction(const cv::Mat& block_8x8) {
    if (block_8x8.rows != 8 || block_8x8.cols != 8) {
        std::cerr << "❌ Input block must be 8x8" << std::endl;
        return 0;
    }
    
    auto& manager = EmbeddingSchemeManager::getInstance();
    
    // Predict scheme using classifier
    std::string predicted_scheme = manager.predictSchemeForExtraction(block_8x8);
    
    // Set the predicted scheme
    manager.setCurrentScheme(predicted_scheme);
    
    std::cout << "🔍 Extracting bit using predicted scheme: " << predicted_scheme << std::endl;
    
    // Perform extraction using the predicted scheme
    try {
        cv::Mat gray_block;
        if (block_8x8.channels() == 3) {
            cv::cvtColor(block_8x8, gray_block, cv::COLOR_BGR2GRAY);
        } else {
            gray_block = block_8x8.clone();
        }
        
        cv::Mat blockDouble;
        gray_block.convertTo(blockDouble, CV_64F);
        cv::Mat dct_block;
        cv::dct(blockDouble, dct_block);
        
        double s0 = calc_s_zero(dct_block);
        double s1 = calc_s_one(dct_block);
        
        uchar extracted_bit = (s0 < s1) ? 1 : 0;
        
        std::cout << "✅ Bit extracted: " << (int)extracted_bit 
                  << " (s0=" << s0 << ", s1=" << s1 << ")" << std::endl;
        
        return extracted_bit;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Extraction failed: " << e.what() << std::endl;
        return 0; // return 0 on failure
    }
}