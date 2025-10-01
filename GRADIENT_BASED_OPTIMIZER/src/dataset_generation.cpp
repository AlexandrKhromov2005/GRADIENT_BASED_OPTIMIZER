#include "dataset_generation.h"
#include "embedding_schemes.h"
#include "config.h"
#include <dirent.h>

#ifdef TORCH_AVAILABLE
#include "embedding_with_classifier.h"
#endif
#include "random_utils.h"
#include "population.h"
#include "gbo.h"
#include "block_metrics.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#define BLOCK_SIZE 8

// Function to extract bit from a block
uchar extract_bit_from_block(const cv::Mat& block) {
    cv::Mat gray_block;
    if (block.channels() == 3) {
        cv::cvtColor(block, gray_block, cv::COLOR_BGR2GRAY);
    } else {
        gray_block = block.clone();
    }
    
    cv::Mat blockDouble;
    gray_block.convertTo(blockDouble, CV_64F);
    cv::Mat dct_block;
    cv::dct(blockDouble, dct_block);
    
    double s0 = calc_s_zero(dct_block);
    double s1 = calc_s_one(dct_block);
    
    return (s0 < s1) ? 1 : 0;
}

cv::Mat jpeg_attack(const cv::Mat& image, int quality) {
    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, quality};
    std::vector<uchar> buffer;
    
    cv::imencode(".jpg", image, buffer, compression_params);
    cv::Mat attacked_image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    
    return attacked_image;
}

cv::Mat contrast_increase_attack(const cv::Mat& image, double alpha) {
    cv::Mat attacked_image;
    image.convertTo(attacked_image, -1, alpha, 0);
    return attacked_image;
}

void generate_dataset(double tau_max) {
    auto& manager = EmbeddingSchemeManager::getInstance();
    
    // Create output directories
    system("mkdir -p dataset");
    system("mkdir -p dataset/Dir1");
    system("mkdir -p dataset/Dir2");
    system("mkdir -p dataset/Dir_rand");
    
    // Automatically find all PNG images in the images directory (excluding watermark)
    std::vector<std::string> image_names;
    std::string images_dir = "images";
    
    DIR* dir = opendir(images_dir.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.find(".png") != std::string::npos && filename != "watermark.png") {
                // Remove .png extension
                std::string stem = filename.substr(0, filename.find_last_of('.'));
                image_names.push_back(stem);
            }
        }
        closedir(dir);
    }
    
    std::sort(image_names.begin(), image_names.end());
    std::cout << "Found " << image_names.size() << " images to process" << std::endl;
    
    std::cout << "Starting dataset generation with tau_max = " << tau_max << std::endl;
    
    for (const std::string& name : image_names) {
        std::cout << "Processing " << name << "..." << std::endl;
        
        std::string image_path = "images/" + name + ".png";
        std::string watermark_path = "images/watermark.png";
        
        cv::Mat original_color = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat original_image;
        
        if (original_color.empty()) {
            std::cout << "Error: Could not load " << name << std::endl;
            continue;
        }
        
        cv::cvtColor(original_color, original_image, cv::COLOR_BGR2GRAY);
        cv::Mat watermark = cv::imread(watermark_path, cv::IMREAD_GRAYSCALE);
        
        if (original_image.empty() || watermark.empty()) {
            std::cout << "Error: Could not process " << name << " or watermark" << std::endl;
            continue;
        }
        
        // Process image in 8x8 blocks
        int block_count = 0;
        for (int y = 0; y <= original_image.rows - BLOCK_SIZE; y += BLOCK_SIZE) {
            for (int x = 0; x <= original_image.cols - BLOCK_SIZE; x += BLOCK_SIZE) {
                cv::Rect block_rect(x, y, BLOCK_SIZE, BLOCK_SIZE);
                cv::Mat block = original_image(block_rect);
                
                int quality = rand_int_1_to_100();
                int total_error_standard = 0;
                int total_error_extended = 0;
                
                // Test both bits (0 and 1) with both schemes and all attack types
                for (uchar test_bit = 0; test_bit <= 1; test_bit++) {
                    // Test with standard_scheme (22 elements)
                    manager.setCurrentScheme("standard_scheme");
                    
                    cv::Mat block_copy_std = block.clone();
                    GBO gbo_standard(test_bit, block_copy_std);
                    gbo_standard.main_loop();
                    cv::Mat embedded_standard = block_copy_std.clone();
                    
                    // Apply attacks to standard_scheme result
                    cv::Mat standard_jpeg = jpeg_attack(embedded_standard, 70);
                    cv::Mat standard_contrast = contrast_increase_attack(embedded_standard, 1.2);
                    
                    // Test extraction for standard_scheme
                    if (extract_bit_from_block(embedded_standard) != test_bit) total_error_standard++;
                    if (extract_bit_from_block(standard_jpeg) != test_bit) total_error_standard++;
                    if (extract_bit_from_block(standard_contrast) != test_bit) total_error_standard++;
                    
                    // Test with extended_scheme (25 elements)
                    manager.setCurrentScheme("extended_scheme");
                    
                    cv::Mat block_copy_ext = block.clone();
                    GBO gbo_extended(test_bit, block_copy_ext);
                    gbo_extended.main_loop();
                    cv::Mat embedded_extended = block_copy_ext.clone();
                    
                    // Apply attacks to extended_scheme result
                    cv::Mat extended_jpeg = jpeg_attack(embedded_extended, 70);
                    cv::Mat extended_contrast = contrast_increase_attack(embedded_extended, 1.2);
                    
                    // Test extraction for extended_scheme
                    if (extract_bit_from_block(embedded_extended) != test_bit) total_error_extended++;
                    if (extract_bit_from_block(extended_jpeg) != test_bit) total_error_extended++;
                    if (extract_bit_from_block(extended_contrast) != test_bit) total_error_extended++;
                }
                
                // Determine classification based on total errors
                std::string output_dir;
                
                // Classification logic: Dir1 if standard < tau_max, Dir2 if extended <= tau_max <= standard, else Dir_rand
                if (total_error_standard < (int)tau_max) {
                    output_dir = "dataset/Dir1";  // standard_scheme good (standard < tau_max)
                } else if (total_error_extended <= (int)tau_max && (int)tau_max <= total_error_standard) {
                    output_dir = "dataset/Dir2";  // extended_scheme acceptable and standard worse (extended <= tau_max <= standard)
                } else {
                    output_dir = "dataset/Dir_rand";  // other cases
                }
                
                // Save the block to appropriate directory
                std::string filename = name + "_block_" + std::to_string(block_count) + ".png";
                std::string output_path = output_dir + "/" + filename;
                
                cv::imwrite(output_path, block);
                
                // Log the results
                std::cout << "Block " << block_count << ": " 
                         << "standard_total(" << total_error_standard << ") "
                         << "extended_total(" << total_error_extended << ") "
                         << "-> " << output_dir << std::endl;
                
                block_count++;
            }
        }
        
        int expected_blocks = (original_image.rows / BLOCK_SIZE) * (original_image.cols / BLOCK_SIZE);
        std::cout << "Processed " << block_count << " blocks for " << name 
                  << " (expected: " << expected_blocks << ", image size: " 
                  << original_image.rows << "x" << original_image.cols << ")" << std::endl;
    }
    
    std::cout << "Dataset generation completed!" << std::endl;
    
    // Count files in each directory
    FILE* fp;
    int count_dir1 = 0, count_dir2 = 0, count_dir_rand = 0;
    
    fp = popen("ls dataset/Dir1/*.png 2>/dev/null | wc -l", "r");
    if (fp) { fscanf(fp, "%d", &count_dir1); pclose(fp); }
    
    fp = popen("ls dataset/Dir2/*.png 2>/dev/null | wc -l", "r");
    if (fp) { fscanf(fp, "%d", &count_dir2); pclose(fp); }
    
    fp = popen("ls dataset/Dir_rand/*.png 2>/dev/null | wc -l", "r");
    if (fp) { fscanf(fp, "%d", &count_dir_rand); pclose(fp); }
    
    int total_blocks = count_dir1 + count_dir2 + count_dir_rand;
    int expected_total = image_names.size() * 4096; // number of images × 4096 blocks each
    
    std::cout << "Results:" << std::endl;
    std::cout << "Dir1 (standard_scheme better): " << count_dir1 << " blocks" << std::endl;
    std::cout << "Dir2 (extended_scheme better): " << count_dir2 << " blocks" << std::endl;
    std::cout << "Dir_rand (equivalent): " << count_dir_rand << " blocks" << std::endl;
    std::cout << "Total: " << total_blocks << " blocks (expected: " << expected_total << ")" << std::endl;
    
    if (total_blocks != expected_total) {
        std::cout << "WARNING: Block count mismatch! Difference: " << (total_blocks - expected_total) << std::endl;
    }
}

#ifdef TORCH_AVAILABLE
// Initialize classifier for dataset generation
bool initialize_classifier_for_dataset() {
    std::string model_path = "final_model_torchscript.pt";
    float threshold = 0.5f;
    
    std::cout << "🤖 Initializing single classifier for dataset generation..." << std::endl;
    
    if (EmbeddingWithClassifier::initializeSingleClassifier(model_path, threshold, true)) {
        std::cout << "✅ Classifier initialized successfully for dataset generation" << std::endl;
        return true;
    } else {
        std::cerr << "❌ Failed to initialize classifier for dataset generation" << std::endl;
        return false;
    }
}

// New dataset generation function with classifier integration
void generate_dataset_with_classifier(double tau_max) {
    std::cout << "🚀 Starting dataset generation WITH classifier integration" << std::endl;
    
    // Initialize classifier
    if (!initialize_classifier_for_dataset()) {
        std::cerr << "❌ Cannot proceed without classifier. Falling back to original method." << std::endl;
        generate_dataset(tau_max);
        return;
    }
    
    // Create output directories
    system("mkdir -p dataset_classifier");
    system("mkdir -p dataset_classifier/scheme2_selected");
    system("mkdir -p dataset_classifier/scheme3_selected");
    system("mkdir -p dataset_classifier/extraction_correct");
    system("mkdir -p dataset_classifier/extraction_incorrect");
    
    // Automatically find all PNG images in the images directory (excluding watermark)
    std::vector<std::string> image_names;
    std::string images_dir = "images";
    
    DIR* dir = opendir(images_dir.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.find(".png") != std::string::npos && filename != "watermark.png") {
                // Remove .png extension
                std::string stem = filename.substr(0, filename.find_last_of('.'));
                image_names.push_back(stem);
            }
        }
        closedir(dir);
    }
    
    std::sort(image_names.begin(), image_names.end());
    std::cout << "Found " << image_names.size() << " images to process" << std::endl;
    
    int total_blocks = 0;
    int scheme2_selected = 0;
    int scheme3_selected = 0;
    int extraction_correct = 0;
    int extraction_incorrect = 0;
    
    for (const std::string& name : image_names) {
        std::cout << "🖼️ Processing " << name << "..." << std::endl;
        
        std::string image_path = "images/" + name + ".png";
        cv::Mat original_color = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat original_image;
        
        if (original_color.empty()) {
            std::cout << "❌ Error: Could not load " << name << std::endl;
            continue;
        }
        
        cv::cvtColor(original_color, original_image, cv::COLOR_BGR2GRAY);
        
        if (original_image.empty()) {
            std::cout << "❌ Error: Could not process " << name << std::endl;
            continue;
        }
        
        // Process image in 8x8 blocks
        int block_count = 0;
        for (int y = 0; y <= original_image.rows - BLOCK_SIZE; y += BLOCK_SIZE) {
            for (int x = 0; x <= original_image.cols - BLOCK_SIZE; x += BLOCK_SIZE) {
                cv::Rect block_rect(x, y, BLOCK_SIZE, BLOCK_SIZE);
                cv::Mat block = original_image(block_rect);
                
                // Test with random bit
                uchar test_bit = rand() % 2;
                
                // Embed using classifier-selected scheme
                cv::Mat embedded_block = EmbeddingWithClassifier::embedBitWithSchemeSelection(
                    block, test_bit);
                
                // Extract using classifier-predicted scheme
                uchar extracted_bit = EmbeddingWithClassifier::extractBitWithSchemePrediction(
                    embedded_block);
                
                // Determine which scheme was selected for embedding
                auto& manager = EmbeddingSchemeManager::getInstance();
                std::string current_scheme_name = manager.getCurrentScheme() ? 
                    manager.getCurrentScheme()->name : "unknown";
                
                // Save block based on selected scheme
                std::string scheme_filename = name + "_block_" + std::to_string(block_count) + ".png";
                std::string scheme_path;
                
                if (current_scheme_name.find("scheme2") != std::string::npos || 
                    current_scheme_name.find("Original") != std::string::npos) {
                    scheme_path = "dataset_classifier/scheme2_selected/" + scheme_filename;
                    scheme2_selected++;
                } else {
                    scheme_path = "dataset_classifier/scheme3_selected/" + scheme_filename;
                    scheme3_selected++;
                }
                
                cv::imwrite(scheme_path, block);
                
                // Save block based on extraction correctness
                std::string correctness_filename = name + "_embedded_block_" + std::to_string(block_count) + ".png";
                std::string correctness_path;
                
                if (test_bit == extracted_bit) {
                    correctness_path = "dataset_classifier/extraction_correct/" + correctness_filename;
                    extraction_correct++;
                } else {
                    correctness_path = "dataset_classifier/extraction_incorrect/" + correctness_filename;
                    extraction_incorrect++;
                }
                
                cv::imwrite(correctness_path, embedded_block);
                
                // Log progress every 100 blocks
                if (block_count % 100 == 0) {
                    std::cout << "⚡ Processed " << block_count << " blocks for " << name << std::endl;
                }
                
                block_count++;
                total_blocks++;
            }
        }
        
        std::cout << "✅ Completed " << name << " with " << block_count << " blocks" << std::endl;
    }
    
    // Print statistics
    std::cout << "\n📊 Dataset Generation Results:" << std::endl;
    std::cout << "Total blocks processed: " << total_blocks << std::endl;
    std::cout << "Scheme2 selected: " << scheme2_selected << " (" << (double)scheme2_selected/total_blocks*100 << "%)" << std::endl;
    std::cout << "Scheme3 selected: " << scheme3_selected << " (" << (double)scheme3_selected/total_blocks*100 << "%)" << std::endl;
    std::cout << "Extraction correct: " << extraction_correct << " (" << (double)extraction_correct/total_blocks*100 << "%)" << std::endl;
    std::cout << "Extraction incorrect: " << extraction_incorrect << " (" << (double)extraction_incorrect/total_blocks*100 << "%)" << std::endl;
    
    double accuracy = (double)extraction_correct / total_blocks * 100.0;
    std::cout << "\n🎯 Overall extraction accuracy: " << accuracy << "%" << std::endl;
    
    if (accuracy > 90.0) {
        std::cout << "🎉 Excellent performance!" << std::endl;
    } else if (accuracy > 70.0) {
        std::cout << "👍 Good performance" << std::endl;
    } else {
        std::cout << "⚠️ Performance needs improvement" << std::endl;
    }
}
#endif