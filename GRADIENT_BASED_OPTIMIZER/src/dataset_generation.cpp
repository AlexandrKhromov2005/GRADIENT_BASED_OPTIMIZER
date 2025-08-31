#include "dataset_generation.h"
#include "embedding_schemes.h"
#include "config.h"
#include "random_utils.h"
#include "population.h"
#include "gbo.h"
#include "block_metrics.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>

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
    std::filesystem::create_directories("dataset");
    std::filesystem::create_directories("dataset/Dir1");
    std::filesystem::create_directories("dataset/Dir2"); 
    std::filesystem::create_directories("dataset/Dir_rand");
    
    std::vector<std::string> image_names = {"airplane", "baboon", "boat", "bridge", 
                                           "earth_from_space", "lake", "lenna", "pepper"};
    
    std::cout << "Starting dataset generation with tau_max = " << tau_max << std::endl;
    
    for (const std::string& name : image_names) {
        std::cout << "Processing " << name << "..." << std::endl;
        
        std::string image_path = "images/" + name + ".png";
        std::string watermark_path = "images/watermark.png";
        
        cv::Mat original_color = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat original_image;
        cv::cvtColor(original_color, original_image, cv::COLOR_BGR2GRAY);
        cv::Mat watermark = cv::imread(watermark_path, cv::IMREAD_GRAYSCALE);
        
        if (original_image.empty() || watermark.empty()) {
            std::cout << "Error: Could not load " << name << " or watermark" << std::endl;
            continue;
        }
        
        // Process image in 8x8 blocks
        int block_count = 0;
        for (int y = 0; y <= original_image.rows - BLOCK_SIZE; y += BLOCK_SIZE) {
            for (int x = 0; x <= original_image.cols - BLOCK_SIZE; x += BLOCK_SIZE) {
                cv::Rect block_rect(x, y, BLOCK_SIZE, BLOCK_SIZE);
                cv::Mat block = original_image(block_rect);
                
                int quality = rand_int_1_to_100();
                int total_error_scheme2 = 0;
                int total_error_scheme3 = 0;
                
                // Test both bits (0 and 1) with both schemes and all attack types
                for (uchar test_bit = 0; test_bit <= 1; test_bit++) {
                    // Test with scheme2 (original)
                    manager.setCurrentScheme("scheme2");
                    
                    cv::Mat block_copy2 = block.clone();
                    GBO gbo_scheme2(test_bit, block_copy2);
                    gbo_scheme2.main_loop();
                    cv::Mat embedded_scheme2 = block_copy2.clone();
                    
                    // Apply attacks to scheme2 result
                    cv::Mat scheme2_jpeg = jpeg_attack(embedded_scheme2, 70);
                    cv::Mat scheme2_contrast = contrast_increase_attack(embedded_scheme2, 1.2);
                    
                    // Test extraction for scheme2
                    if (extract_bit_from_block(embedded_scheme2) != test_bit) total_error_scheme2++;
                    if (extract_bit_from_block(scheme2_jpeg) != test_bit) total_error_scheme2++;
                    if (extract_bit_from_block(scheme2_contrast) != test_bit) total_error_scheme2++;
                    
                    // Test with scheme3 (experimental)
                    manager.setCurrentScheme("scheme3");
                    
                    cv::Mat block_copy3 = block.clone();
                    GBO gbo_scheme3(test_bit, block_copy3);
                    gbo_scheme3.main_loop();
                    cv::Mat embedded_scheme3 = block_copy3.clone();
                    
                    // Apply attacks to scheme3 result
                    cv::Mat scheme3_jpeg = jpeg_attack(embedded_scheme3, 70);
                    cv::Mat scheme3_contrast = contrast_increase_attack(embedded_scheme3, 1.2);
                    
                    // Test extraction for scheme3
                    if (extract_bit_from_block(embedded_scheme3) != test_bit) total_error_scheme3++;
                    if (extract_bit_from_block(scheme3_jpeg) != test_bit) total_error_scheme3++;
                    if (extract_bit_from_block(scheme3_contrast) != test_bit) total_error_scheme3++;
                }
                
                // Determine classification based on total errors
                std::string output_dir;
                
                // Classification logic: Dir1 if tau2 < tau_max, Dir2 if tau3 <= tau_max <= tau2, else Dir_rand
                if (total_error_scheme2 < (int)tau_max) {
                    output_dir = "dataset/Dir1";  // scheme2 good (tau2 < tau_max)
                } else if (total_error_scheme3 <= (int)tau_max && (int)tau_max <= total_error_scheme2) {
                    output_dir = "dataset/Dir2";  // scheme3 acceptable and scheme2 worse (tau3 <= tau_max <= tau2)
                } else {
                    output_dir = "dataset/Dir_rand";  // other cases
                }
                
                // Save the block to appropriate directory
                std::string filename = name + "_block_" + std::to_string(block_count) + ".png";
                std::string output_path = output_dir + "/" + filename;
                
                cv::imwrite(output_path, block);
                
                // Log the results
                std::cout << "Block " << block_count << ": " 
                         << "scheme2_total(" << total_error_scheme2 << ") "
                         << "scheme3_total(" << total_error_scheme3 << ") "
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
    int count_dir1 = std::distance(std::filesystem::directory_iterator("dataset/Dir1"), 
                                  std::filesystem::directory_iterator{});
    int count_dir2 = std::distance(std::filesystem::directory_iterator("dataset/Dir2"), 
                                  std::filesystem::directory_iterator{});
    int count_dir_rand = std::distance(std::filesystem::directory_iterator("dataset/Dir_rand"), 
                                      std::filesystem::directory_iterator{});
    
    int total_blocks = count_dir1 + count_dir2 + count_dir_rand;
    int expected_total = 8 * 4096; // 8 images × 4096 blocks each
    
    std::cout << "Results:" << std::endl;
    std::cout << "Dir1 (scheme2 better): " << count_dir1 << " blocks" << std::endl;
    std::cout << "Dir2 (scheme3 better): " << count_dir2 << " blocks" << std::endl;
    std::cout << "Dir_rand (equivalent): " << count_dir_rand << " blocks" << std::endl;
    std::cout << "Total: " << total_blocks << " blocks (expected: " << expected_total << ")" << std::endl;
    
    if (total_blocks != expected_total) {
        std::cout << "WARNING: Block count mismatch! Difference: " << (total_blocks - expected_total) << std::endl;
    }
}