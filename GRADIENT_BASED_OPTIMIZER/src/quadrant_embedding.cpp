#include "quadrant_embedding.h"
#include "image_processing_custom.h"
#include "gbo.h"
#include "population.h"
#include "config.h"
#include "jpeg/quantization_tables.h"
#include <iostream>
#include <random>

std::shared_ptr<QuadrantClassifier> QuadrantEmbedding::classifier_ = nullptr;
bool QuadrantEmbedding::classifier_ready_ = false;

bool QuadrantEmbedding::initializeQuadrantClassifier(const std::string& model_path, bool use_cuda) {
    std::cout << "🚀 Initializing QuadrantEmbedding with model: " << model_path << std::endl;

    try {
        classifier_ = std::make_shared<QuadrantClassifier>(model_path, use_cuda);
        classifier_ready_ = classifier_->isLoaded();

        if (classifier_ready_) {
            std::cout << "✅ QuadrantEmbedding initialized successfully" << std::endl;
        } else {
            std::cerr << "❌ Failed to initialize QuadrantEmbedding" << std::endl;
        }

        return classifier_ready_;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception during QuadrantEmbedding initialization: " << e.what() << std::endl;
        classifier_ready_ = false;
        return false;
    }
}

cv::Mat QuadrantEmbedding::embedWatermarkQuadrants(const cv::Mat& image_1024x1024,
                                                   const cv::Mat& watermark) {
    // Verify image is 1024x1024
    if (image_1024x1024.rows != 1024 || image_1024x1024.cols != 1024) {
        std::cerr << "❌ Error: Image must be 1024x1024, got " << image_1024x1024.rows
                  << "x" << image_1024x1024.cols << std::endl;
        return image_1024x1024.clone();
    }

    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image_1024x1024.channels() == 3) {
        cv::cvtColor(image_1024x1024, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image_1024x1024.clone();
    }

    // Initialize quantization tables for JPEG attack simulation
    initialize_quantization_mats();

    // Convert watermark to binary
    std::vector<int> wm_vec = convertWatermarkToBinary(watermark);
    if (wm_vec.size() != WM_SIZE) {
        std::cerr << "❌ Error: Watermark size mismatch. Expected " << WM_SIZE
                  << ", got " << wm_vec.size() << std::endl;
        return gray_image.clone();
    }

    // Pattern: 1-2-1-2 / 3-4-3-4 / 1-2-1-2 / 3-4-3-4
    // Type 1: NONE, Type 2: JPEG70, Type 3: CONTRAST, Type 4: JPEG80
    AttackType pattern[4][4] = {
        {AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
        {AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80},
        {AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
        {AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80}
    };

    std::cout << "📍 Embedding watermark into 16 quadrants (1024x1024 -> 16x256x256)..." << std::endl;

    // Create result image
    cv::Mat result(1024, 1024, gray_image.type());

    // Process all 16 quadrants
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            int x = col * 256;
            int y = row * 256;

            // Extract quadrant
            cv::Mat quadrant = gray_image(cv::Rect(x, y, 256, 256)).clone();

            // Get attack type for this quadrant
            AttackType attack = pattern[row][col];

            // Embed watermark
            std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
            for (size_t i = 0; i < blocks.size(); ++i) {
                GBO gbo(wm_vec[i % WM_SIZE], blocks[i], attack);
                gbo.main_loop();
            }
            cv::Mat embedded_quadrant = merge8x8Blocks(blocks, 256, 256);

            // Copy back to result
            embedded_quadrant.copyTo(result(cv::Rect(x, y, 256, 256)));
        }
    }

    std::cout << "✅ Watermark embedded into all 16 quadrants" << std::endl;
    return result;
}

// Overload: Embed using bit vector (for dataset generation)
cv::Mat QuadrantEmbedding::embedWatermarkQuadrants(const cv::Mat& image_1024x1024,
                                                   const std::vector<int>& wm_bits) {
    // Verify image is 1024x1024
    if (image_1024x1024.rows != 1024 || image_1024x1024.cols != 1024) {
        std::cerr << "❌ Error: Image must be 1024x1024, got " << image_1024x1024.rows
                  << "x" << image_1024x1024.cols << std::endl;
        return image_1024x1024.clone();
    }

    // Verify watermark size
    if (wm_bits.size() != WM_SIZE) {
        std::cerr << "❌ Error: Watermark bit vector size mismatch. Expected " << WM_SIZE
                  << ", got " << wm_bits.size() << std::endl;
        return image_1024x1024.clone();
    }

    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image_1024x1024.channels() == 3) {
        cv::cvtColor(image_1024x1024, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image_1024x1024.clone();
    }

    // Initialize quantization tables for JPEG attack simulation
    initialize_quantization_mats();

    // Pattern: 1-2-1-2 / 3-4-3-4 / 1-2-1-2 / 3-4-3-4
    // Type 1: NONE, Type 2: JPEG70, Type 3: CONTRAST, Type 4: JPEG80
    AttackType pattern[4][4] = {
        {AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
        {AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80},
        {AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
        {AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80}
    };

    std::cout << "📍 Embedding random watermark into 16 quadrants (1024x1024 -> 16x256x256)..." << std::endl;

    // Create result image
    cv::Mat result(1024, 1024, gray_image.type());

    // Process all 16 quadrants
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            int x = col * 256;
            int y = row * 256;

            // Extract quadrant
            cv::Mat quadrant = gray_image(cv::Rect(x, y, 256, 256)).clone();

            // Get attack type for this quadrant
            AttackType attack = pattern[row][col];

            // Embed watermark using the bit vector
            std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
            for (size_t i = 0; i < blocks.size(); ++i) {
                GBO gbo(wm_bits[i % WM_SIZE], blocks[i], attack);
                gbo.main_loop();
            }
            cv::Mat embedded_quadrant = merge8x8Blocks(blocks, 256, 256);

            // Copy back to result
            embedded_quadrant.copyTo(result(cv::Rect(x, y, 256, 256)));
        }
    }

    std::cout << "✅ Random watermark embedded into all 16 quadrants" << std::endl;
    return result;
}

cv::Mat QuadrantEmbedding::extractWatermarkWithClassifier(const cv::Mat& image_1024x1024) {
    if (!classifier_ready_ || !classifier_) {
        std::cerr << "❌ Error: Classifier not initialized!" << std::endl;
        return cv::Mat();
    }

    // Verify image is 1024x1024
    if (image_1024x1024.rows != 1024 || image_1024x1024.cols != 1024) {
        std::cerr << "❌ Error: Image must be 1024x1024, got " << image_1024x1024.rows
                  << "x" << image_1024x1024.cols << std::endl;
        return cv::Mat();
    }

    // Classifier expects RGB/BGR image (it will convert internally)
    // Use classifier to predict which attack type was applied
    auto prediction = classifier_->predict(image_1024x1024, true);  // use TTA

    // Now convert to grayscale for watermark extraction
    cv::Mat gray_image;
    if (image_1024x1024.channels() == 3) {
        cv::cvtColor(image_1024x1024, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image_1024x1024.clone();
    }

    std::cout << "🎯 Classifier prediction: " << prediction.class_name
              << " (confidence: " << (prediction.confidence * 100) << "%)" << std::endl;
    std::cout << "   Probabilities: scheme_0=" << prediction.prob_scheme0
              << ", scheme_1=" << prediction.prob_scheme1
              << ", scheme_2=" << prediction.prob_scheme2
              << ", scheme_3=" << prediction.prob_scheme3 << std::endl;

    // Quadrant positions for each attack type (pattern 1-2-1-2 / 3-4-3-4 / 1-2-1-2 / 3-4-3-4)
    // Type 1 (NONE): positions [0,0], [0,2], [2,0], [2,2]
    // Type 2 (JPEG70): positions [0,1], [0,3], [2,1], [2,3]
    // Type 3 (CONTRAST): positions [1,0], [1,2], [3,0], [3,2]
    // Type 4 (JPEG80): positions [1,1], [1,3], [3,1], [3,3]

    std::vector<std::pair<int, int>> quadrant_positions;
    std::string attack_name;

    switch (prediction.predicted_class) {
        case 0:  // scheme_0 -> Type 1 (NONE attack)
            quadrant_positions = {{0,0}, {0,2}, {2,0}, {2,2}};
            attack_name = "NONE";
            break;
        case 1:  // scheme_1 -> Type 2 (JPEG70 attack)
            quadrant_positions = {{0,1}, {0,3}, {2,1}, {2,3}};
            attack_name = "JPEG70";
            break;
        case 2:  // scheme_2 -> Type 3 (CONTRAST attack)
            quadrant_positions = {{1,0}, {1,2}, {3,0}, {3,2}};
            attack_name = "CONTRAST";
            break;
        case 3:  // scheme_3 -> Type 4 (JPEG80 attack)
            quadrant_positions = {{1,1}, {1,3}, {3,1}, {3,3}};
            attack_name = "JPEG80";
            break;
        default:
            std::cerr << "❌ Invalid prediction class: " << prediction.predicted_class << std::endl;
            return cv::Mat();
    }

    std::cout << "📤 Extracting from quadrants optimized for " << attack_name << " attack" << std::endl;

    // Extract watermark from all 4 quadrants of this type and use voting
    std::vector<std::vector<int>> all_extractions;

    for (const auto& pos : quadrant_positions) {
        int row = pos.first;
        int col = pos.second;
        int x = col * 256;
        int y = row * 256;

        // Extract this quadrant
        cv::Mat quadrant = gray_image(cv::Rect(x, y, 256, 256)).clone();
        std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
        std::vector<int> wm_vec(WM_SIZE);

        for (size_t i = 0; i < WM_SIZE && i < blocks.size(); ++i) {
            cv::Mat block = blocks[i];

            // Convert to double and apply DCT
            cv::Mat blockDouble;
            block.convertTo(blockDouble, CV_64F);
            cv::Mat dct_block;
            cv::dct(blockDouble, dct_block);

            // Extract bit using S0 and S1 comparison
            double s0 = calc_s_zero(dct_block);
            double s1 = calc_s_one(dct_block);

            wm_vec[i] = (s0 < s1) ? 1 : 0;
        }

        all_extractions.push_back(wm_vec);
    }

    // Voting: for each bit position, use majority vote from 4 quadrants
    // If votes are equal (2:2), choose randomly
    std::vector<int> final_wm_vec(WM_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (size_t i = 0; i < WM_SIZE; ++i) {
        int count_ones = 0;
        for (const auto& extraction : all_extractions) {
            if (extraction[i] == 1) count_ones++;
        }

        // Majority vote: if tie (2:2), choose randomly
        if (count_ones > 2) {
            final_wm_vec[i] = 1;  // 3 or 4 votes for 1
        } else if (count_ones < 2) {
            final_wm_vec[i] = 0;  // 0 or 1 votes for 1
        } else {
            // Tie: 2 votes for 1, 2 votes for 0 -> random choice
            final_wm_vec[i] = dis(gen);
        }
    }

    // Convert binary to watermark image
    cv::Mat extracted_wm = convertBinaryToWatermark(final_wm_vec);
    std::cout << "✅ Watermark extracted successfully (with voting from 4 quadrants)" << std::endl;

    return extracted_wm;
}

std::shared_ptr<QuadrantClassifier> QuadrantEmbedding::getClassifier() {
    return classifier_;
}
