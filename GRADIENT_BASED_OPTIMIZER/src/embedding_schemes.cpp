#include "embedding_schemes.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

// Global dynamic vector size
size_t CURRENT_VEC_SIZE = VEC_SIZE_DEFAULT;

// JSON parsing - simple implementation for this specific case
#include <sstream>
#include <algorithm>

EmbeddingSchemeManager& EmbeddingSchemeManager::getInstance() {
    static EmbeddingSchemeManager instance;
    return instance;
}

std::vector<std::pair<int, int>> parseCoordinateArray(const std::string& arrayStr) {
    std::vector<std::pair<int, int>> coords;
    std::istringstream stream(arrayStr);
    std::string token;
    
    while (std::getline(stream, token, ']')) {
        size_t bracketPos = token.find('[');
        if (bracketPos != std::string::npos) {
            std::string pairStr = token.substr(bracketPos + 1);
            size_t commaPos = pairStr.find(',');
            if (commaPos != std::string::npos) {
                int x = std::stoi(pairStr.substr(0, commaPos));
                int y = std::stoi(pairStr.substr(commaPos + 1));
                coords.emplace_back(x, y);
            }
        }
    }
    
    return coords;
}

bool EmbeddingSchemeManager::loadSchemes(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open schemes file: " << filename << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing for our specific structure
    // Parse scheme1
    EmbeddingScheme scheme1;
    scheme1.name = "Original Scheme";
    scheme1.description = "Original embedding scheme";
    
    // scheme1 REG0
    scheme1.REG0 = {
        {7, 1}, {6, 1}, {5, 1}, {5, 3}, {4, 3}, {3, 3},
        {3, 5}, {2, 5}, {1, 5}, {1, 7}, {0, 7}
    };
    
    // scheme1 REG1
    scheme1.REG1 = {
        {7, 0}, {6, 0}, {6, 2}, {5, 2}, {4, 2},
        {4, 4}, {3, 4}, {2, 4}, {2, 6}, {1, 6}, {0, 6}
    };
    
    // scheme1 ZONE0
    scheme1.ZONE0 = {
        {6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7},
        {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1},
        {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7}
    };
    
    schemes["scheme1"] = scheme1;

    // Parse scheme2
    EmbeddingScheme scheme2;
    scheme2.name = "Alternative Scheme";
    scheme2.description = "Second embedding scheme with new coefficients";
    
    // scheme2 REG0
    scheme2.REG0 = {
        {7, 0}, {6, 0}, {7, 1}, {6, 1}, {6, 2}, 
        {5, 1}, {5, 3}, {4, 4}, {3, 4}, {3, 5}, {2, 5}
    };
    
    // scheme2 REG1
    scheme2.REG1 = {
        {5, 2}, {4, 2}, {4, 3}, {3, 3}, {2, 4}, 
        {2, 6}, {1, 5}, {1, 6}, {1, 7}, {0, 6}, {0, 7}
    };
    
    // scheme2 ZONE0 (combined REG0 + REG1)
    scheme2.ZONE0 = {
        {7, 0}, {6, 0}, {7, 1}, {6, 1}, {6, 2}, {5, 1}, {5, 3}, {4, 4}, {3, 4}, {3, 5}, {2, 5},
        {5, 2}, {4, 2}, {4, 3}, {3, 3}, {2, 4}, {2, 6}, {1, 5}, {1, 6}, {1, 7}, {0, 6}, {0, 7}
    };
    
    schemes["scheme2"] = scheme2;

    // Parse scheme3 with variable sizes (12 REG0 + 13 REG1 = 25 ZONE0)
    EmbeddingScheme scheme3;
    scheme3.name = "Variable Size Scheme";
    scheme3.description = "Scheme with 12 REG0 elements and 13 REG1 elements";
    
    // scheme3 REG0 (12 elements)
    scheme3.REG0 = {
        {2, 2},
        {7, 1}, {6, 1}, {5, 1},
        {5, 3}, {4, 3}, {3, 3},
        {3, 5}, {2, 5}, {1, 5},
        {1, 7}, {0, 7}
    };
    
    // scheme3 REG1 (13 elements)
    scheme3.REG1 = {
        {1, 3}, {3, 1},
        {7, 0}, {6, 0},
        {6, 2}, {5, 2}, {4, 2},
        {4, 4}, {3, 4}, {2, 4},
        {2, 6}, {1, 6}, {0, 6}
    };
    
    // scheme3 ZONE0 (25 elements total - REG0 + REG1)
    scheme3.ZONE0 = {
        {2, 2}, {1, 3}, {3, 1},
        
        {6, 0}, {5, 1}, {4, 2}, {3, 3},
        {2, 4}, {1, 5}, {0, 6}, {0, 7},
        {1, 6}, {2, 5}, {3, 4}, {4, 3},
        {5, 2}, {6, 1}, {7, 0}, {7, 1},
        {6, 2}, {5, 3}, {4, 4}, {3, 5},
        {2, 6}, {1, 7}
    };
    
    schemes["scheme3"] = scheme3;

    std::cout << "Loaded " << schemes.size() << " embedding schemes" << std::endl;
    return true;
}

const EmbeddingScheme* EmbeddingSchemeManager::getScheme(const std::string& schemeId) const {
    auto it = schemes.find(schemeId);
    return (it != schemes.end()) ? &it->second : nullptr;
}

std::vector<std::string> EmbeddingSchemeManager::getAvailableSchemes() const {
    std::vector<std::string> result;
    for (const auto& pair : schemes) {
        result.push_back(pair.first);
    }
    return result;
}

void EmbeddingSchemeManager::setCurrentScheme(const std::string& schemeId) {
    if (schemes.find(schemeId) != schemes.end()) {
        currentSchemeId = schemeId;
        CURRENT_VEC_SIZE = schemes[schemeId].getTotalVectorSize();
        std::cout << "Using embedding scheme: " << schemes[schemeId].name 
                  << " (vector size: " << CURRENT_VEC_SIZE << ")" << std::endl;
    } else {
        std::cerr << "Warning: Scheme '" << schemeId << "' not found, using default" << std::endl;
        CURRENT_VEC_SIZE = VEC_SIZE_DEFAULT;
    }
}

const EmbeddingScheme* EmbeddingSchemeManager::getCurrentScheme() const {
    return getScheme(currentSchemeId);
}

// Global accessor functions
const std::vector<std::pair<int, int>>& getCurrentREG0() {
    const EmbeddingScheme* scheme = EmbeddingSchemeManager::getInstance().getCurrentScheme();
    if (scheme) {
        return scheme->REG0;
    }
    static std::vector<std::pair<int, int>> empty;
    return empty;
}

const std::vector<std::pair<int, int>>& getCurrentREG1() {
    const EmbeddingScheme* scheme = EmbeddingSchemeManager::getInstance().getCurrentScheme();
    if (scheme) {
        return scheme->REG1;
    }
    static std::vector<std::pair<int, int>> empty;
    return empty;
}

const std::vector<std::pair<int, int>>& getCurrentZONE0() {
    const EmbeddingScheme* scheme = EmbeddingSchemeManager::getInstance().getCurrentScheme();
    if (scheme) {
        return scheme->ZONE0;
    }
    static std::vector<std::pair<int, int>> empty;
    return empty;
}

#ifdef TORCH_AVAILABLE
// Classifier integration methods
bool EmbeddingSchemeManager::initializeClassifier(const std::vector<std::string>& model_paths,
                                                  const std::vector<float>& thresholds,
                                                  bool use_cuda) {
    try {
        classifier_ = std::make_unique<EnsembleClassifier>(model_paths, thresholds, use_cuda);
        classifier_initialized_ = true;
        std::cout << "✅ Classifier initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize classifier: " << e.what() << std::endl;
        classifier_initialized_ = false;
        return false;
    }
}

std::string EmbeddingSchemeManager::selectSchemeForEmbedding(const cv::Mat& block_8x8) {
    if (!classifier_initialized_ || !classifier_) {
        std::cout << "⚠️ Classifier not initialized, using default scheme2" << std::endl;
        return "scheme2";
    }
    
    try {
        auto result = classifier_->predict(block_8x8, true); // use TTA
        
        // Map classifier result to embedding scheme
        std::string selected_scheme;
        if (result.predicted_class == 0) {
            selected_scheme = "scheme2";  // scheme_0 -> scheme2
        } else {
            selected_scheme = "scheme3";  // scheme_1 -> scheme3
        }
        
        std::cout << "🎯 Block classified as " << result.class_name 
                  << " (confidence: " << (result.confidence * 100) << "%) -> using " 
                  << selected_scheme << std::endl;
        
        return selected_scheme;
    } catch (const std::exception& e) {
        std::cerr << "❌ Classifier prediction failed: " << e.what() << std::endl;
        return "scheme2"; // fallback
    }
}

std::string EmbeddingSchemeManager::predictSchemeForExtraction(const cv::Mat& block_8x8) {
    if (!classifier_initialized_ || !classifier_) {
        std::cout << "⚠️ Classifier not initialized, trying scheme2 for extraction" << std::endl;
        return "scheme2";
    }
    
    try {
        auto result = classifier_->predict(block_8x8, true); // use TTA
        
        // Map classifier result to extraction scheme
        std::string predicted_scheme;
        if (result.predicted_class == 0) {
            predicted_scheme = "scheme2";  // scheme_0 -> scheme2
        } else {
            predicted_scheme = "scheme3";  // scheme_1 -> scheme3
        }
        
        std::cout << "🔍 Block predicted as " << result.class_name 
                  << " (confidence: " << (result.confidence * 100) << "%) -> extracting with " 
                  << predicted_scheme << std::endl;
        
        return predicted_scheme;
    } catch (const std::exception& e) {
        std::cerr << "❌ Classifier prediction failed: " << e.what() << std::endl;
        return "scheme2"; // fallback
    }
}
#endif