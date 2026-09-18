#include "embedding_schemes.h"
#include "scheme_json.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

// Global dynamic vector size
size_t CURRENT_VEC_SIZE = VEC_SIZE_DEFAULT;

EmbeddingSchemeManager& EmbeddingSchemeManager::getInstance() {
    static EmbeddingSchemeManager instance;
    return instance;
}

bool EmbeddingSchemeManager::loadSchemes(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open schemes file: " << filename << std::endl;
        return false;
    }
    // One byte over the limit is enough for the parser to reject an oversized file.
    std::string content(kSchemesJsonMaxBytes + 1, '\0');
    file.read(&content[0], static_cast<std::streamsize>(content.size()));
    content.resize(static_cast<size_t>(file.gcount()));

    std::string error;
    if (!loadSchemesFromString(content, &error)) {
        std::cerr << filename << ": " << error << std::endl;
        return false;
    }
    std::cout << "Loaded " << schemes.size() << " embedding schemes" << std::endl;
    return true;
}

bool EmbeddingSchemeManager::loadSchemesFromString(const std::string& json, std::string* error) {
    std::map<std::string, EmbeddingScheme> parsed;
    std::string message;
    if (!parseSchemesJson(json, parsed, message)) {
        if (error) *error = message;
        return false;
    }
    schemes = std::move(parsed);
    // Keep the active scheme if the new set still has it; its size may have changed.
    if (schemes.find(currentSchemeId) == schemes.end()) currentSchemeId = schemes.begin()->first;
    CURRENT_VEC_SIZE = schemes[currentSchemeId].getTotalVectorSize();
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
        std::cerr << "Warning: Scheme '" << schemeId << "' not found, keeping '" << currentSchemeId << "'" << std::endl;
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
        use_single_classifier_ = false;
        std::cout << "✅ Ensemble Classifier initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize ensemble classifier: " << e.what() << std::endl;
        classifier_initialized_ = false;
        return false;
    }
}

bool EmbeddingSchemeManager::initializeSingleClassifier(const std::string& model_path,
                                                       float threshold,
                                                       bool use_cuda) {
    try {
        single_classifier_ = std::make_unique<SingleClassifier>(model_path, threshold, use_cuda);
        single_classifier_initialized_ = single_classifier_->isLoaded();
        use_single_classifier_ = true;
        std::cout << "✅ Single Classifier initialized successfully" << std::endl;
        return single_classifier_initialized_;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize single classifier: " << e.what() << std::endl;
        single_classifier_initialized_ = false;
        return false;
    }
}

std::string EmbeddingSchemeManager::selectSchemeForEmbedding(const cv::Mat& block_8x8) {
    // Check if single classifier is available and preferred
    if (use_single_classifier_ && single_classifier_initialized_ && single_classifier_) {
        try {
            auto result = single_classifier_->predict(block_8x8, true); // use TTA
            
            // Map classifier result to embedding scheme
            std::string selected_scheme;
            if (result.predicted_class == 0) {
                selected_scheme = "scheme2";  // scheme_0 -> scheme2
            } else {
                selected_scheme = "scheme3";  // scheme_1 -> scheme3
            }
            
            std::cout << "🎯 Block classified as " << result.class_name 
                      << " (confidence: " << (result.confidence * 100) << "%) -> using " 
                      << selected_scheme << " [Single Classifier]" << std::endl;
            
            return selected_scheme;
        } catch (const std::exception& e) {
            std::cerr << "❌ Single classifier prediction failed: " << e.what() << std::endl;
        }
    }
    
    // Fallback to ensemble classifier
    if (classifier_initialized_ && classifier_) {
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
                      << selected_scheme << " [Ensemble Classifier]" << std::endl;
            
            return selected_scheme;
        } catch (const std::exception& e) {
            std::cerr << "❌ Ensemble classifier prediction failed: " << e.what() << std::endl;
        }
    }
    
    std::cout << "⚠️ No classifier initialized, using default scheme2" << std::endl;
    return "scheme2";
}

std::string EmbeddingSchemeManager::predictSchemeForExtraction(const cv::Mat& block_8x8) {
    // Check if single classifier is available and preferred
    if (use_single_classifier_ && single_classifier_initialized_ && single_classifier_) {
        try {
            auto result = single_classifier_->predict(block_8x8, true); // use TTA
            
            // Map classifier result to extraction scheme
            std::string predicted_scheme;
            if (result.predicted_class == 0) {
                predicted_scheme = "scheme2";  // scheme_0 -> scheme2
            } else {
                predicted_scheme = "scheme3";  // scheme_1 -> scheme3
            }
            
            std::cout << "🔍 Block predicted as " << result.class_name 
                      << " (confidence: " << (result.confidence * 100) << "%) -> extracting with " 
                      << predicted_scheme << " [Single Classifier]" << std::endl;
            
            return predicted_scheme;
        } catch (const std::exception& e) {
            std::cerr << "❌ Single classifier prediction failed: " << e.what() << std::endl;
        }
    }
    
    // Fallback to ensemble classifier
    if (classifier_initialized_ && classifier_) {
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
                      << predicted_scheme << " [Ensemble Classifier]" << std::endl;
            
            return predicted_scheme;
        } catch (const std::exception& e) {
            std::cerr << "❌ Ensemble classifier prediction failed: " << e.what() << std::endl;
        }
    }
    
    std::cout << "⚠️ No classifier initialized, trying scheme2 for extraction" << std::endl;
    return "scheme2";
}
#endif