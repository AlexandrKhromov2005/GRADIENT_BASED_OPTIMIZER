#ifndef EMBEDDING_SCHEMES_H
#define EMBEDDING_SCHEMES_H

#include <vector>
#include <string>
#include <array>
#include <map>
#include "config.h"

#ifdef TORCH_AVAILABLE
#include "ensemble_classifier.h"
#include "single_classifier.h"
#endif

struct EmbeddingScheme {
    std::string name;
    std::string description;
    std::vector<std::pair<int, int>> REG0;
    std::vector<std::pair<int, int>> REG1;
    std::vector<std::pair<int, int>> ZONE0;
    
    // Get sizes for dynamic vector allocation
    size_t getREG0Size() const { return REG0.size(); }
    size_t getREG1Size() const { return REG1.size(); }
    size_t getZONE0Size() const { return ZONE0.size(); }
    size_t getTotalVectorSize() const { return ZONE0.size(); }
};

class EmbeddingSchemeManager {
public:
    static EmbeddingSchemeManager& getInstance();
    bool loadSchemes(const std::string& filename = "embedding_schemes.json");
    const EmbeddingScheme* getScheme(const std::string& schemeId) const;
    std::vector<std::string> getAvailableSchemes() const;
    void setCurrentScheme(const std::string& schemeId);
    const EmbeddingScheme* getCurrentScheme() const;
    
#ifdef TORCH_AVAILABLE
    // Classifier integration methods (for ensemble)
    bool initializeClassifier(const std::vector<std::string>& model_paths, 
                             const std::vector<float>& thresholds, 
                             bool use_cuda = true);
    
    // Single classifier integration methods
    bool initializeSingleClassifier(const std::string& model_path,
                                   float threshold = 0.5f,
                                   bool use_cuda = true);
    
    std::string selectSchemeForEmbedding(const cv::Mat& block_8x8);
    std::string predictSchemeForExtraction(const cv::Mat& block_8x8);
#endif

private:
    EmbeddingSchemeManager() = default;
    std::map<std::string, EmbeddingScheme> schemes;
    std::string currentSchemeId = "scheme1";
    
#ifdef TORCH_AVAILABLE
    std::unique_ptr<EnsembleClassifier> classifier_;
    std::unique_ptr<SingleClassifier> single_classifier_;
    bool classifier_initialized_ = false;
    bool single_classifier_initialized_ = false;
    bool use_single_classifier_ = false;  // flag to choose between ensemble and single
#endif
};

// Global functions to access current scheme data
const std::vector<std::pair<int, int>>& getCurrentREG0();
const std::vector<std::pair<int, int>>& getCurrentREG1();
const std::vector<std::pair<int, int>>& getCurrentZONE0();

#endif // EMBEDDING_SCHEMES_H