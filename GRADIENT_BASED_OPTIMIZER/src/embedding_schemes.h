#ifndef EMBEDDING_SCHEMES_H
#define EMBEDDING_SCHEMES_H

#include <vector>
#include <string>
#include <array>
#include <map>
#include "config.h"

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

private:
    EmbeddingSchemeManager() = default;
    std::map<std::string, EmbeddingScheme> schemes;
    std::string currentSchemeId = "scheme1";
};

// Global functions to access current scheme data
const std::vector<std::pair<int, int>>& getCurrentREG0();
const std::vector<std::pair<int, int>>& getCurrentREG1();
const std::vector<std::pair<int, int>>& getCurrentZONE0();

#endif // EMBEDDING_SCHEMES_H