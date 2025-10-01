#include "GRADIENT_BASED_OPTIMIZER/src/dataset_generation.h"
#include <iostream>

int main() {
    std::cout << "🚀 Starting classifier experiment with final_model.pt" << std::endl;
    
    // Run experiment with tau_max = 10.0 (default parameter)
    generate_dataset_with_classifier(10.0);
    
    std::cout << "✅ Experiment completed!" << std::endl;
    return 0;
}