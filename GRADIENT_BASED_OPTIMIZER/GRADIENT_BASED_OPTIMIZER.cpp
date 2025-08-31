#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "src/launch.h"
#include "src/embedding_schemes.h"
#include "src/dataset_generation.h"

int main(int argc, char* argv[])
{
    bool test_mode = false;
    bool dataset_mode = false;
    double tau_max = 10.0;
    std::string scheme_id = "scheme1";
    
    // Initialize schemes manager
    auto& manager = EmbeddingSchemeManager::getInstance();
    manager.loadSchemes();
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") {
            test_mode = true;
            std::cout << "Test mode: 1 iteration per image" << std::endl;
        } else if (arg == "--dataset" || arg == "-d") {
            dataset_mode = true;
            std::cout << "Dataset generation mode" << std::endl;
        } else if (arg == "--tau-max") {
            if (i + 1 < argc) {
                tau_max = std::stod(argv[++i]);
            }
        } else if (arg == "--scheme" || arg == "-s") {
            if (i + 1 < argc) {
                scheme_id = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--test|-t] [--dataset|-d] [--tau-max VALUE] [--scheme|-s scheme_id] [--help|-h]" << std::endl;
            std::cout << "  --test, -t         Test mode (1 iteration per image)" << std::endl;
            std::cout << "  --dataset, -d      Dataset generation mode (scheme2 vs scheme3)" << std::endl;
            std::cout << "  --tau-max VALUE    Maximum error threshold for dataset (default: 10.0)" << std::endl;
            std::cout << "  --scheme, -s ID    Use embedding scheme (scheme1, scheme2, scheme3)" << std::endl;
            std::cout << "  --help, -h         Show this help" << std::endl;
            std::cout << "  default            Main mode (10 iterations per image)" << std::endl;
            
            auto schemes = manager.getAvailableSchemes();
            std::cout << "\nAvailable schemes:" << std::endl;
            for (const auto& id : schemes) {
                const auto* scheme = manager.getScheme(id);
                if (scheme) {
                    std::cout << "  " << id << " - " << scheme->name << std::endl;
                }
            }
            return 0;
        }
    }
    
    // Set the chosen scheme
    manager.setCurrentScheme(scheme_id);
    
    auto start = std::chrono::high_resolution_clock::now();

    if (dataset_mode) {
        std::cout << "Dataset generation mode: comparing scheme2 vs scheme3" << std::endl;
        generate_dataset(tau_max);
    } else {
        if (!test_mode) {
            std::cout << "Main mode: 10 iterations per image" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started" << std::endl;

            const std::string image = "images/" + name + ".png";
            const std::string cvz = "images/watermark.png";
            const std::string new_image = "images/new_" + name + ".png";
            const std::string extracted_cvz = "images/" + name + "_wm.png";

            int iterations = test_mode ? 1 : 10;
            launch(image, new_image, cvz, extracted_cvz, iterations);

            std::cout << name << " is finished" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;

    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);

    std::cout << "\nTotal execution time: ";
    bool need_space = false;

    if (hours.count() > 0) {
        std::cout << hours.count() << "h ";
        need_space = true;
    }
    if (minutes.count() > 0 || need_space) {
        std::cout << minutes.count() << "m ";
        need_space = true;
    }
    std::cout << seconds.count() << "s" << std::endl;

    return 0;
}
