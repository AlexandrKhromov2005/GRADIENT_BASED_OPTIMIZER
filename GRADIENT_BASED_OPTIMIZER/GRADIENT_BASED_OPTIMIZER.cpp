#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "src/launch.h"
#include "src/embedding_schemes.h"
#include "src/dataset_generation.h"

#ifdef TORCH_AVAILABLE
#include "src/embedding_with_classifier.h"
#endif

int main(int argc, char* argv[])
{
    bool test_mode = false;
    bool dataset_mode = false;
    bool classifier_mode = false;
    bool dataset_classifier_mode = false;
    bool example_mode = false;
    bool quadrant_dataset_mode = false;
    bool quadrant_classifier_mode = false;
    bool attack_dataset_mode = false;
    bool attack_classifier_mode = false;
    bool known_attack_mode = false;
    bool known_attack_1024_mode = false;
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
        } else if (arg == "--classifier" || arg == "-c") {
            classifier_mode = true;
            std::cout << "Metrics evaluation with classifier integration" << std::endl;
        } else if (arg == "--dataset-classifier") {
            dataset_classifier_mode = true;
            std::cout << "Dataset generation with classifier integration mode" << std::endl;
        } else if (arg == "--example" || arg == "-e") {
            example_mode = true;
            std::cout << "Classifier integration example mode" << std::endl;
        } else if (arg == "--quadrant-dataset" || arg == "-q") {
            quadrant_dataset_mode = true;
            std::cout << "Quadrant dataset generation mode" << std::endl;
        } else if (arg == "--quadrant-classifier") {
            quadrant_classifier_mode = true;
            std::cout << "Quadrant classifier mode (best_model_ultrahighres.pt, 1024x1024)" << std::endl;
        } else if (arg == "--attack-dataset") {
            attack_dataset_mode = true;
            std::cout << "Attack dataset generation mode (NoAttack, JPEG70, JPEG80, ContrastIncrease)" << std::endl;
        } else if (arg == "--attack-classifier") {
            attack_classifier_mode = true;
            std::cout << "Attack classifier mode (model_torchscript.pt, 1024x1024)" << std::endl;
        } else if (arg == "--known-attack") {
            known_attack_mode = true;
            std::cout << "Known attack mode (embed with NONE, extract with known attack type)" << std::endl;
        } else if (arg == "--known-attack-1024") {
            known_attack_1024_mode = true;
            std::cout << "Known attack 1024x1024 mode (embed all quadrants with NONE, extract with voting)" << std::endl;
        } else if (arg == "--tau-max") {
            if (i + 1 < argc) {
                tau_max = std::stod(argv[++i]);
            }
        } else if (arg == "--scheme" || arg == "-s") {
            if (i + 1 < argc) {
                scheme_id = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --test, -t              Test mode (1 iteration per image)" << std::endl;
            std::cout << "  --dataset, -d           Dataset generation mode (scheme2 vs scheme3)" << std::endl;
            std::cout << "  --classifier, -c        Metrics evaluation WITH classifier integration" << std::endl;
            std::cout << "  --dataset-classifier    Full experiment: dataset generation WITH classifier" << std::endl;
            std::cout << "  --example, -e           Run classifier integration example" << std::endl;
            std::cout << "  --quadrant-dataset, -q  Generate quadrant-based dataset (1024x1024, 16 quadrants, random WM)" << std::endl;
            std::cout << "  --quadrant-classifier   Quadrant classifier experiment (best_model_ultrahighres.pt, 1024x1024 images)" << std::endl;
            std::cout << "  --attack-dataset        Generate dataset for 4 attacks (NoAttack, JPEG70, JPEG80, ContrastIncrease)" << std::endl;
            std::cout << "  --attack-classifier     Attack type classifier experiment (model_torchscript.pt, 1024x1024 images)" << std::endl;
            std::cout << "  --known-attack          Known attack mode (embed: NONE, extract: attack-aware)" << std::endl;
            std::cout << "  --known-attack-1024     Known attack 1024x1024 mode with voting among quadrants" << std::endl;
            std::cout << "  --tau-max VALUE         Maximum error threshold for dataset (default: 10.0)" << std::endl;
            std::cout << "  --scheme, -s ID         Use embedding scheme (scheme1, scheme2, scheme3)" << std::endl;
            std::cout << "  --help, -h              Show this help" << std::endl;
            std::cout << "  default                 Main mode (10 iterations per image)" << std::endl;
            
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

    if (quadrant_dataset_mode) {
#ifdef TORCH_AVAILABLE
        std::cout << "🎯 Quadrant dataset generation mode (1024x1024, 16 quadrants, random WM)" << std::endl;
        std::string input_dir = "images_1024";
        std::string output_dir = "dataset_quadrant_1024_extended";
        generate_quadrant_dataset_1024(input_dir, output_dir);
#else
        std::cerr << "❌ Quadrant dataset mode requires PyTorch installation" << std::endl;
        return 1;
#endif
    } else if (attack_dataset_mode) {
        std::cout << "🎯 Attack dataset generation mode (1024x1024)" << std::endl;
        std::string input_dir = "images_1024";
        std::string output_dir = "dataset_attack_1024";
        generate_attack_dataset_1024(input_dir, output_dir);
    } else if (dataset_mode) {
        std::cout << "Dataset generation mode: comparing scheme2 vs scheme3" << std::endl;
        generate_dataset(tau_max);
    } else if (dataset_classifier_mode) {
#ifdef TORCH_AVAILABLE
        std::cout << "🤖 Full experiment: Dataset generation WITH classifier integration (final_model.pt)" << std::endl;
        generate_dataset_with_classifier(tau_max);
#else
        std::cerr << "❌ Dataset classifier mode requires PyTorch installation" << std::endl;
        return -1;
#endif
    } else if (quadrant_classifier_mode) {
#ifdef TORCH_AVAILABLE
        std::cout << "🎯 Quadrant classifier experiment mode" << std::endl;

        if (!test_mode) {
            std::cout << "Quadrant classifier mode: 10 iterations per image (1024x1024, 16 quadrants)" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started (with quadrant classifier)" << std::endl;

            const std::string image = "test_images_1024/" + name + ".png";
            const std::string cvz = "images/watermark_32x32.png";
            const std::string new_image = "test_images_1024/new_" + name + ".png";
            const std::string extracted_cvz = "test_images_1024/" + name + "_wm.png";

            int iterations = test_mode ? 1 : 10;
            launch_with_quadrant_classifier(image, new_image, cvz, extracted_cvz, iterations);

            std::cout << name << " is finished (with quadrant classifier)" << std::endl;
        }
#else
        std::cerr << "❌ Quadrant classifier mode requires PyTorch (libtorch) installation" << std::endl;
        std::cerr << "    Please install PyTorch C++ and recompile" << std::endl;
        return -1;
#endif
    } else if (attack_classifier_mode) {
#ifdef TORCH_AVAILABLE
        std::cout << "🎯 Attack type classifier experiment mode (model_torchscript.pt)" << std::endl;

        if (!test_mode) {
            std::cout << "Attack classifier mode: 10 iterations per image (1024x1024, attack type prediction)" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started (with attack type classifier)" << std::endl;

            const std::string image = "test_images_1024/" + name + ".png";
            const std::string cvz = "test_images_1024/watermark.png";
            const std::string new_image = "test_images_1024/new_" + name + ".png";
            const std::string extracted_cvz = "test_images_1024/" + name + "_wm.png";

            int iterations = test_mode ? 1 : 10;
            launch_with_attack_classifier(image, new_image, cvz, extracted_cvz, iterations);

            std::cout << name << " is finished (with attack type classifier)" << std::endl;
        }
#else
        std::cerr << "❌ Attack classifier mode requires PyTorch (libtorch) installation" << std::endl;
        std::cerr << "    Please install PyTorch C++ and recompile" << std::endl;
        return -1;
#endif
    } else if (classifier_mode) {
#ifdef TORCH_AVAILABLE
        std::cout << "🤖 Metrics evaluation WITH classifier integration" << std::endl;

        if (!test_mode) {
            std::cout << "Classifier mode: 10 iterations per image" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started (with classifier)" << std::endl;

            const std::string image = "images/" + name + ".png";
            const std::string cvz = "images/watermark_32x32.png";
            const std::string new_image = "images/new_" + name + ".png";
            const std::string extracted_cvz = "images/" + name + "_wm.png";

            int iterations = test_mode ? 1 : 10;
            launch_with_classifier(image, new_image, cvz, extracted_cvz, iterations);

            std::cout << name << " is finished (with classifier)" << std::endl;
        }
#else
        std::cerr << "❌ Classifier mode requires PyTorch (libtorch) installation" << std::endl;
        std::cerr << "    Please install PyTorch C++ and recompile" << std::endl;
        return -1;
#endif
    } else if (example_mode) {
#ifdef TORCH_AVAILABLE
        std::cout << "🧪 Running classifier integration example..." << std::endl;
        
        // Initialize single classifier with final_model.pt
        std::string model_path = "final_model_torchscript.pt";
        float threshold = 0.5f;
        
        if (EmbeddingWithClassifier::initializeSingleClassifier(model_path, threshold, true)) {
            // Test on a single image
            std::string test_image_path = "images/lenna.png";
            cv::Mat test_image = cv::imread(test_image_path, cv::IMREAD_GRAYSCALE);
            
            if (!test_image.empty()) {
                std::cout << "📸 Testing with: " << test_image_path << std::endl;
                
                // Take first 8x8 block
                if (test_image.rows >= 8 && test_image.cols >= 8) {
                    cv::Rect block_rect(0, 0, 8, 8);
                    cv::Mat test_block = test_image(block_rect);
                    
                    // Test embedding and extraction
                    for (uchar bit = 0; bit <= 1; bit++) {
                        std::cout << "\n--- Testing bit: " << (int)bit << " ---" << std::endl;
                        
                        cv::Mat embedded = EmbeddingWithClassifier::embedBitWithSchemeSelection(
                            test_block, bit);
                        uchar extracted = EmbeddingWithClassifier::extractBitWithSchemePrediction(
                            embedded);
                        
                        std::cout << (bit == extracted ? "✅" : "❌") 
                                  << " Original: " << (int)bit 
                                  << ", Extracted: " << (int)extracted << std::endl;
                    }
                } else {
                    std::cerr << "❌ Test image too small" << std::endl;
                }
            } else {
                std::cerr << "❌ Could not load test image: " << test_image_path << std::endl;
            }
        } else {
            std::cerr << "❌ Failed to initialize classifier for example" << std::endl;
        }
#else
        std::cerr << "❌ Example mode requires PyTorch (libtorch) installation" << std::endl;
        std::cerr << "    Please install PyTorch C++ and recompile" << std::endl;
        return -1;
#endif
    } else if (known_attack_mode) {
        if (!test_mode) {
            std::cout << "Known Attack mode: 10 iterations per image" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started (known attack mode)" << std::endl;

            const std::string image = "images/" + name + ".png";
            const std::string cvz = "images/watermark_32x32.png";
            const std::string new_image = "images/new_" + name + ".png";
            const std::string extracted_cvz = "images/" + name + "_wm.png";

            int iterations = test_mode ? 1 : 10;
            launch_with_known_attack(image, new_image, cvz, extracted_cvz, iterations);

            std::cout << name << " is finished" << std::endl;
        }
    } else if (known_attack_1024_mode) {
        if (!test_mode) {
            std::cout << "Known Attack 1024x1024 mode: 10 iterations per image" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started (known attack 1024x1024 mode)" << std::endl;

            const std::string image = "test_images_1024/" + name + ".png";
            const std::string cvz = "test_images_1024/watermark.png";
            const std::string new_image = "test_images_1024/new_" + name + ".png";
            const std::string extracted_cvz = "test_images_1024/" + name + "_wm.png";

            int iterations = test_mode ? 1 : 10;
            launch_with_known_attack_1024(image, new_image, cvz, extracted_cvz, iterations);

            std::cout << name << " is finished" << std::endl;
        }
    } else {
        if (!test_mode) {
            std::cout << "Main mode: 10 iterations per image" << std::endl;
        }

        std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                          "earth_from_space", "lake", "lenna", "pepper" };

        for (std::string name : names) {
            std::cout << name << " is started" << std::endl;

            const std::string image = "images/" + name + ".png";
            const std::string cvz = "images/watermark_32x32.png";
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
