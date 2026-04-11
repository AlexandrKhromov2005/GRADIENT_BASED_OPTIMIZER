#ifndef LAUNCH_H
#define LAUNCH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "image_processing_custom.h"
#include "gbo.h"
#include "block_metrics.h"
#include "image_metrics.h"
#include <iostream>
#include "attacks.h"
#include <functional>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include "jpeg/quantization_tables.h"

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);

// Launch with known attack type for extraction (embedding always with NONE)
void launch_with_known_attack(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);

// Launch with known attack type for 1024x1024 images with voting
void launch_with_known_attack_1024(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);

#ifdef TORCH_AVAILABLE
void launch_with_classifier(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);

// Launch with quadrant classifier for 512x512 images
void launch_with_quadrant_classifier(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);

// Launch with attack type classifier (model_torchscript.pt) for 1024x1024 images
void launch_with_attack_classifier(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);
#endif

// 4-quadrant embedding with different objectives for dataset generation
void embed_quadrants_with_objectives(const std::string& image_path, const std::string& output_path);

// Dataset generation function
void generate_quadrant_dataset(const std::string& input_dir, const std::string& output_base_dir);

// Dataset generation for 4-attack classifier (1024x1024 images)
void generate_attack_dataset_1024(const std::string& input_dir, const std::string& output_base_dir);

#ifdef TORCH_AVAILABLE
// Dataset generation with 16 quadrants (1024x1024) and random watermarks
void generate_quadrant_dataset_1024(const std::string& input_dir, const std::string& output_base_dir);
#endif // TORCH_AVAILABLE

#endif // LAUNCH_H

