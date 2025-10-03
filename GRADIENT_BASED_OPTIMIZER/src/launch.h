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
#include "jpeg/quantization_tables.h"

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);

#ifdef TORCH_AVAILABLE
void launch_with_classifier(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations = 10);
#endif

// 4-quadrant embedding with different objectives for dataset generation
void embed_quadrants_with_objectives(const std::string& image_path, const std::string& output_path);

// Dataset generation function
void generate_quadrant_dataset(const std::string& input_dir, const std::string& output_base_dir);

#endif // LAUNCH_H

