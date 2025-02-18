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
#include <filesystem>
#include <algorithm>

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm);

#endif // LAUNCH_H

