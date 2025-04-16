#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "src/launch.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    const std::string image = "images/airplane.png";  
    const std::string cvz = "images/watermark.png";
    const std::string new_image = "images/new_airplane.png";  
    const std::string extracted_cvz = "images/airplane_wm.png";  

    std::cout << "Processing image: " << image << std::endl;
    launch(image, new_image, cvz, extracted_cvz);
    std::cout << "Processing completed." << std::endl;

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