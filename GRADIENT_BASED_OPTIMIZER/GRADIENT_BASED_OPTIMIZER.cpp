#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "src/launch.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge",
                                      "earth_from_space", "lake", "lenna", "pepper" };

    for (std::string name : names) {
        std::cout << name << " is started" << std::endl;

        const std::string image = "images/" + name + ".png";
        const std::string cvz = "images/watermark.png";
        const std::string new_image = "images/new_" + name + ".png";
        const std::string extracted_cvz = "images/" + name + "_wm.png";

        launch(image, new_image, cvz, extracted_cvz);

        std::cout << name << " is finished" << std::endl;
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
