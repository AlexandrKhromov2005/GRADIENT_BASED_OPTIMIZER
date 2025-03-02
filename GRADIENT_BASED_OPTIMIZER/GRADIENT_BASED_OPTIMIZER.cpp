#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/launch.h"

int main()
{
    std::vector<std::string> names = { "airplane", "baboon", "boat", "bridge", "earth_from_space", "lake", "lenna", "pepper" };

    for (std::string name : names) {
        std::cout << name << " is started" << std::endl;

        const std::string image = "images/" + name + ".png";
        const std::string cvz = "images/watermark.png";
        const std::string new_image = "images/new_" + name + ".png";
        const std::string extracted_cvz = "images/" + name + "_wm.png";

        launch(image, new_image, cvz, extracted_cvz);

        std::cout << name << " is finished" << std::endl;
    }

}
