#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/launch.h"

int main()
{
    const std::string image = "images/lenna.png";
    const std::string cvz = "images/watermark.png";
    const std::string new_image = "images/new_lenna.png";
    const std::string extracted_cvz = "images/lenna_wm.png";

    launch(image, new_image, cvz, extracted_cvz);

}
