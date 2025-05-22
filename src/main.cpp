#include <opencv2/opencv.hpp>
#include <iostream>
#include <immintrin.h>
using namespace cv;
using namespace std;

int main() {
    // Проверка поддержки SIMD
    cout << "Поддержка SSE: " << (cv::checkHardwareSupport(CV_CPU_SSE) ? "Да" : "Нет") << endl;
    cout << "Поддержка SSE2: " << (cv::checkHardwareSupport(CV_CPU_SSE2) ? "Да" : "Нет") << endl;
    cout << "Поддержка SSE3: " << (cv::checkHardwareSupport(CV_CPU_SSE3) ? "Да" : "Нет") << endl;
    cout << "Поддержка SSE4_1: " << (cv::checkHardwareSupport(CV_CPU_SSE4_1) ? "Да" : "Нет") << endl;
    cout << "Поддержка SSE4_2: " << (cv::checkHardwareSupport(CV_CPU_SSE4_2) ? "Да" : "Нет") << endl;
    cout << "Поддержка AVX: " << (cv::checkHardwareSupport(CV_CPU_AVX) ? "Да" : "Нет") << endl;

    // Проверка количества ядер
    cout << "Количество ядер процессора: " << cv::getNumberOfCPUs() << endl;

    return 0;
}