#include <iostream>
#include <string>
#include "src/gbo_api.h"
#include <opencv2/opencv.hpp>

static void printUsage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " embed  <cover> <watermark> <output> [--scheme ID]\n"
              << "  " << prog << " extract <watermarked> <output_wm>\n"
              << "  " << prog << " metrics <original> <watermarked> [--wm-orig WM1 --wm-extr WM2]\n"
              << "  " << prog << " attack  <image> <output> --type TYPE [--param VALUE]\n"
              << "  " << prog << " schemes\n"
              << "  " << prog << " --help\n"
              << "\nCommands:\n"
              << "  embed    Embed a binary watermark into a grayscale image.\n"
              << "  extract  Extract a watermark from a watermarked image.\n"
              << "  metrics  Compute image quality metrics (MSE, PSNR, SSIM, NCC).\n"
              << "  attack   Simulate an attack on an image.\n"
              << "  schemes  List available embedding schemes.\n"
              << "\nAttack types: jpeg, brightness+, brightness-, contrast+, contrast-,\n"
              << "              salt-pepper, median, gaussian\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") { printUsage(argv[0]); return 0; }

    if (!gbo::init()) {
        std::cerr << "Error: could not load embedding_schemes.json\n";
        return 1;
    }

    if (cmd == "schemes") {
        for (const auto& s : gbo::availableSchemes())
            std::cout << "  " << s << "\n";
        return 0;
    }

    if (cmd == "embed") {
        if (argc < 5) { std::cerr << "embed requires: <cover> <watermark> <output>\n"; return 1; }
        std::string cover_path = argv[2];
        std::string wm_path    = argv[3];
        std::string out_path   = argv[4];
        for (int i = 5; i < argc; ++i) {
            if ((std::string(argv[i]) == "--scheme" || std::string(argv[i]) == "-s") && i+1 < argc)
                gbo::setScheme(argv[++i]);
        }
        cv::Mat cover = cv::imread(cover_path, cv::IMREAD_GRAYSCALE);
        cv::Mat wm    = cv::imread(wm_path, cv::IMREAD_GRAYSCALE);
        if (cover.empty()) { std::cerr << "Cannot read cover image: " << cover_path << "\n"; return 1; }
        if (wm.empty())    { std::cerr << "Cannot read watermark: " << wm_path << "\n"; return 1; }

        cv::Mat result = gbo::embedWatermark(cover, wm);
        cv::imwrite(out_path, result);
        std::cout << "Watermarked image saved to " << out_path << "\n";
        return 0;
    }

    if (cmd == "extract") {
        if (argc < 4) { std::cerr << "extract requires: <watermarked> <output_wm>\n"; return 1; }
        cv::Mat img = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        if (img.empty()) { std::cerr << "Cannot read image: " << argv[2] << "\n"; return 1; }

        cv::Mat wm = gbo::extractWatermark(img);
        cv::imwrite(argv[3], wm);
        std::cout << "Extracted watermark saved to " << argv[3] << "\n";
        return 0;
    }

    if (cmd == "metrics") {
        if (argc < 4) { std::cerr << "metrics requires: <original> <watermarked>\n"; return 1; }
        cv::Mat a = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        cv::Mat b = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
        if (a.empty() || b.empty()) { std::cerr << "Cannot read images\n"; return 1; }

        std::cout << "MSE:  " << gbo::computeMSE(a, b)  << "\n"
                  << "PSNR: " << gbo::computePSNR(a, b) << " dB\n"
                  << "SSIM: " << gbo::computeSSIM(a, b) << "\n"
                  << "NCC:  " << gbo::computeNCC(a, b)  << "\n";

        // Optional watermark BER
        std::string wm1, wm2;
        for (int i = 4; i < argc; ++i) {
            if (std::string(argv[i]) == "--wm-orig" && i+1 < argc) wm1 = argv[++i];
            if (std::string(argv[i]) == "--wm-extr" && i+1 < argc) wm2 = argv[++i];
        }
        if (!wm1.empty() && !wm2.empty()) {
            cv::Mat w1 = cv::imread(wm1, cv::IMREAD_GRAYSCALE);
            cv::Mat w2 = cv::imread(wm2, cv::IMREAD_GRAYSCALE);
            if (!w1.empty() && !w2.empty())
                std::cout << "BER:  " << gbo::computeBER(w1, w2) << "\n";
        }
        return 0;
    }

    if (cmd == "attack") {
        if (argc < 5) { std::cerr << "attack requires: <image> <output> --type TYPE\n"; return 1; }
        cv::Mat img = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        std::string out = argv[3];
        if (img.empty()) { std::cerr << "Cannot read image: " << argv[2] << "\n"; return 1; }

        std::string type;
        double param = 0;
        bool has_param = false;
        for (int i = 4; i < argc; ++i) {
            if (std::string(argv[i]) == "--type" && i+1 < argc) type = argv[++i];
            if (std::string(argv[i]) == "--param" && i+1 < argc) { param = std::stod(argv[++i]); has_param = true; }
        }

        cv::Mat result;
        if      (type == "jpeg")        result = gbo::attackJPEG(img, has_param ? (int)param : 70);
        else if (type == "brightness+") result = gbo::attackBrightnessIncrease(img, has_param ? (int)param : 30);
        else if (type == "brightness-") result = gbo::attackBrightnessDecrease(img, has_param ? (int)param : 30);
        else if (type == "contrast+")   result = gbo::attackContrastIncrease(img, has_param ? param : 1.2);
        else if (type == "contrast-")   result = gbo::attackContrastDecrease(img, has_param ? param : 0.8);
        else if (type == "salt-pepper") result = gbo::attackSaltPepper(img, has_param ? param : 0.01);
        else if (type == "median")      result = gbo::attackMedianFilter(img, has_param ? (int)param : 3);
        else if (type == "gaussian")    result = gbo::attackGaussianFilter(img, has_param ? (int)param : 3);
        else { std::cerr << "Unknown attack type: " << type << "\n"; return 1; }

        cv::imwrite(out, result);
        std::cout << "Attacked image saved to " << out << "\n";
        return 0;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    printUsage(argv[0]);
    return 1;
}
