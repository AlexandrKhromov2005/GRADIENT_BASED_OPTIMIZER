#ifndef IMAGE_METRICS_H
#define IMAGE_METRICS_H

#include <opencv2/opencv.hpp>

// ���������� ������������������ ������ (MSE) ����� ����� �������������
double computeMSE(const cv::Mat& image1, const cv::Mat& image2);

// ���������� ��������� �������� ������� � ���� (PSNR) ����� ����� �������������
double computePSNR(const cv::Mat& image1, const cv::Mat& image2);

// ���������� ������� ������ (BER) ��� �������� �����������
double computeBER(const cv::Mat& image1, const cv::Mat& image2);

// ���������� ������������� ���������� (NCC) ����� ����� �������������
double computeNCC(const cv::Mat& img1, const cv::Mat& img2);

double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);

#endif // IMAGE_METRICS_H
