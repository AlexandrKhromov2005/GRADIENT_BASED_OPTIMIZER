#include "launch.h"


void embend_wm(const std::string& image, const std::string& new_image, const std::string& wm) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);

	size_t image_size = image_vec.size();
	for (size_t i = 0; i < image_size; ++i) {
		std::cout << "\rProcess: " << i << " / " << image_size << std::flush;
		GBO gbo(wm_vec[i % WM_SIZE], image_vec[i]);
		gbo.main_loop();

	}
	std::cout << "\rProcess: " << image_size << " / " << image_size << std::endl;

	const cv::Mat cv_new_image = merge8x8Blocks(image_vec, cv_image.rows, cv_image.cols);
	writeImage(new_image, cv_new_image);
}

void get_wm(const std::string& image, const std::string& new_image) {
	const cv::Mat cv_image = readImage(image);
	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec(WM_SIZE, 0);

	for (size_t i = 0; i < image_vec.size(); ++i) {
		cv::Mat dbl_block;
		image_vec[i].convertTo(dbl_block, CV_64F);
		cv::Mat dct_block;
		cv::dct(dbl_block, dct_block);
		double s0 = calc_s_zero(dct_block);
		double s1 = calc_s_one(dct_block);
		if (s0 < s1) {
			++wm_vec[i % WM_SIZE];
		}
	}

	for (size_t i = 0; i < WM_SIZE; ++i) {
		switch (wm_vec[i])
		{
		case 0:
			wm_vec[i] = 0;
			break;
		case 1:
			wm_vec[i] = 0;
			break;
		case 2:
			wm_vec[i] = rand() % 2;
			break;
		case 3:
			wm_vec[i] = 1;
			break;
		case 4:
			wm_vec[i] = 1;
			break;
		default:
			break;
		}
	}
	const cv::Mat wm = convertBinaryToWatermark(wm_vec);
	writeImage(new_image, wm);
}

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm) {
	embend_wm(image, new_image, wm);
	get_wm(new_image, new_wm);
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_new_image = readImage(new_image);
	cv::Mat cv_wm = readImage(wm);
	cv::Mat cv_new_wm = readImage(new_wm);

	double mse = computeMSE(cv_image, cv_new_image);
	double psnr = computePSNR(cv_image, cv_new_image);
	double ncc = computeNCC(cv_image, cv_new_image);
	double ber = computeBER(cv_wm, cv_new_wm);
	double ssim = computeSSIM(cv_image, cv_new_image);

	std::cout << "MSE: " << mse << std::endl;
	std::cout << "PSNR: " << psnr << std::endl;
	std::cout << "NCC: " << ncc << std::endl;
	std::cout << "BER: " << ber << std::endl;
	std::cout << "SSIM: " << ssim << std::endl;


}

