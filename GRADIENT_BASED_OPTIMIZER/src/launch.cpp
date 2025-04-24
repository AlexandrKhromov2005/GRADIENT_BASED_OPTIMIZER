#include "launch.h"
#include <omp.h>


void embend_wm(const std::string& image, const std::string& new_image, const std::string& wm, const int index) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);

	initialize_quantization_mats();

	size_t image_size = image_vec.size();

#pragma omp parallel for
	for (size_t i = 0; i < image_size; ++i) {
		GBO gbo(wm_vec[i % WM_SIZE], image_vec[i], index);
		gbo.main_loop();
	}

	const cv::Mat cv_new_image = merge8x8Blocks(image_vec, cv_image.rows, cv_image.cols);
	writeImage(new_image, cv_new_image);
}

void get_wm(const std::string& image, const std::string& new_image, const int index) {
	const cv::Mat cv_image = readImage(image);
	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec(WM_SIZE, 0);

	for (size_t i = 0; i < image_vec.size(); ++i) {
		cv::Mat dbl_block;
		image_vec[i].convertTo(dbl_block, CV_64F);
		cv::Mat dct_block;
		cv::dct(dbl_block, dct_block);
		double s0 = calc_s_zero(dct_block, index);
		double s1 = calc_s_one(dct_block, index);
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

cv::Mat get_wm(const cv::Mat& cv_image, const int index) {
	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec(WM_SIZE, 0);

	for (size_t i = 0; i < image_vec.size(); ++i) {
		cv::Mat dbl_block;
		image_vec[i].convertTo(dbl_block, CV_64F);
		cv::Mat dct_block;
		cv::dct(dbl_block, dct_block);
		double s0 = calc_s_zero(dct_block, index);
		double s1 = calc_s_one(dct_block, index);
		if (s0 < s1) {
			++wm_vec[i % WM_SIZE];
		}
	}

	for (size_t i = 0; i < WM_SIZE; ++i) {
		switch (wm_vec[i]) {
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

	return convertBinaryToWatermark(wm_vec);
}

using AttackFunction = std::function<cv::Mat(const cv::Mat&)>;
using MetricCalculator = std::function<double(const cv::Mat&, const cv::Mat&)>;

struct AttackConfig {
	std::string name;
	AttackFunction attack;
	bool use_cropped_comparison = false;
};

std::string getFileNameWithoutExtension(const std::string& path) {
	size_t lastSlashPos = path.find_last_of('/');
	size_t dotPos = path.find_last_of('.');

	if (lastSlashPos != std::string::npos && dotPos != std::string::npos) {
		return path.substr(lastSlashPos + 1, dotPos - lastSlashPos - 1);
	}

	return "";
}

void processAttack(
	const std::vector<cv::Mat>& embeded_images,
	const cv::Mat& cv_image,
	const cv::Mat& cv_wm,
	const AttackConfig& config,
	MetricCalculator metric,
	int iterations = 10,
	const std::string& output_file = "",
	const int index = 0)
{
	double mse_total = 0, psnr_total = 0, ncc_total = 0, ber_total = 0, ssim_total = 0;
	double max_mse = 0, max_psnr = 0, max_ncc = 0, max_ber = 0, max_ssim = 0;
	double min_mse = DBL_MAX, min_psnr = DBL_MAX, min_ncc = DBL_MAX, min_ber = DBL_MAX, min_ssim = DBL_MAX;



	std::ofstream output(output_file, std::ios::app);
	if (!output.is_open()) {
		std::cerr << "Error opening file: " << output_file << std::endl;
		return;
	}

	output << "Attack: " << config.name << std::endl;

	for (size_t i = 0; i < iterations; ++i) {
		cv::Mat img = config.attack(embeded_images[i].clone());
		cv::Mat original = config.use_cropped_comparison ?
			config.attack(cv_image.clone()) : cv_image.clone();

		cv::Mat wm = get_wm(img, index);

		double mse = metric(original, img);
		double psnr= computePSNR(original, img);
		double ncc = computeNCC(original, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(original, img);

		max_mse = std::max(max_mse, mse);
		min_mse = std::min(min_mse, mse);
		max_psnr = std::max(max_psnr, psnr);
		min_psnr = std::min(min_psnr, psnr);
		max_ncc = std::max(max_ncc, ncc);
		min_ncc = std::min(min_ncc, ncc);
		max_ber = std::max(max_ber, ber);
		min_ber = std::min(min_ber, ber);
		max_ssim = std::max(max_ssim, ssim);
		min_ssim = std::min(min_ssim, ssim);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;
	}

	output << "Average MSE: " << min_mse << " " << mse_total / iterations << " " << max_mse << std::endl
		<< "Average PSNR: " << min_psnr << " " << psnr_total / iterations << " " << max_psnr << std::endl
		<< "Average NCC: " << min_ncc << " " << ncc_total / iterations << " " << max_ncc << std::endl
		<< "Average BER: " << min_ber << " " << ber_total / iterations << " " << max_ber << std::endl
		<< "Average SSIM: " << min_ssim << " " << ssim_total / iterations << " " << max_ssim << std::endl
		<< std::endl;


	output.close();
}

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, const int index) {
	std::vector<cv::Mat> embeded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	//auto start_embed = std::chrono::high_resolution_clock::now();

	for (size_t i = 0; i < 10; ++i) {
		//auto iter_start = std::chrono::high_resolution_clock::now();

		embend_wm(image, new_image, wm, index);
		get_wm(new_image, new_wm, index);
		embeded_images.push_back(readImage(new_image));
	}

	//auto end_embed = std::chrono::high_resolution_clock::now();
	//auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed);

	//std::cout << "\nTotal embedding time: " << embed_duration.count() << " ms" << std::endl;
	//std::cout << "Average time per image: " << embed_duration.count() / 10 << " ms" << std::endl;

	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.5); }},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.5); }},
		{"Salt Pepper Noise", [](const cv::Mat& img) { return saltPepperNoise(img, 0.05); }},
		{"Speckle Noise", [](const cv::Mat& img) { return speckleNoise(img, 0.05); }},
		{"Histogram Equalization", [](const cv::Mat& img) { return histogramEqualization(img); }},
		{"Sharpening", [](const cv::Mat& img) { return sharpening(img); }},
		{"JPEG Compression (QF=90)", [](const cv::Mat& img) { return jpegCompression(img, 90); }},
		{"JPEG Compression (QF=80)", [](const cv::Mat& img) { return jpegCompression(img, 80); }},
		{"JPEG Compression (QF=70)", [](const cv::Mat& img) { return jpegCompression(img, 70); }},
		{"Gaussian Filtering", [](const cv::Mat& img) { return gaussianFiltering(img, 5); }},
		{"Median Filtering", [](const cv::Mat& img) { return medianFiltering(img, 5); }},
		{"Average Filtering", [](const cv::Mat& img) { return averageFiltering(img, 5); }},
		{"Cropping from Corner", [](const cv::Mat& img) { return cropFromCorner(img, 100); }, true},
		{"Cropping from Center", [](const cv::Mat& img) { return cropFromCenter(img, 100); }, true},
		{"Cropping from Edge", [](const cv::Mat& img) { return cropFromEdge(img, 100); }, true}
	};

	std::string result_filename = "res" + std::to_string(index) + "/results_" + getFileNameWithoutExtension(image) + ".txt";

	auto start_attacks = std::chrono::high_resolution_clock::now();

	for (const auto& attack : attacks) {

		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE;
		processAttack(embeded_images, cv_image, cv_wm, attack, metric, 10, result_filename, index);
	}
}



