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

cv::Mat get_wm(const cv::Mat& cv_image) {
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


/*void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm) {
	double mse_total = 0, psnr_total = 0, ncc_total = 0, ber_total = 0, ssim_total = 0;
	std::vector<cv::Mat> embeded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	for (size_t i = 0; i < 10; ++i) {
		embend_wm(image, new_image, wm);
		get_wm(new_image, new_wm);

		cv::Mat cv_new_image = readImage(new_image);

		cv::Mat cv_new_wm = readImage(new_wm);

		embeded_images.push_back(cv_new_image);

		double mse = computeMSE(cv_image, cv_new_image);
		double psnr = computePSNR(cv_image, cv_new_image);
		double ncc = computeNCC(cv_image, cv_new_image);
		double ber = computeBER(cv_wm, cv_new_wm);
		double ssim = computeSSIM(cv_image, cv_new_image);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;
	}

	double mse_avg = mse_total / 10;
	double psnr_avg = psnr_total / 10;
	double ncc_avg = ncc_total / 10;
	double ber_avg = ber_total / 10;
	double ssim_avg = ssim_total / 10;

	std::cout << "No attack" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0;
	psnr_total = 0;
	ncc_total = 0;
	ber_total = 0;
	ssim_total = 0;

	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = brightnessIncrease(embeded_images[i].clone(), 10);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;

	}

	mse_avg = mse_total / 10;
	psnr_avg = psnr_total / 10;
	ncc_avg = ncc_total / 10;
	ber_avg = ber_total / 10;
	ssim_avg = ssim_total / 10;

	std::cout << "Brightness increase" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0;
	psnr_total = 0;
	ncc_total = 0;
	ber_total = 0;
	ssim_total = 0;

	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = brightnessDecrease(embeded_images[i].clone(), 10);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;

	}

	mse_avg = mse_total / 10;
	psnr_avg = psnr_total / 10;
	ncc_avg = ncc_total / 10;
	ber_avg = ber_total / 10;
	ssim_avg = ssim_total / 10;

	std::cout << "Brightness decrease" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0;
	psnr_total = 0;
	ncc_total = 0;
	ber_total = 0;
	ssim_total = 0;

	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = contrastIncrease(embeded_images[i].clone(), 1.5);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;

	}

	mse_avg = mse_total / 10;
	psnr_avg = psnr_total / 10;
	ncc_avg = ncc_total / 10;
	ber_avg = ber_total / 10;
	ssim_avg = ssim_total / 10;

	std::cout << "Contrast increase" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0;
	psnr_total = 0;
	ncc_total = 0;
	ber_total = 0;
	ssim_total = 0;

	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = contrastDecrease(embeded_images[i].clone(), 0.5);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;

	}

	mse_avg = mse_total / 10;
	psnr_avg = psnr_total / 10;
	ncc_avg = ncc_total / 10;
	ber_avg = ber_total / 10;
	ssim_avg = ssim_total / 10;

	std::cout << "Contrast decrease" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0;
	psnr_total = 0;
	ncc_total = 0;
	ber_total = 0;
	ssim_total = 0;

	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = saltPepperNoise(embeded_images[i].clone(), 0.05);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;

	}

	mse_avg = mse_total / 10;
	psnr_avg = psnr_total / 10;
	ncc_avg = ncc_total / 10;
	ber_avg = ber_total / 10;
	ssim_avg = ssim_total / 10;

	std::cout << "Salt Pepper Noise" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0;
	psnr_total = 0;
	ncc_total = 0;
	ber_total = 0;
	ssim_total = 0;

	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = speckleNoise(embeded_images[i].clone(), 0.05);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse;
		psnr_total += psnr;
		ncc_total += ncc;
		ber_total += ber;
		ssim_total += ssim;

	}

	mse_avg = mse_total / 10;
	psnr_avg = psnr_total / 10;
	ncc_avg = ncc_total / 10;
	ber_avg = ber_total / 10;
	ssim_avg = ssim_total / 10;

	std::cout << "Speсkle Noise" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = histogramEqualization(embeded_images[i].clone());
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Histogram Equalization" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = sharpening(embeded_images[i].clone());
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Sharpening" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = jpegCompression(embeded_images[i].clone(), 90);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "JPEG Compression (QF=90)" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = jpegCompression(embeded_images[i].clone(), 80);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "JPEG Compression (QF=80)" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = jpegCompression(embeded_images[i].clone(), 70);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "JPEG Compression (QF=70)" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = gaussianFiltering(embeded_images[i].clone(), 5);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Gaussian Filtering" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = medianFiltering(embeded_images[i].clone(), 5);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Median Filtering" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = averageFiltering(embeded_images[i].clone(), 5);
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(cv_image, img);
		double psnr = computePSNR(cv_image, img);
		double ncc = computeNCC(cv_image, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(cv_image, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Average Filtering" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = cropFromCorner(embeded_images[i].clone(), 100);
		cv::Mat original_cropped = cropFromCorner(cv_image.clone(), 100); 
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(original_cropped, img); 
		double psnr = computePSNR(original_cropped, img);
		double ncc = computeNCC(original_cropped, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(original_cropped, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Cropping from Corner" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = cropFromCenter(embeded_images[i].clone(), 100);
		cv::Mat original_cropped = cropFromCenter(cv_image.clone(), 100); 
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(original_cropped, img); 
		double psnr = computePSNR(original_cropped, img);
		double ncc = computeNCC(original_cropped, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(original_cropped, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Cropping from Center" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;

	mse_total = 0; psnr_total = 0; ncc_total = 0; ber_total = 0; ssim_total = 0;
	for (size_t i = 0; i < 10; ++i) {
		cv::Mat img = cropFromEdge(embeded_images[i].clone(), 100);
		cv::Mat original_cropped = cropFromEdge(cv_image.clone(), 100); 
		cv::Mat wm = get_wm(img);

		double mse = computeMSE(original_cropped, img); 
		double psnr = computePSNR(original_cropped, img);
		double ncc = computeNCC(original_cropped, img);
		double ber = computeBER(cv_wm, wm);
		double ssim = computeSSIM(original_cropped, img);

		mse_total += mse; psnr_total += psnr; ncc_total += ncc; ber_total += ber; ssim_total += ssim;
	}
	mse_avg = mse_total / 10; psnr_avg = psnr_total / 10; ncc_avg = ncc_total / 10; ber_avg = ber_total / 10; ssim_avg = ssim_total / 10;
	std::cout << "Cropping from Edge" << std::endl;
	std::cout << "Average MSE: " << mse_avg << std::endl;
	std::cout << "Average PSNR: " << psnr_avg << std::endl;
	std::cout << "Average NCC: " << ncc_avg << std::endl;
	std::cout << "Average BER: " << ber_avg << std::endl;
	std::cout << "Average SSIM: " << ssim_avg << std::endl;
}*/

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
	const std::string& output_file = "")
{
	double mse_total = 0, psnr_total = 0, ncc_total = 0, ber_total = 0, ssim_total = 0;

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

		cv::Mat wm = get_wm(img);

		mse_total += metric(original, img);
		psnr_total += computePSNR(original, img);
		ncc_total += computeNCC(original, img);
		ber_total += computeBER(cv_wm, wm);
		ssim_total += computeSSIM(original, img);
	}

	output << "Average MSE: " << mse_total / iterations << std::endl
		<< "Average PSNR: " << psnr_total / iterations << std::endl
		<< "Average NCC: " << ncc_total / iterations << std::endl
		<< "Average BER: " << ber_total / iterations << std::endl
		<< "Average SSIM: " << ssim_total / iterations << std::endl
		<< std::endl;

	output.close();
}

void launch(const std::string& image, const std::string& new_image,const std::string& wm, const std::string& new_wm){
	std::vector<cv::Mat> embeded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	for (size_t i = 0; i < 10; ++i) {
		embend_wm(image, new_image, wm);
		get_wm(new_image, new_wm);
		embeded_images.push_back(readImage(new_image));
	}

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

	std::string result_filename = "results_" + getFileNameWithoutExtension(image) + ".txt";

	for (const auto& attack : attacks) {
		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE; // Пример, метрика должна быть передана сюда
		processAttack(embeded_images, cv_image, cv_wm, attack, metric, 10, result_filename);
	}
}



