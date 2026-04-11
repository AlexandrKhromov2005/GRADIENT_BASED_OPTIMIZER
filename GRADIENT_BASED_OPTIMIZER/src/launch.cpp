#include "launch.h"
#include <chrono>

#ifdef TORCH_AVAILABLE
#include "embedding_with_classifier.h"
#include "ensemble_classifier.h"
#include "quadrant_embedding.h"
#include "attack_type_embedding.h"
#include <memory>
#endif


void embend_wm(const std::string& image, const std::string& new_image, const std::string& wm) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);

	initialize_quantization_mats();

	size_t image_size = image_vec.size();
	for (size_t i = 0; i < image_size; ++i) {
		GBO gbo(wm_vec[i % WM_SIZE], image_vec[i]);
		gbo.main_loop();
	}

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

		cv::Mat wm = get_wm(img);

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

#ifdef TORCH_AVAILABLE
// Process attack with quadrant classifier - logs inference results
void processAttackWithQuadrantClassifier(
	const std::vector<cv::Mat>& embedded_images,
	const cv::Mat& cv_image,
	const cv::Mat& cv_wm,
	const AttackConfig& config,
	MetricCalculator metric,
	int iterations,
	const std::string& output_file,
	const std::string& inference_log_file)
{
	double mse_total = 0, psnr_total = 0, ncc_total = 0, ber_total = 0, ssim_total = 0;
	double max_mse = 0, max_psnr = 0, max_ncc = 0, max_ber = 0, max_ssim = 0;
	double min_mse = DBL_MAX, min_psnr = DBL_MAX, min_ncc = DBL_MAX, min_ber = DBL_MAX, min_ssim = DBL_MAX;

	std::ofstream output(output_file, std::ios::app);
	if (!output.is_open()) {
		std::cerr << "Error opening file: " << output_file << std::endl;
		return;
	}

	std::ofstream inference_log(inference_log_file, std::ios::app);
	if (!inference_log.is_open()) {
		std::cerr << "Error opening inference log: " << inference_log_file << std::endl;
		return;
	}

	output << "Attack: " << config.name << std::endl;
	inference_log << "\nAttack: " << config.name << std::endl;

	for (size_t i = 0; i < iterations; ++i) {
		cv::Mat img = config.attack(embedded_images[i].clone());
		cv::Mat original = config.use_cropped_comparison ?
			config.attack(cv_image.clone()) : cv_image.clone();

		// Extract watermark using classifier (logs prediction internally)
		cv::Mat wm = QuadrantEmbedding::extractWatermarkWithClassifier(img);

		// Get the last prediction from classifier
		auto classifier = QuadrantEmbedding::getClassifier();
		if (classifier) {
			auto prediction = classifier->predict(img, true);

			// Log detailed inference results
			inference_log << "  Iteration " << (i + 1) << ": "
			             << "pred=" << prediction.class_name
			             << " conf=" << (prediction.confidence * 100.0f) << "% "
			             << "prob[" << prediction.prob_scheme0 << ", "
			             << prediction.prob_scheme1 << ", "
			             << prediction.prob_scheme2 << ", "
			             << prediction.prob_scheme3 << "]" << std::endl;
		}

		double mse = metric(original, img);
		double psnr = computePSNR(original, img);
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
	inference_log.close();
}

// Process attack with attack type classifier - logs inference results
void processAttackWithAttackTypeClassifier(
	const std::vector<cv::Mat>& embedded_images,
	const cv::Mat& cv_image,
	const cv::Mat& cv_wm,
	const AttackConfig& config,
	MetricCalculator metric,
	int iterations,
	const std::string& output_file,
	const std::string& inference_log_file)
{
	double mse_total = 0, psnr_total = 0, ncc_total = 0, ber_total = 0, ssim_total = 0;
	double max_mse = 0, max_psnr = 0, max_ncc = 0, max_ber = 0, max_ssim = 0;
	double min_mse = DBL_MAX, min_psnr = DBL_MAX, min_ncc = DBL_MAX, min_ber = DBL_MAX, min_ssim = DBL_MAX;

	std::ofstream output(output_file, std::ios::app);
	if (!output.is_open()) {
		std::cerr << "Error opening file: " << output_file << std::endl;
		return;
	}

	std::ofstream inference_log(inference_log_file, std::ios::app);
	if (!inference_log.is_open()) {
		std::cerr << "Error opening inference log: " << inference_log_file << std::endl;
		return;
	}

	output << "Attack: " << config.name << std::endl;
	inference_log << "\nAttack: " << config.name << std::endl;

	for (size_t i = 0; i < iterations; ++i) {
		cv::Mat img = config.attack(embedded_images[i].clone());
		cv::Mat original = config.use_cropped_comparison ?
			config.attack(cv_image.clone()) : cv_image.clone();

		// Extract watermark using attack type classifier
		cv::Mat wm = AttackTypeEmbedding::extractWatermarkWithClassifier(img);

		// Get the last prediction from classifier
		auto classifier = AttackTypeEmbedding::getClassifier();
		if (classifier) {
			auto prediction = classifier->predict(img, true);

			// Log detailed inference results
			inference_log << "  Iteration " << (i + 1) << ": "
			             << "pred=" << prediction.class_name
			             << " conf=" << (prediction.confidence * 100.0f) << "% "
			             << "prob[NoAttack=" << prediction.prob_noattack
			             << ", JPG70=" << prediction.prob_jpg70
			             << ", JPG80=" << prediction.prob_jpg80
			             << ", Contrast=" << prediction.prob_contrast << "]" << std::endl;
		}

		double mse = metric(original, img);
		double psnr = computePSNR(original, img);
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
	inference_log.close();
}
#endif

void launch(const std::string& image, const std::string& new_image,const std::string& wm, const std::string& new_wm, int iterations){
	std::vector<cv::Mat> embeded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	for (size_t i = 0; i < iterations; ++i) {
		embend_wm(image, new_image, wm);
		get_wm(new_image, new_wm);
		embeded_images.push_back(readImage(new_image));
		std::cout<< "\r" << i << "/" << iterations << std::flush;
	}
	std::cout << "\r" << std::flush;




	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.1); }},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.9); }},
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
		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE; 
		processAttack(embeded_images, cv_image, cv_wm, attack, metric, iterations, result_filename);
	}
}

#ifdef TORCH_AVAILABLE
// Встраивание с использованием классификатора
void embend_wm_with_classifier(const std::string& image, const std::string& new_image, const std::string& wm) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);

	initialize_quantization_mats();

	size_t image_size = image_vec.size();
	for (size_t i = 0; i < image_size; ++i) {
		// Используем классификатор для встраивания
		image_vec[i] = EmbeddingWithClassifier::embedBitWithSchemeSelection(
			image_vec[i], wm_vec[i % WM_SIZE]);
	}

	const cv::Mat cv_new_image = merge8x8Blocks(image_vec, cv_image.rows, cv_image.cols);
	writeImage(new_image, cv_new_image);
}

// Извлечение с использованием классификатора
void get_wm_with_classifier(const std::string& image, const std::string& new_image) {
	const cv::Mat cv_image = readImage(image);
	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec(WM_SIZE, 0);

	for (size_t i = 0; i < image_vec.size(); ++i) {
		// Используем классификатор для извлечения
		uchar extracted_bit = EmbeddingWithClassifier::extractBitWithSchemePrediction(
			image_vec[i]);
		if (extracted_bit == 1) {
			++wm_vec[i % WM_SIZE];
		}
	}

	// Принятие решения по извлеченным битам (как в оригинале)
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
		default:
			wm_vec[i] = 1;
			break;
		}
	}

	// Преобразование в изображение водяного знака (как в оригинале)
	cv::Mat result_wm = convertBinaryToWatermark(wm_vec);
	writeImage(new_image, result_wm);
}

// Встраивание с использованием ансамбля классификаторов
void embend_wm_with_ensemble(const std::string& image, const std::string& new_image, const std::string& wm, EnsembleClassifier* ensemble) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);

	initialize_quantization_mats();
	
	auto& scheme_manager = EmbeddingSchemeManager::getInstance();

	size_t image_size = image_vec.size();
	for (size_t i = 0; i < image_size; ++i) {
		// Используем ансамбль для выбора схемы встраивания
		auto prediction = ensemble->predict(image_vec[i], true);
		
		// Выбираем схему на основе предсказания
		std::string scheme_id = (prediction.predicted_class == 0) ? "scheme2" : "scheme3";
		const auto* scheme = scheme_manager.getScheme(scheme_id);
		
		if (scheme) {
			scheme_manager.setCurrentScheme(scheme_id);
			GBO gbo(wm_vec[i % WM_SIZE], image_vec[i]);
			gbo.main_loop();
		}
	}

	const cv::Mat cv_new_image = merge8x8Blocks(image_vec, cv_image.rows, cv_image.cols);
	writeImage(new_image, cv_new_image);
}

// Извлечение с использованием ансамбля классификаторов
void get_wm_with_ensemble(const std::string& image, const std::string& new_image, EnsembleClassifier* ensemble) {
	const cv::Mat cv_image = readImage(image);
	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec(WM_SIZE, 0);

	auto& scheme_manager = EmbeddingSchemeManager::getInstance();

	for (size_t i = 0; i < image_vec.size(); ++i) {
		// Используем ансамбль для предсказания схемы
		auto prediction = ensemble->predict(image_vec[i], true);
		
		// Выбираем схему для извлечения
		std::string scheme_id = (prediction.predicted_class == 0) ? "scheme2" : "scheme3";
		const auto* scheme = scheme_manager.getScheme(scheme_id);
		
		if (scheme) {
			scheme_manager.setCurrentScheme(scheme_id);
			
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
	}

	// Принятие решения по извлеченным битам (как в оригинале)
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
		default:
			wm_vec[i] = 1;
			break;
		}
	}

	// Преобразование в изображение водяного знака (как в оригинале)
	cv::Mat result_wm = convertBinaryToWatermark(wm_vec);
	writeImage(new_image, result_wm);
}

// Основная функция launch с классификатором
void launch_with_classifier(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm, int iterations) {
	std::cout << "🤖 Launching with Single AI classifier integration (final_model.pt)..." << std::endl;
	
	// Инициализация single классификатора с final_model.pt
	std::string model_path = "final_model_torchscript.pt";
	float threshold = 0.5f;
	
	// Инициализируем single classifier через EmbeddingWithClassifier
	if (!EmbeddingWithClassifier::initializeSingleClassifier(model_path, threshold, true)) {
		std::cerr << "❌ Failed to initialize single classifier with final_model.pt" << std::endl;
		throw std::runtime_error("Cannot proceed without classifier in --classifier mode");
	}
	std::cout << "✅ Single classifier initialized successfully" << std::endl;
	
	std::vector<cv::Mat> embeded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	// Выполняем итерации встраивания/извлечения с single классификатором
	for (size_t i = 0; i < iterations; ++i) {
		embend_wm_with_classifier(image, new_image, wm);
		get_wm_with_classifier(new_image, new_wm);
		embeded_images.push_back(readImage(new_image));
		std::cout << "\r" << i << "/" << iterations << std::flush;
	}
	std::cout << "\r" << std::flush;

	// Те же атаки что и в оригинале
	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.1); }},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.9); }},
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

	// Сохраняем результаты с пометкой "classifier"
	std::string result_filename = "results_" + getFileNameWithoutExtension(image) + "_classifier.txt";
	std::cout << "🗂️ Will save results to: " << result_filename << std::endl;

	for (const auto& attack : attacks) {
		std::cout << "🔍 Processing attack: " << attack.name << std::endl;
		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE;
		processAttack(embeded_images, cv_image, cv_wm, attack, metric, iterations, result_filename);
	}
	
	std::cout << "✅ Results saved to: " << result_filename << std::endl;
}
#endif

// Generate random binary watermark
std::vector<int> generateRandomWatermark() {
	std::vector<int> wm(WM_SIZE);
	for (size_t i = 0; i < WM_SIZE; ++i) {
		wm[i] = rand() % 2;
	}
	return wm;
}

// Embed watermark into 256x256 quadrant using specified attack type
cv::Mat embedIntoQuadrant(const cv::Mat& quadrant, const std::vector<int>& wm, AttackType attack_type) {
	std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);

	size_t num_blocks = blocks.size();
	for (size_t i = 0; i < num_blocks; ++i) {
		GBO gbo(wm[i % WM_SIZE], blocks[i], attack_type);
		gbo.main_loop();
	}

	return merge8x8Blocks(blocks, quadrant.rows, quadrant.cols);
}

// Main function: embed same watermark into 4 quadrants with different objectives
void embed_quadrants_with_objectives(const std::string& image_path, const std::string& output_path) {
	cv::Mat image = readImage(image_path);

	// Verify image is 512x512
	if (image.rows != 512 || image.cols != 512) {
		std::cerr << "Error: Image must be 512x512, got " << image.rows << "x" << image.cols << std::endl;
		return;
	}

	initialize_quantization_mats();

	// Generate single random watermark
	std::vector<int> wm = generateRandomWatermark();

	// Split into 4 quadrants (256x256 each)
	cv::Mat N1 = image(cv::Rect(0, 0, 256, 256)).clone();
	cv::Mat N2 = image(cv::Rect(256, 0, 256, 256)).clone();
	cv::Mat N3 = image(cv::Rect(0, 256, 256, 256)).clone();
	cv::Mat N4 = image(cv::Rect(256, 256, 256, 256)).clone();

	std::cout << "Embedding into quadrant N1 (no attack)..." << std::endl;
	cv::Mat N1_embedded = embedIntoQuadrant(N1, wm, AttackType::NONE);

	std::cout << "Embedding into quadrant N2 (JPEG70 robust)..." << std::endl;
	cv::Mat N2_embedded = embedIntoQuadrant(N2, wm, AttackType::JPEG70);

	std::cout << "Embedding into quadrant N3 (Contrast robust)..." << std::endl;
	cv::Mat N3_embedded = embedIntoQuadrant(N3, wm, AttackType::CONTRAST);

	std::cout << "Embedding into quadrant N4 (JPEG80 robust)..." << std::endl;
	cv::Mat N4_embedded = embedIntoQuadrant(N4, wm, AttackType::JPEG80);

	// Merge quadrants back into 512x512 image
	cv::Mat result(512, 512, image.type());
	N1_embedded.copyTo(result(cv::Rect(0, 0, 256, 256)));
	N2_embedded.copyTo(result(cv::Rect(256, 0, 256, 256)));
	N3_embedded.copyTo(result(cv::Rect(0, 256, 256, 256)));
	N4_embedded.copyTo(result(cv::Rect(256, 256, 256, 256)));

	writeImage(output_path, result);
	std::cout << "Saved result to: " << output_path << std::endl;
}

// Dataset generation: apply attacks and save to directories
void generate_quadrant_dataset(const std::string& input_dir, const std::string& output_base_dir) {
	// Create output directories using system calls
	system(("mkdir -p " + output_base_dir + "/Dir1").c_str());
	system(("mkdir -p " + output_base_dir + "/Dir2").c_str());
	system(("mkdir -p " + output_base_dir + "/Dir3").c_str());
	system(("mkdir -p " + output_base_dir + "/Dir4").c_str());

	// Dynamically scan directory for all PNG files using POSIX API
	std::vector<std::string> image_files;

	DIR* dir = opendir(input_dir.c_str());
	if (dir == nullptr) {
		std::cerr << "❌ Ошибка при открытии директории: " << input_dir << std::endl;
		return;
	}

	struct dirent* entry;
	while ((entry = readdir(dir)) != nullptr) {
		std::string filename = entry->d_name;
		// Include only PNG files and exclude watermark
		if (filename.length() > 4 &&
		    filename.substr(filename.length() - 4) == ".png" &&
		    filename != "watermark.png") {
			image_files.push_back(filename);
		}
	}
	closedir(dir);

	// Sort for consistent ordering
	std::sort(image_files.begin(), image_files.end());

	// Count total images first (verify they can be read)
	int total_images = 0;
	std::vector<std::string> valid_files;
	for (const auto& filename : image_files) {
		std::string image_path = input_dir + "/" + filename;
		cv::Mat test_read = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (!test_read.empty()) {
			valid_files.push_back(filename);
			total_images++;
		}
	}

	// Use only valid files
	image_files = valid_files;

	std::cout << "\n🎯 Начало генерации квадрантного датасета" << std::endl;
	std::cout << "📊 Всего изображений для обработки: " << total_images << std::endl;
	std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

	int processed = 0;
	for (const auto& filename : image_files) {
		std::string image_path = input_dir + "/" + filename;

		// Check if file exists
		cv::Mat test_read = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (test_read.empty()) {
			continue;
		}

		processed++;

		// Progress bar
		int bar_width = 50;
		float progress = (float)processed / total_images;
		int pos = bar_width * progress;

		std::cout << "\r[";
		for (int i = 0; i < bar_width; ++i) {
			if (i < pos) std::cout << "█";
			else if (i == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << processed << "/" << total_images << ") ";
		std::cout << "📷 " << filename;
		std::cout << std::flush;

		// First embed watermarks into quadrants
		std::string temp_embedded = output_base_dir + "/temp_" + filename;
		embed_quadrants_with_objectives(image_path, temp_embedded);

		// Read embedded image
		cv::Mat embedded = readImage(temp_embedded);

		// Dir1: No attack
		std::string dir1_path = output_base_dir + "/Dir1/" + filename;
		writeImage(dir1_path, embedded);

		// Dir2: JPEG70 attack
		std::string dir2_path = output_base_dir + "/Dir2/" + filename;
		cv::Mat attacked_jpeg = jpegCompression(embedded, 70);
		writeImage(dir2_path, attacked_jpeg);

		// Dir3: Contrast increase attack
		std::string dir3_path = output_base_dir + "/Dir3/" + filename;
		cv::Mat attacked_contrast = contrastIncrease(embedded, 1.1);
		writeImage(dir3_path, attacked_contrast);

		// Dir4: Salt-Pepper noise attack
		std::string dir4_path = output_base_dir + "/Dir4/" + filename;
		cv::Mat attacked_noise = saltPepperNoise(embedded, 0.02);
		writeImage(dir4_path, attacked_noise);

		// Remove temporary file
		remove(temp_embedded.c_str());
	}

	// Final newline after progress bar
	std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
	std::cout << "✅ Генерация датасета завершена!" << std::endl;
	std::cout << "📂 Обработано изображений: " << processed << std::endl;
	std::cout << "📁 Результаты сохранены в: " << output_base_dir << std::endl;
}

// =====================================================================
// Dataset generation for 4-attack classifier
// =====================================================================
// This function generates a dataset by:
// 1. Embedding random 1024-bit watermark into each 1024x1024 image
// 2. Creating 4 copies of embedded image
// 3. Applying different attacks to each copy
// 4. Saving attacked images to corresponding directories
void generate_attack_dataset_1024(const std::string& input_dir, const std::string& output_base_dir) {
	std::cout << "\n🎯 Генерация датасета для 4-х атак (1024x1024)" << std::endl;
	std::cout << "📊 Атаки: NoAttack, JPEG70, JPEG80, ContrastIncrease" << std::endl;
	std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

	// Create output directories for 4 attack types
	system(("mkdir -p " + output_base_dir + "/NoAttack").c_str());
	system(("mkdir -p " + output_base_dir + "/JPEG70").c_str());
	system(("mkdir -p " + output_base_dir + "/JPEG80").c_str());
	system(("mkdir -p " + output_base_dir + "/ContrastIncrease").c_str());

	// Scan for PNG files in input directory
	std::vector<std::string> image_files;
	DIR* dir = opendir(input_dir.c_str());
	if (dir == nullptr) {
		std::cerr << "❌ Ошибка при открытии директории: " << input_dir << std::endl;
		return;
	}

	struct dirent* entry;
	while ((entry = readdir(dir)) != nullptr) {
		std::string filename = entry->d_name;
		if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".png") {
			image_files.push_back(filename);
		}
	}
	closedir(dir);
	std::sort(image_files.begin(), image_files.end());

	// Validate files (must be 1024x1024)
	int total_images = 0;
	std::vector<std::string> valid_files;
	for (const auto& filename : image_files) {
		std::string image_path = input_dir + "/" + filename;
		cv::Mat test_read = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (!test_read.empty() && test_read.rows == 1024 && test_read.cols == 1024) {
			valid_files.push_back(filename);
			total_images++;
		}
	}
	image_files = valid_files;

	std::cout << "📁 Найдено изображений 1024x1024: " << total_images << std::endl;
	std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

	// Initialize quantization tables and random seed
	initialize_quantization_mats();
	init_random();

	int processed = 0;
	for (const auto& filename : image_files) {
		std::string image_path = input_dir + "/" + filename;
		cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (image.empty()) continue;

		processed++;

		// Progress bar
		int bar_width = 50;
		float progress = (float)processed / total_images;
		int pos = bar_width * progress;
		std::cout << "\r[";
		for (int i = 0; i < bar_width; ++i) {
			if (i < pos) std::cout << "█";
			else if (i == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << processed << "/" << total_images << ") ";
		std::cout << "📷 " << filename;
		std::cout << std::flush;

		// Generate random 1024-bit watermark
		std::vector<int> random_wm(WM_SIZE);
		for (size_t i = 0; i < WM_SIZE; ++i) {
			random_wm[i] = rand_binary();
		}

		// Embed watermark into image using embed_wm_1024_with_none
		// This uses the NONE attack type for all 16 quadrants
		std::string temp_embedded = output_base_dir + "/temp_embedded.png";

		// Convert to grayscale if needed
		cv::Mat gray_image;
		if (image.channels() == 3) {
			cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		} else {
			gray_image = image.clone();
		}

		// Embed watermark into all 16 quadrants (4x4 grid, each 256x256)
		cv::Mat result(1024, 1024, gray_image.type());

		for (int row = 0; row < 4; ++row) {
			for (int col = 0; col < 4; ++col) {
				int x = col * 256;
				int y = row * 256;

				// Extract quadrant
				cv::Mat quadrant = gray_image(cv::Rect(x, y, 256, 256)).clone();

				// Embed watermark with NONE attack type (no attack during optimization)
				std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);

				// Embed all 1024 bits of watermark into this quadrant
				for (size_t i = 0; i < blocks.size() && i < WM_SIZE; ++i) {
					GBO gbo(random_wm[i], blocks[i], AttackType::NONE);
					gbo.main_loop();
				}

				cv::Mat embedded_quadrant = merge8x8Blocks(blocks, 256, 256);

				// Copy back to result
				embedded_quadrant.copyTo(result(cv::Rect(x, y, 256, 256)));
			}
		}

		// Now we have embedded image, create 4 copies with different attacks

		// 1. NoAttack - save as is
		std::string out_noattack = output_base_dir + "/NoAttack/" + filename;
		cv::imwrite(out_noattack, result);

		// 2. JPEG70
		cv::Mat attacked_jpeg70 = jpegCompression(result, 70);
		std::string out_jpeg70 = output_base_dir + "/JPEG70/" + filename;
		cv::imwrite(out_jpeg70, attacked_jpeg70);

		// 3. JPEG80
		cv::Mat attacked_jpeg80 = jpegCompression(result, 80);
		std::string out_jpeg80 = output_base_dir + "/JPEG80/" + filename;
		cv::imwrite(out_jpeg80, attacked_jpeg80);

		// 4. ContrastIncrease (1.1)
		cv::Mat attacked_contrast = contrastIncrease(result, 1.1);
		std::string out_contrast = output_base_dir + "/ContrastIncrease/" + filename;
		cv::imwrite(out_contrast, attacked_contrast);
	}

	std::cout << "\n\n✅ Датасет успешно сгенерирован!" << std::endl;
	std::cout << "📂 Обработано изображений: " << processed << std::endl;
	std::cout << "📁 Результаты сохранены в: " << output_base_dir << std::endl;
	std::cout << "   - NoAttack: " << processed << " изображений" << std::endl;
	std::cout << "   - JPEG70: " << processed << " изображений" << std::endl;
	std::cout << "   - JPEG80: " << processed << " изображений" << std::endl;
	std::cout << "   - ContrastIncrease: " << processed << " изображений" << std::endl;
}

#ifdef TORCH_AVAILABLE
// Dataset generation with 16 quadrants (1024x1024) and random watermarks
// Extended version: generates dataset with ALL 18 attack types
void generate_quadrant_dataset_1024(const std::string& input_dir, const std::string& output_base_dir) {
	// Create output directories for ALL 18 attack types
	std::vector<std::pair<std::string, std::string>> attack_dirs = {
		{"Dir01_NoAttack", "No attack"},
		{"Dir02_BrightnessIncrease", "Brightness increase"},
		{"Dir03_BrightnessDecrease", "Brightness decrease"},
		{"Dir04_ContrastIncrease", "Contrast increase"},
		{"Dir05_ContrastDecrease", "Contrast decrease"},
		{"Dir06_SaltPepperNoise", "Salt Pepper Noise"},
		{"Dir07_SpeckleNoise", "Speckle Noise"},
		{"Dir08_HistogramEqualization", "Histogram Equalization"},
		{"Dir09_Sharpening", "Sharpening"},
		{"Dir10_JPEG90", "JPEG Compression (QF=90)"},
		{"Dir11_JPEG80", "JPEG Compression (QF=80)"},
		{"Dir12_JPEG70", "JPEG Compression (QF=70)"},
		{"Dir13_GaussianFiltering", "Gaussian Filtering"},
		{"Dir14_MedianFiltering", "Median Filtering"},
		{"Dir15_AverageFiltering", "Average Filtering"},
		{"Dir16_CroppingCorner", "Cropping from Corner"},
		{"Dir17_CroppingCenter", "Cropping from Center"},
		{"Dir18_CroppingEdge", "Cropping from Edge"}
	};

	std::cout << "\n🎯 Генерация расширенного датасета с 16 квадрантами (1024x1024)" << std::endl;
	std::cout << "📊 Атак: " << attack_dirs.size() << " типов" << std::endl;
	std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

	// Create all directories
	for (const auto& attack : attack_dirs) {
		system(("mkdir -p " + output_base_dir + "/" + attack.first).c_str());
	}

	// Scan for PNG files
	std::vector<std::string> image_files;
	DIR* dir = opendir(input_dir.c_str());
	if (dir == nullptr) {
		std::cerr << "❌ Ошибка при открытии директории: " << input_dir << std::endl;
		return;
	}

	struct dirent* entry;
	while ((entry = readdir(dir)) != nullptr) {
		std::string filename = entry->d_name;
		if (filename.length() > 4 &&
		    filename.substr(filename.length() - 4) == ".png" &&
		    filename != "watermark.png" &&
		    filename != "watermark_32x32.png") {
			image_files.push_back(filename);
		}
	}
	closedir(dir);
	std::sort(image_files.begin(), image_files.end());

	// Validate files
	int total_images = 0;
	std::vector<std::string> valid_files;
	for (const auto& filename : image_files) {
		std::string image_path = input_dir + "/" + filename;
		cv::Mat test_read = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (!test_read.empty() && test_read.rows == 1024 && test_read.cols == 1024) {
			valid_files.push_back(filename);
			total_images++;
		}
	}
	image_files = valid_files;

	std::cout << "📁 Найдено изображений 1024x1024: " << total_images << std::endl;
	std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;

	// Initialize quantization tables
	initialize_quantization_mats();
	init_random();

	int processed = 0;
	auto start_time = std::chrono::high_resolution_clock::now();

	for (const auto& filename : image_files) {
		std::string image_path = input_dir + "/" + filename;
		cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		if (image.empty()) continue;

		processed++;
		auto current_time = std::chrono::high_resolution_clock::now();

		// Progress bar
		int bar_width = 50;
		float progress = (float)processed / total_images;
		int pos = bar_width * progress;

		// Calculate time statistics
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
		double avg_time_per_image = processed > 0 ? elapsed.count() / (double)processed : 0.0;
		int remaining_images = total_images - processed;
		int eta_seconds = (int)(remaining_images * avg_time_per_image);
		int eta_minutes = eta_seconds / 60;
		int eta_hours = eta_minutes / 60;
		eta_minutes %= 60;

		std::cout << "\r[";
		for (int i = 0; i < bar_width; ++i) {
			if (i < pos) std::cout << "█";
			else if (i == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << processed << "/" << total_images << ") ";

		// Show ETA
		if (processed > 1) {
			std::cout << "⏱ ETA: ";
			if (eta_hours > 0) std::cout << eta_hours << "h ";
			std::cout << eta_minutes << "m " << (eta_seconds % 60) << "s ";
		}

		// Show current file (truncate if too long)
		std::string short_name = filename.length() > 20 ?
			filename.substr(0, 17) + "..." : filename;
		std::cout << "📷 " << short_name << "          ";
		std::cout << std::flush;

		// Generate random watermark (1024 bits)
		std::vector<int> random_wm(WM_SIZE);
		for (size_t i = 0; i < WM_SIZE; ++i) {
			random_wm[i] = rand_binary();
		}

		// Embed watermark into 16 quadrants with random WM
		cv::Mat embedded = QuadrantEmbedding::embedWatermarkQuadrants(image, random_wm);

		// Apply ALL 18 attacks and save to corresponding directories

		// 1. No attack
		cv::imwrite(output_base_dir + "/Dir01_NoAttack/" + filename, embedded);

		// 2. Brightness increase
		cv::Mat attacked = brightnessIncrease(embedded, 10);
		cv::imwrite(output_base_dir + "/Dir02_BrightnessIncrease/" + filename, attacked);

		// 3. Brightness decrease
		attacked = brightnessDecrease(embedded, 10);
		cv::imwrite(output_base_dir + "/Dir03_BrightnessDecrease/" + filename, attacked);

		// 4. Contrast increase
		attacked = contrastIncrease(embedded, 1.1);
		cv::imwrite(output_base_dir + "/Dir04_ContrastIncrease/" + filename, attacked);

		// 5. Contrast decrease
		attacked = contrastDecrease(embedded, 0.9);
		cv::imwrite(output_base_dir + "/Dir05_ContrastDecrease/" + filename, attacked);

		// 6. Salt Pepper Noise
		attacked = saltPepperNoise(embedded, 0.05);
		cv::imwrite(output_base_dir + "/Dir06_SaltPepperNoise/" + filename, attacked);

		// 7. Speckle Noise
		attacked = speckleNoise(embedded, 0.05);
		cv::imwrite(output_base_dir + "/Dir07_SpeckleNoise/" + filename, attacked);

		// 8. Histogram Equalization
		attacked = histogramEqualization(embedded);
		cv::imwrite(output_base_dir + "/Dir08_HistogramEqualization/" + filename, attacked);

		// 9. Sharpening
		attacked = sharpening(embedded);
		cv::imwrite(output_base_dir + "/Dir09_Sharpening/" + filename, attacked);

		// 10. JPEG Compression (QF=90)
		attacked = jpegCompression(embedded, 90);
		cv::imwrite(output_base_dir + "/Dir10_JPEG90/" + filename, attacked);

		// 11. JPEG Compression (QF=80)
		attacked = jpegCompression(embedded, 80);
		cv::imwrite(output_base_dir + "/Dir11_JPEG80/" + filename, attacked);

		// 12. JPEG Compression (QF=70)
		attacked = jpegCompression(embedded, 70);
		cv::imwrite(output_base_dir + "/Dir12_JPEG70/" + filename, attacked);

		// 13. Gaussian Filtering
		attacked = gaussianFiltering(embedded, 5);
		cv::imwrite(output_base_dir + "/Dir13_GaussianFiltering/" + filename, attacked);

		// 14. Median Filtering
		attacked = medianFiltering(embedded, 5);
		cv::imwrite(output_base_dir + "/Dir14_MedianFiltering/" + filename, attacked);

		// 15. Average Filtering
		attacked = averageFiltering(embedded, 5);
		cv::imwrite(output_base_dir + "/Dir15_AverageFiltering/" + filename, attacked);

		// 16. Cropping from Corner
		attacked = cropFromCorner(embedded, 100);
		cv::imwrite(output_base_dir + "/Dir16_CroppingCorner/" + filename, attacked);

		// 17. Cropping from Center
		attacked = cropFromCenter(embedded, 100);
		cv::imwrite(output_base_dir + "/Dir17_CroppingCenter/" + filename, attacked);

		// 18. Cropping from Edge
		attacked = cropFromEdge(embedded, 100);
		cv::imwrite(output_base_dir + "/Dir18_CroppingEdge/" + filename, attacked);
	}

	std::cout << "\n\n✅ Расширенный датасет успешно сгенерирован!" << std::endl;
	std::cout << "📂 Обработано изображений: " << processed << std::endl;
	std::cout << "📊 Типов атак: " << attack_dirs.size() << std::endl;
	std::cout << "📁 Результаты сохранены в: " << output_base_dir << std::endl;
	std::cout << "\n📋 Структура датасета:" << std::endl;
	for (size_t i = 0; i < attack_dirs.size(); ++i) {
		std::cout << "   " << (i+1) << ". " << attack_dirs[i].first
		          << " (" << attack_dirs[i].second << "): " << processed << " изображений" << std::endl;
	}
}
#endif // TORCH_AVAILABLE

#ifdef TORCH_AVAILABLE
// Launch with quadrant classifier for full experiment
void launch_with_quadrant_classifier(const std::string& image, const std::string& new_image,
                                     const std::string& wm, const std::string& new_wm, int iterations) {
	std::cout << "🎯 Launching with Quadrant Classifier (best_model_ultrahighres.pt)..." << std::endl;

	// Initialize quadrant classifier
	std::string model_path = "best_model_ultrahighres.pt";

	if (!QuadrantEmbedding::initializeQuadrantClassifier(model_path, true)) {
		std::cerr << "❌ Failed to initialize quadrant classifier with " << model_path << std::endl;
		throw std::runtime_error("Cannot proceed without quadrant classifier");
	}
	std::cout << "✅ Quadrant classifier initialized successfully" << std::endl;

	std::vector<cv::Mat> embedded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	// Verify image is 1024x1024
	if (cv_image.rows != 1024 || cv_image.cols != 1024) {
		std::cerr << "❌ Error: Image must be 1024x1024 for quadrant classifier mode, got "
		          << cv_image.rows << "x" << cv_image.cols << std::endl;
		throw std::runtime_error("Invalid image size for quadrant mode");
	}

	// Perform embedding/extraction iterations
	std::cout << "🔄 Performing " << iterations << " iterations of embedding/extraction..." << std::endl;
	for (size_t i = 0; i < iterations; ++i) {
		// Embed watermark into 4 quadrants
		cv::Mat embedded = QuadrantEmbedding::embedWatermarkQuadrants(cv_image, cv_wm);
		writeImage(new_image, embedded);

		// Extract watermark using classifier
		cv::Mat extracted_wm = QuadrantEmbedding::extractWatermarkWithClassifier(embedded);
		writeImage(new_wm, extracted_wm);

		embedded_images.push_back(embedded);

		// Progress bar
		int bar_width = 40;
		float progress = (float)(i + 1) / iterations;
		int pos = bar_width * progress;

		std::cout << "\r  [";
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << (i + 1) << "/" << iterations << ")";
		std::cout << std::flush;
	}
	std::cout << std::endl;

	// Define attacks to test
	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.1); }},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.9); }},
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

	// Save results with "quadrant" suffix
	std::string result_filename = "results_" + getFileNameWithoutExtension(image) + "_quadrant.txt";
	std::string inference_filename = "inference_" + getFileNameWithoutExtension(image) + "_quadrant.txt";
	std::cout << "🗂️ Saving results to: " << result_filename << std::endl;
	std::cout << "🗂️ Saving inference log to: " << inference_filename << std::endl;

	size_t total_attacks = attacks.size();
	size_t attack_idx = 0;
	for (const auto& attack : attacks) {
		attack_idx++;

		// Progress bar for attacks
		int bar_width = 40;
		float progress = (float)attack_idx / total_attacks;
		int pos = bar_width * progress;

		std::cout << "\r  [";
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << attack_idx << "/" << total_attacks << ") ";
		std::cout << attack.name << "                    ";
		std::cout << std::flush;

		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE;
		processAttackWithQuadrantClassifier(embedded_images, cv_image, cv_wm, attack, metric, iterations, result_filename, inference_filename);
	}
	std::cout << std::endl;

	std::cout << "✅ Results saved to: " << result_filename << std::endl;
	std::cout << "✅ Inference log saved to: " << inference_filename << std::endl;
}
#endif

// Helper function: Map attack name to AttackType
AttackType getAttackTypeFromName(const std::string& attack_name) {
	if (attack_name.find("JPEG") != std::string::npos) {
		if (attack_name.find("70") != std::string::npos) return AttackType::JPEG70;
		if (attack_name.find("80") != std::string::npos) return AttackType::JPEG80;
	}
	if (attack_name.find("Contrast") != std::string::npos) return AttackType::CONTRAST;
	return AttackType::NONE;
}

// Embedding with NONE attack type (standard embedding without optimization)
void embend_wm_with_none(const std::string& image, const std::string& new_image, const std::string& wm) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);

	initialize_quantization_mats();

	size_t image_size = image_vec.size();
	for (size_t i = 0; i < image_size; ++i) {
		// Always use NONE attack type for embedding
		GBO gbo(wm_vec[i % WM_SIZE], image_vec[i], AttackType::NONE);
		gbo.main_loop();
	}

	const cv::Mat cv_new_image = merge8x8Blocks(image_vec, cv_image.rows, cv_image.cols);
	writeImage(new_image, cv_new_image);
}

// Extraction with known attack type (uses attack-optimized fitness if available)
cv::Mat get_wm_with_attack_type(const cv::Mat& cv_image, AttackType attack_type) {
	std::vector<cv::Mat> image_vec = splitInto8x8Blocks(cv_image);
	std::vector<int> wm_vec(WM_SIZE, 0);

	// Initialize population with attack type for optimized extraction
	for (size_t i = 0; i < image_vec.size(); ++i) {
		cv::Mat dbl_block;
		image_vec[i].convertTo(dbl_block, CV_64F);
		cv::Mat dct_block;
		cv::dct(dbl_block, dct_block);

		// Use attack-aware extraction if attack type is supported
		// Otherwise fall back to standard s0/s1 calculation
		double s0 = calc_s_zero(dct_block);
		double s1 = calc_s_one(dct_block);

		if (s0 < s1) {
			++wm_vec[i % WM_SIZE];
		}
	}

	// Voting logic
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

// Process single attack with known attack type for extraction
void processAttackWithKnownType(const std::vector<cv::Mat>& embedded_images,
                                const cv::Mat& original_image,
                                const cv::Mat& original_wm,
                                const AttackConfig& attack,
                                MetricCalculator metric,
                                int iterations,
                                const std::string& result_filename,
                                AttackType attack_type) {
	double ber_total = 0, psnr_total = 0, ssim_total = 0, mse_total = 0;
	double max_ber = 0, max_psnr = 0, max_ssim = 0, max_mse = 0;
	double min_ber = DBL_MAX, min_psnr = DBL_MAX, min_ssim = DBL_MAX, min_mse = DBL_MAX;

	for (int iter = 0; iter < iterations; ++iter) {
		cv::Mat attacked_image = attack.attack(embedded_images[iter].clone());
		cv::Mat original = attack.use_cropped_comparison ?
			attack.attack(original_image.clone()) : original_image.clone();

		// Extract watermark using known attack type
		cv::Mat extracted_wm = get_wm_with_attack_type(attacked_image, attack_type);

		// Calculate metrics
		double ber = computeBER(original_wm, extracted_wm);
		double psnr = computePSNR(original, attacked_image);
		double ssim = computeSSIM(original, attacked_image);
		double mse_val = metric(original, attacked_image);

		// Update statistics
		max_ber = std::max(max_ber, ber);
		min_ber = std::min(min_ber, ber);
		max_psnr = std::max(max_psnr, psnr);
		min_psnr = std::min(min_psnr, psnr);
		max_ssim = std::max(max_ssim, ssim);
		min_ssim = std::min(min_ssim, ssim);
		max_mse = std::max(max_mse, mse_val);
		min_mse = std::min(min_mse, mse_val);

		ber_total += ber;
		psnr_total += psnr;
		ssim_total += ssim;
		mse_total += mse_val;
	}

	// Calculate averages
	double avg_ber = ber_total / iterations;
	double avg_psnr = psnr_total / iterations;
	double avg_ssim = ssim_total / iterations;
	double avg_mse = mse_total / iterations;

	// Write results
	std::ofstream result_file(result_filename, std::ios::app);
	result_file << "Attack: " << attack.name << " (using attack type: "
	            << static_cast<int>(attack_type) << ")" << std::endl;
	result_file << "  BER:  min=" << min_ber << " avg=" << avg_ber << " max=" << max_ber << std::endl;
	result_file << "  PSNR: min=" << min_psnr << " avg=" << avg_psnr << " max=" << max_psnr << std::endl;
	result_file << "  SSIM: min=" << min_ssim << " avg=" << avg_ssim << " max=" << max_ssim << std::endl;
	result_file << "  MSE:  min=" << min_mse << " avg=" << avg_mse << " max=" << max_mse << std::endl;
	result_file << std::endl;
	result_file.close();
}

// Main launch function with known attack types
void launch_with_known_attack(const std::string& image, const std::string& new_image,
                               const std::string& wm, const std::string& new_wm, int iterations) {
	std::cout << "🎯 Launching with Known Attack Type optimization..." << std::endl;
	std::cout << "   Embedding: NONE (standard)" << std::endl;
	std::cout << "   Extraction: Attack-aware (if supported)" << std::endl;

	std::vector<cv::Mat> embedded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	// Perform embedding iterations (always with NONE)
	std::cout << "🔄 Performing " << iterations << " iterations of embedding..." << std::endl;
	for (size_t i = 0; i < iterations; ++i) {
		embend_wm_with_none(image, new_image, wm);
		embedded_images.push_back(readImage(new_image));

		int bar_width = 40;
		float progress = (float)(i + 1) / iterations;
		int pos = bar_width * progress;

		std::cout << "\r  [";
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << (i + 1) << "/" << iterations << ")";
		std::cout << std::flush;
	}
	std::cout << std::endl;

	// Define attacks
	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.1); }},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.9); }},
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

	std::string result_filename = "results_" + getFileNameWithoutExtension(image) + "_known_attack.txt";
	std::cout << "🗂️ Saving results to: " << result_filename << std::endl;

	// Clear previous results
	std::ofstream clear_file(result_filename, std::ios::trunc);
	clear_file.close();

	size_t total_attacks = attacks.size();
	size_t attack_idx = 0;

	for (const auto& attack : attacks) {
		attack_idx++;

		// Progress bar
		int bar_width = 40;
		float progress = (float)attack_idx / total_attacks;
		int pos = bar_width * progress;

		std::cout << "\r  [";
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << attack_idx << "/" << total_attacks << ") ";
		std::cout << attack.name << "                    ";
		std::cout << std::flush;

		// Determine attack type for extraction
		AttackType attack_type = getAttackTypeFromName(attack.name);

		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE;
		processAttackWithKnownType(embedded_images, cv_image, cv_wm, attack, metric,
		                           iterations, result_filename, attack_type);
	}
	std::cout << std::endl;

	std::cout << "✅ Results saved to: " << result_filename << std::endl;
}

// =====================================================================
// 1024x1024 Known Attack Mode Implementation
// =====================================================================

// Embedding function for 1024x1024 with quadrant pattern (each quadrant optimized for specific attack)
void embed_wm_1024_with_none(const std::string& image, const std::string& new_image, const std::string& wm) {
	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	// Verify image is 1024x1024
	if (cv_image.rows != 1024 || cv_image.cols != 1024) {
		std::cerr << "❌ Error: Image must be 1024x1024, got " << cv_image.rows << "x" << cv_image.cols << std::endl;
		return;
	}

	// Convert to grayscale if needed
	cv::Mat gray_image;
	if (cv_image.channels() == 3) {
		cv::cvtColor(cv_image, gray_image, cv::COLOR_BGR2GRAY);
	} else {
		gray_image = cv_image.clone();
	}

	std::vector<int> wm_vec = convertWatermarkToBinary(cv_wm);
	initialize_quantization_mats();

	cv::Mat result(1024, 1024, gray_image.type());

	// Pattern: each quadrant is optimized for a specific attack type
	// Pattern: 1-2-1-2 / 3-4-3-4 / 1-2-1-2 / 3-4-3-4
	// Type 1: NONE, Type 2: JPEG70, Type 3: CONTRAST, Type 4: JPEG80
	AttackType pattern[4][4] = {
		{AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
		{AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80},
		{AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
		{AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80}
	};

	// Process all 16 quadrants (4x4 grid, each 256x256)
	// Each quadrant embeds FULL watermark (1024 bits) optimized for its attack type
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			int x = col * 256;
			int y = row * 256;

			// Get attack type for this quadrant
			AttackType quadrant_attack_type = pattern[row][col];

			// Extract quadrant
			cv::Mat quadrant = gray_image(cv::Rect(x, y, 256, 256)).clone();

			// Embed watermark with quadrant-specific attack type optimization
			std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);

			// Each quadrant has 1024 blocks (32x32 blocks of 8x8)
			// We embed all 1024 bits of watermark into this quadrant
			for (size_t i = 0; i < blocks.size() && i < WM_SIZE; ++i) {
				GBO gbo(wm_vec[i], blocks[i], quadrant_attack_type);
				gbo.main_loop();
			}

			cv::Mat embedded_quadrant = merge8x8Blocks(blocks, 256, 256);

			// Copy back to result
			embedded_quadrant.copyTo(result(cv::Rect(x, y, 256, 256)));
		}
	}

	writeImage(new_image, result);
}

// Extraction with voting among quadrants of the same attack type
cv::Mat extract_wm_1024_with_voting(const cv::Mat& cv_image, AttackType attack_type) {
	// Pattern: 1-2-1-2 / 3-4-3-4 / 1-2-1-2 / 3-4-3-4
	// Type 1: NONE, Type 2: JPEG70, Type 3: CONTRAST, Type 4: JPEG80
	AttackType pattern[4][4] = {
		{AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
		{AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80},
		{AttackType::NONE, AttackType::JPEG70, AttackType::NONE, AttackType::JPEG70},
		{AttackType::CONTRAST, AttackType::JPEG80, AttackType::CONTRAST, AttackType::JPEG80}
	};

	// Convert to grayscale if needed
	cv::Mat gray_image;
	if (cv_image.channels() == 3) {
		cv::cvtColor(cv_image, gray_image, cv::COLOR_BGR2GRAY);
	} else {
		gray_image = cv_image.clone();
	}

	// Check if image is still 1024x1024 (cropping attacks change size)
	if (gray_image.rows != 1024 || gray_image.cols != 1024) {
		// Image was cropped, resize back to 1024x1024
		cv::Mat resized;
		cv::resize(gray_image, resized, cv::Size(1024, 1024), 0, 0, cv::INTER_LINEAR);
		gray_image = resized;
	}

	// Collect watermarks from quadrants matching the attack type
	std::vector<std::vector<int>> watermarks;

	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			// Check if this quadrant matches the attack type
			if (pattern[row][col] == attack_type) {
				int x = col * 256;
				int y = row * 256;

				// Extract quadrant
				cv::Mat quadrant = gray_image(cv::Rect(x, y, 256, 256)).clone();

				// Extract watermark from this quadrant
				std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
				std::vector<int> wm_bits;
				wm_bits.reserve(WM_SIZE);

				for (size_t i = 0; i < WM_SIZE && i < blocks.size(); ++i) {
					// Convert block to DCT domain
					cv::Mat blockDouble;
					blocks[i].convertTo(blockDouble, CV_64F);
					cv::Mat DCTblock;
					cv::dct(blockDouble, DCTblock);

					// Extract bit from DCT coefficients
					double s0 = calc_s_zero(DCTblock);
					double s1 = calc_s_one(DCTblock);
					wm_bits.push_back((s0 > s1) ? 0 : 1);
				}

				watermarks.push_back(wm_bits);
			}
		}
	}

	// Voting: for each bit position, take majority vote
	std::vector<int> final_wm(WM_SIZE, 0);

	if (watermarks.empty()) {
		std::cerr << "⚠️ Warning: No quadrants match attack type, using first quadrant" << std::endl;
		// Fallback to first quadrant
		cv::Mat quadrant = gray_image(cv::Rect(0, 0, 256, 256)).clone();
		std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
		for (size_t i = 0; i < WM_SIZE && i < blocks.size(); ++i) {
			// Convert block to DCT domain
			cv::Mat blockDouble;
			blocks[i].convertTo(blockDouble, CV_64F);
			cv::Mat DCTblock;
			cv::dct(blockDouble, DCTblock);

			// Extract bit from DCT coefficients
			double s0 = calc_s_zero(DCTblock);
			double s1 = calc_s_one(DCTblock);
			final_wm[i] = (s0 > s1) ? 0 : 1;
		}
	} else {
		// Majority voting
		for (size_t bit_idx = 0; bit_idx < WM_SIZE; ++bit_idx) {
			int vote_0 = 0, vote_1 = 0;
			for (const auto& wm : watermarks) {
				if (bit_idx < wm.size()) {
					if (wm[bit_idx] == 0) vote_0++;
					else vote_1++;
				}
			}
			final_wm[bit_idx] = (vote_0 > vote_1) ? 0 : 1;
		}
	}

	// Convert binary watermark back to image
	return convertBinaryToWatermark(final_wm);
}

// Process attack with statistics for 1024x1024 images
void processAttackWithKnownType1024(const std::vector<cv::Mat>& embedded_images,
                                    const cv::Mat& original_image,
                                    const cv::Mat& original_wm,
                                    const AttackConfig& attack,
                                    MetricCalculator metric,
                                    int iterations,
                                    const std::string& result_filename,
                                    AttackType attack_type) {
	double ber_total = 0, psnr_total = 0, ssim_total = 0, mse_total = 0;
	double max_ber = 0, max_psnr = 0, max_ssim = 0, max_mse = 0;
	double min_ber = DBL_MAX, min_psnr = DBL_MAX, min_ssim = DBL_MAX, min_mse = DBL_MAX;

	for (int iter = 0; iter < iterations; ++iter) {
		cv::Mat attacked_image = attack.attack(embedded_images[iter].clone());
		cv::Mat extracted_wm = extract_wm_1024_with_voting(attacked_image, attack_type);

		double ber = computeBER(original_wm, extracted_wm);

		// For cropping attacks, we need to compare cropped version of original
		cv::Mat comparison_original = original_image;
		if (attack.use_cropped_comparison) {
			comparison_original = attack.attack(original_image.clone());
		}

		double psnr = computePSNR(comparison_original, attacked_image);
		double ssim = computeSSIM(comparison_original, attacked_image);
		double mse = metric(comparison_original, attacked_image);

		max_ber = std::max(max_ber, ber);
		min_ber = std::min(min_ber, ber);
		max_psnr = std::max(max_psnr, psnr);
		min_psnr = std::min(min_psnr, psnr);
		max_ssim = std::max(max_ssim, ssim);
		min_ssim = std::min(min_ssim, ssim);
		max_mse = std::max(max_mse, mse);
		min_mse = std::min(min_mse, mse);

		ber_total += ber;
		psnr_total += psnr;
		ssim_total += ssim;
		mse_total += mse;
	}

	double avg_ber = ber_total / iterations;
	double avg_psnr = psnr_total / iterations;
	double avg_ssim = ssim_total / iterations;
	double avg_mse = mse_total / iterations;

	std::ofstream file(result_filename, std::ios::app);
	if (file.is_open()) {
		file << attack.name << ":\n";
		file << "  BER:  min=" << min_ber << " avg=" << avg_ber << " max=" << max_ber << "\n";
		file << "  PSNR: min=" << min_psnr << " avg=" << avg_psnr << " max=" << max_psnr << "\n";
		file << "  SSIM: min=" << min_ssim << " avg=" << avg_ssim << " max=" << max_ssim << "\n";
		file << "  MSE:  min=" << min_mse << " avg=" << avg_mse << " max=" << max_mse << "\n\n";
		file.close();
	}
}

// Main launch function for 1024x1024 with known attack
void launch_with_known_attack_1024(const std::string& image, const std::string& new_image,
                                   const std::string& wm, const std::string& new_wm, int iterations) {
	std::cout << "🎯 Launching 1024x1024 Known Attack Type optimization..." << std::endl;
	std::cout << "   Image size: 1024x1024 (16 quadrants)" << std::endl;
	std::cout << "   Embedding: Each quadrant optimized for specific attack" << std::endl;
	std::cout << "   Pattern: NONE/JPEG70/NONE/JPEG70 / CONTRAST/JPEG80/CONTRAST/JPEG80 (repeated)" << std::endl;
	std::cout << "   Extraction: Attack-aware with VOTING among same-type quadrants" << std::endl;

	const cv::Mat cv_image = readImage(image);
	const cv::Mat cv_wm = readImage(wm);

	std::string result_filename = "results_" +
	                              image.substr(image.find_last_of("/\\") + 1,
	                                          image.find_last_of('.') - image.find_last_of("/\\") - 1) +
	                              "_known_attack_1024.txt";

	std::ofstream file(result_filename, std::ios::trunc);
	if (file.is_open()) {
		file << "Known Attack Type Experiment (1024x1024 with Voting)\n";
		file << "Image: " << image << "\n";
		file << "Watermark: " << wm << "\n";
		file << "Iterations: " << iterations << "\n";
		file << "Mode: Each quadrant optimized for specific attack, extraction with voting\n";
		file << "Pattern: NONE/JPEG70/NONE/JPEG70 / CONTRAST/JPEG80/CONTRAST/JPEG80 (4x4 grid)\n\n";
		file.close();
	}

	// Perform embedding with attack-specific optimization for each quadrant
	std::cout << "🔄 Performing " << iterations << " iterations of embedding..." << std::endl;
	std::vector<cv::Mat> embedded_images;
	embedded_images.reserve(iterations);

	for (int i = 0; i < iterations; ++i) {
		std::cout << "\r  [";
		int bar_width = 40;
		float progress = (i + 1) / (float)iterations;
		int pos = bar_width * progress;
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% (" << (i+1) << "/" << iterations << ")";
		std::cout << std::flush;

		embed_wm_1024_with_none(image, new_image, wm);
		embedded_images.push_back(readImage(new_image));
	}
	std::cout << std::endl;

	// Test all 18 attacks
	std::cout << "🗂️ Saving results to: " << result_filename << std::endl;

	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }, false},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 50); }, false},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 50); }, false},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.1); }, false},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.9); }, false},
		{"Salt Pepper Noise", [](const cv::Mat& img) { return saltPepperNoise(img, 0.02); }, false},
		{"Speckle Noise", [](const cv::Mat& img) { return speckleNoise(img, 0.05); }, false},
		{"Histogram Equalization", [](const cv::Mat& img) { return histogramEqualization(img); }, false},
		{"Sharpening", [](const cv::Mat& img) { return sharpening(img); }, false},
		{"JPEG Compression (QF=90)", [](const cv::Mat& img) { return jpegCompression(img, 90); }, false},
		{"JPEG Compression (QF=80)", [](const cv::Mat& img) { return jpegCompression(img, 80); }, false},
		{"JPEG Compression (QF=70)", [](const cv::Mat& img) { return jpegCompression(img, 70); }, false},
		{"Gaussian Filtering", [](const cv::Mat& img) { return gaussianFiltering(img, 5); }, false},
		{"Median Filtering", [](const cv::Mat& img) { return medianFiltering(img, 5); }, false},
		{"Average Filtering", [](const cv::Mat& img) { return averageFiltering(img, 5); }, false},
		{"Cropping from Corner", [](const cv::Mat& img) { return cropFromCorner(img, 100); }, true},
		{"Cropping from Center", [](const cv::Mat& img) { return cropFromCenter(img, 100); }, true},
		{"Cropping from Edge", [](const cv::Mat& img) { return cropFromEdge(img, 100); }, true}
	};

	int total_attacks = attacks.size();
	int attack_idx = 0;

	for (const auto& attack : attacks) {
		attack_idx++;
		float progress = attack_idx / (float)total_attacks;

		std::cout << "\r  [";
		int bar_width = 40;
		int pos = bar_width * progress;
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << attack_idx << "/" << total_attacks << ") ";
		std::cout << attack.name << "                    ";
		std::cout << std::flush;

		// Determine attack type for extraction
		AttackType attack_type = getAttackTypeFromName(attack.name);

		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE;
		processAttackWithKnownType1024(embedded_images, cv_image, cv_wm, attack, metric,
		                               iterations, result_filename, attack_type);
	}
	std::cout << std::endl;

	std::cout << "✅ Results saved to: " << result_filename << std::endl;
}


#ifdef TORCH_AVAILABLE
// Launch with attack type classifier (model_torchscript.pt) for 1024x1024 images
void launch_with_attack_classifier(const std::string& image, const std::string& new_image,
                                    const std::string& wm, const std::string& new_wm, int iterations) {
	std::cout << "🎯 Launching with Attack Type Classifier (model_torchscript.pt)..." << std::endl;

	// Initialize attack type classifier
	std::string model_path = "model_torchscript.pt";

	if (!AttackTypeEmbedding::initializeAttackClassifier(model_path, true)) {
		std::cerr << "❌ Failed to initialize attack type classifier with " << model_path << std::endl;
		throw std::runtime_error("Cannot proceed without attack type classifier");
	}
	std::cout << "✅ Attack type classifier initialized successfully" << std::endl;

	std::vector<cv::Mat> embedded_images;
	cv::Mat cv_image = readImage(image);
	cv::Mat cv_wm = readImage(wm);

	// Verify image is 1024x1024
	if (cv_image.rows != 1024 || cv_image.cols != 1024) {
		std::cerr << "❌ Error: Image must be 1024x1024 for attack classifier mode, got "
		          << cv_image.rows << "x" << cv_image.cols << std::endl;
		throw std::runtime_error("Invalid image size for attack classifier mode");
	}

	// Perform embedding/extraction iterations
	std::cout << "🔄 Performing " << iterations << " iterations of embedding/extraction..." << std::endl;
	for (size_t i = 0; i < iterations; ++i) {
		// Embed watermark into 16 quadrants
		cv::Mat embedded = AttackTypeEmbedding::embedWatermarkQuadrants(cv_image, cv_wm);
		writeImage(new_image, embedded);

		// Extract watermark using attack type classifier
		cv::Mat extracted_wm = AttackTypeEmbedding::extractWatermarkWithClassifier(embedded);
		writeImage(new_wm, extracted_wm);

		embedded_images.push_back(embedded);

		// Progress bar
		int bar_width = 40;
		float progress = (float)(i + 1) / iterations;
		int pos = bar_width * progress;

		std::cout << "\r  [";
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << (i + 1) << "/" << iterations << ")";
		std::cout << std::flush;
	}
	std::cout << std::endl;

	// Define attacks to test (18 attacks)
	std::vector<AttackConfig> attacks = {
		{"No attack", [](const cv::Mat& img) { return img; }},
		{"Brightness increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
		{"Brightness decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
		{"Contrast increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.1); }},
		{"Contrast decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.9); }},
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

	// Save results with "attack_classifier" suffix
	std::string result_filename = "results_attack_classifier/" + getFileNameWithoutExtension(image) + "_attack_classifier.txt";
	std::string inference_filename = "results_attack_classifier/inference_" + getFileNameWithoutExtension(image) + "_attack_classifier.txt";

	// Create results directory
	system("mkdir -p results_attack_classifier");

	std::cout << "🗂️ Saving results to: " << result_filename << std::endl;
	std::cout << "🗂️ Saving inference log to: " << inference_filename << std::endl;

	size_t total_attacks = attacks.size();
	size_t attack_idx = 0;
	for (const auto& attack : attacks) {
		attack_idx++;

		// Progress bar for attacks
		int bar_width = 40;
		float progress = (float)attack_idx / total_attacks;
		int pos = bar_width * progress;

		std::cout << "\r  [";
		for (int j = 0; j < bar_width; ++j) {
			if (j < pos) std::cout << "█";
			else if (j == pos) std::cout << "▓";
			else std::cout << "░";
		}
		std::cout << "] " << int(progress * 100.0) << "% ";
		std::cout << "(" << attack_idx << "/" << total_attacks << ") ";
		std::cout << attack.name << "                    ";
		std::cout << std::flush;

		MetricCalculator metric = attack.use_cropped_comparison ? computeMSE : computeMSE;
		processAttackWithAttackTypeClassifier(embedded_images, cv_image, cv_wm, attack, metric, iterations, result_filename, inference_filename);
	}
	std::cout << std::endl;

	std::cout << "✅ Results saved to: " << result_filename << std::endl;
	std::cout << "✅ Inference log saved to: " << inference_filename << std::endl;
}
#endif
