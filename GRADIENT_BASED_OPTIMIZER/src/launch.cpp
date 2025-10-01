#include "launch.h"

#ifdef TORCH_AVAILABLE
#include "embedding_with_classifier.h"
#include "ensemble_classifier.h"
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



