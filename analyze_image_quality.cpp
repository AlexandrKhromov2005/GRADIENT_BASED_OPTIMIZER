#include "GRADIENT_BASED_OPTIMIZER/src/image_metrics.h"
#include "GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>

// Функция для получения списка PNG файлов в директории
std::vector<std::string> getPngFiles(const std::string& directory) {
    std::vector<std::string> files;
    DIR* dir = opendir(directory.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.find(".png") != std::string::npos) {
                files.push_back(directory + "/" + filename);
            }
        }
        closedir(dir);
    }
    return files;
}

// Анализ качества изображений после встраивания
void analyzeImageQuality() {
    std::cout << "🔍 Анализ качества изображений после встраивания водяных знаков" << std::endl;
    std::cout << "=============================================================" << std::endl;
    
    // Статистики
    double total_psnr = 0, total_ssim = 0, total_mse = 0, total_ncc = 0;
    double total_ber = 0; // BER для водяных знаков
    int processed_images = 0;
    
    double min_psnr = DBL_MAX, max_psnr = 0;
    double min_ssim = DBL_MAX, max_ssim = 0;
    double min_ber = DBL_MAX, max_ber = 0;
    
    std::ofstream report("image_quality_report.txt");
    report << "Анализ качества изображений с классификатором final_model.pt" << std::endl;
    report << "=============================================================" << std::endl;
    report << std::endl;
    
    // Анализируем оригинальные и обработанные изображения
    std::vector<std::string> original_images = getPngFiles("images");
    
    for (const std::string& orig_path : original_images) {
        // Пропускаем watermark.png
        if (orig_path.find("watermark") != std::string::npos) continue;
        
        std::string filename = orig_path.substr(orig_path.find_last_of('/') + 1);
        std::string stem = filename.substr(0, filename.find_last_of('.'));
        std::string embedded_path = "images/new_" + stem + ".png";
        std::string extracted_wm_path = "images/" + stem + "_wm.png";
        
        // Проверяем существование обработанных файлов
        cv::Mat original = cv::imread(orig_path, cv::IMREAD_GRAYSCALE);
        cv::Mat embedded = cv::imread(embedded_path, cv::IMREAD_GRAYSCALE);
        cv::Mat original_wm = cv::imread("images/watermark.png", cv::IMREAD_GRAYSCALE);
        cv::Mat extracted_wm = cv::imread(extracted_wm_path, cv::IMREAD_GRAYSCALE);
        
        if (original.empty() || embedded.empty()) {
            std::cout << "⚠️  Пропущено: " << stem << " (файлы не найдены)" << std::endl;
            continue;
        }
        
        // Вычисляем метрики качества изображения
        double mse = computeMSE(original, embedded);
        double psnr = computePSNR(original, embedded);
        double ssim = computeSSIM(original, embedded);
        double ncc = computeNCC(original, embedded);
        
        // Вычисляем BER для водяного знака (если есть извлеченный)
        double ber = 0;
        if (!original_wm.empty() && !extracted_wm.empty()) {
            ber = computeBER(original_wm, extracted_wm);
        }
        
        // Обновляем статистики
        total_psnr += psnr;
        total_ssim += ssim;
        total_mse += mse;
        total_ncc += ncc;
        total_ber += ber;
        processed_images++;
        
        min_psnr = std::min(min_psnr, psnr);
        max_psnr = std::max(max_psnr, psnr);
        min_ssim = std::min(min_ssim, ssim);
        max_ssim = std::max(max_ssim, ssim);
        min_ber = std::min(min_ber, ber);
        max_ber = std::max(max_ber, ber);
        
        // Выводим результаты для каждого изображения
        std::cout << "📊 " << stem << ":" << std::endl;
        std::cout << "   PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
        std::cout << "   SSIM: " << std::fixed << std::setprecision(4) << ssim << std::endl;
        std::cout << "   MSE:  " << std::fixed << std::setprecision(2) << mse << std::endl;
        std::cout << "   NCC:  " << std::fixed << std::setprecision(4) << ncc << std::endl;
        if (ber > 0) {
            std::cout << "   BER:  " << std::fixed << std::setprecision(4) << ber << std::endl;
        }
        std::cout << std::endl;
        
        // Записываем в отчет
        report << "Изображение: " << stem << std::endl;
        report << "  PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
        report << "  SSIM: " << std::fixed << std::setprecision(4) << ssim << std::endl;
        report << "  MSE:  " << std::fixed << std::setprecision(2) << mse << std::endl;
        report << "  NCC:  " << std::fixed << std::setprecision(4) << ncc << std::endl;
        if (ber > 0) {
            report << "  BER:  " << std::fixed << std::setprecision(4) << ber << std::endl;
        }
        report << std::endl;
    }
    
    if (processed_images > 0) {
        // Средние значения
        double avg_psnr = total_psnr / processed_images;
        double avg_ssim = total_ssim / processed_images;
        double avg_mse = total_mse / processed_images;
        double avg_ncc = total_ncc / processed_images;
        double avg_ber = total_ber / processed_images;
        
        std::cout << "🎯 ИТОГОВАЯ СТАТИСТИКА (" << processed_images << " изображений):" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "📈 PSNR:  мин=" << std::fixed << std::setprecision(2) << min_psnr 
                  << " dB, среднее=" << avg_psnr << " dB, макс=" << max_psnr << " dB" << std::endl;
        std::cout << "📈 SSIM:  мин=" << std::fixed << std::setprecision(4) << min_ssim 
                  << ", среднее=" << avg_ssim << ", макс=" << max_ssim << std::endl;
        std::cout << "📈 MSE:   среднее=" << std::fixed << std::setprecision(2) << avg_mse << std::endl;
        std::cout << "📈 NCC:   среднее=" << std::fixed << std::setprecision(4) << avg_ncc << std::endl;
        if (avg_ber > 0) {
            std::cout << "📈 BER:   мин=" << std::fixed << std::setprecision(4) << min_ber 
                      << ", среднее=" << avg_ber << ", макс=" << max_ber << std::endl;
        }
        
        // Записываем итоговую статистику в отчет
        report << std::endl << "ИТОГОВАЯ СТАТИСТИКА (" << processed_images << " изображений):" << std::endl;
        report << "=============================================" << std::endl;
        report << "PSNR:  мин=" << std::fixed << std::setprecision(2) << min_psnr 
               << " dB, среднее=" << avg_psnr << " dB, макс=" << max_psnr << " dB" << std::endl;
        report << "SSIM:  мин=" << std::fixed << std::setprecision(4) << min_ssim 
               << ", среднее=" << avg_ssim << ", макс=" << max_ssim << std::endl;
        report << "MSE:   среднее=" << std::fixed << std::setprecision(2) << avg_mse << std::endl;
        report << "NCC:   среднее=" << std::fixed << std::setprecision(4) << avg_ncc << std::endl;
        if (avg_ber > 0) {
            report << "BER:   мин=" << std::fixed << std::setprecision(4) << min_ber 
                   << ", среднее=" << avg_ber << ", макс=" << max_ber << std::endl;
        }
        
        // Качественная оценка
        std::cout << std::endl << "🏆 КАЧЕСТВЕННАЯ ОЦЕНКА:" << std::endl;
        if (avg_psnr >= 40) {
            std::cout << "✅ PSNR: Отличное качество (≥40 dB)" << std::endl;
        } else if (avg_psnr >= 30) {
            std::cout << "🟡 PSNR: Хорошее качество (30-40 dB)" << std::endl;
        } else {
            std::cout << "🔴 PSNR: Низкое качество (<30 dB)" << std::endl;
        }
        
        if (avg_ssim >= 0.9) {
            std::cout << "✅ SSIM: Отличное структурное сходство (≥0.9)" << std::endl;
        } else if (avg_ssim >= 0.7) {
            std::cout << "🟡 SSIM: Хорошее структурное сходство (0.7-0.9)" << std::endl;
        } else {
            std::cout << "🔴 SSIM: Низкое структурное сходство (<0.7)" << std::endl;
        }
        
        if (avg_ber <= 0.05) {
            std::cout << "✅ BER: Отличная точность извлечения (≤5%)" << std::endl;
        } else if (avg_ber <= 0.15) {
            std::cout << "🟡 BER: Хорошая точность извлечения (5-15%)" << std::endl;
        } else {
            std::cout << "🔴 BER: Низкая точность извлечения (>15%)" << std::endl;
        }
        
        report << std::endl << "КАЧЕСТВЕННАЯ ОЦЕНКА:" << std::endl;
        report << "PSNR: " << (avg_psnr >= 40 ? "Отличное" : (avg_psnr >= 30 ? "Хорошее" : "Низкое")) << " качество" << std::endl;
        report << "SSIM: " << (avg_ssim >= 0.9 ? "Отличное" : (avg_ssim >= 0.7 ? "Хорошее" : "Низкое")) << " структурное сходство" << std::endl;
        report << "BER: " << (avg_ber <= 0.05 ? "Отличная" : (avg_ber <= 0.15 ? "Хорошая" : "Низкая")) << " точность извлечения" << std::endl;
        
    } else {
        std::cout << "❌ Не найдено обработанных изображений для анализа" << std::endl;
        std::cout << "💡 Запустите сначала эксперимент: ./build/gradient_based_optimizer --classifier" << std::endl;
    }
    
    report.close();
    std::cout << std::endl << "📝 Детальный отчет сохранен в: image_quality_report.txt" << std::endl;
}

int main() {
    analyzeImageQuality();
    return 0;
}