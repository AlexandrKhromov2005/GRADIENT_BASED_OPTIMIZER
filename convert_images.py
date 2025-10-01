#!/usr/bin/env python3
import os
import subprocess
import sys

def convert_image_via_opencv_cpp():
    """Используем саму программу для преобразования через OpenCV"""
    
    # Создаем временную C++ программу для конвертации
    cpp_code = '''
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return -1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    // Читаем изображение с любым форматом
    cv::Mat image = cv::imread(input_path, cv::IMREAD_UNCHANGED);
    
    if (image.empty()) {
        std::cout << "Error: Could not load " << input_path << std::endl;
        return -1;
    }
    
    // Сохраняем как PNG
    bool success = cv::imwrite(output_path, image);
    
    if (success) {
        std::cout << "Converted: " << input_path << " -> " << output_path << std::endl;
        return 0;
    } else {
        std::cout << "Error: Could not save " << output_path << std::endl;
        return -1;
    }
}
'''
    
    # Сохраняем код в файл
    with open('temp_converter.cpp', 'w') as f:
        f.write(cpp_code)
    
    # Компилируем с правильными флагами OpenCV
    compile_cmd = [
        'g++', '-std=c++17', 
        '-I/usr/include/opencv4',
        'temp_converter.cpp', 
        '-o', 'temp_converter',
        '-lopencv_core', '-lopencv_imgproc', '-lopencv_imgcodecs'
    ]
    
    try:
        subprocess.run(compile_cmd, check=True, capture_output=True)
        print("✅ Конвертер скомпилирован")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка компиляции: {e}")
        print(f"stdout: {e.stdout.decode()}")
        print(f"stderr: {e.stderr.decode()}")
        return False

def main():
    os.chdir('/home/alex/projects/GRADIENT_BASED_OPTIMIZER')
    
    if not convert_image_via_opencv_cpp():
        print("Не удалось создать конвертер")
        return
    
    images_dir = 'images'
    converted_count = 0
    
    for filename in os.listdir(images_dir):
        if filename.endswith('.png'):
            filepath = os.path.join(images_dir, filename)
            
            # Проверяем формат
            result = subprocess.run(['file', filepath], capture_output=True, text=True)
            
            if 'GIF' in result.stdout or 'TIFF' in result.stdout:
                print(f"🔄 Конвертирую {filename}...")
                
                temp_path = os.path.join(images_dir, f'temp_{filename}')
                
                # Конвертируем
                convert_result = subprocess.run(
                    ['./temp_converter', filepath, temp_path], 
                    capture_output=True, text=True
                )
                
                if convert_result.returncode == 0:
                    # Заменяем оригинальный файл
                    os.remove(filepath)
                    os.rename(temp_path, filepath)
                    converted_count += 1
                    print(f"✅ {filename} конвертирован")
                else:
                    print(f"❌ Ошибка конвертации {filename}: {convert_result.stderr}")
    
    print(f"\n🎉 Конвертировано файлов: {converted_count}")
    
    # Очищаем временные файлы
    if os.path.exists('temp_converter'):
        os.remove('temp_converter')
    if os.path.exists('temp_converter.cpp'):
        os.remove('temp_converter.cpp')

if __name__ == '__main__':
    main()