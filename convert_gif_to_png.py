#!/usr/bin/env python3
import os
from PIL import Image
import subprocess

def convert_gif_to_png():
    os.chdir('/home/alex/projects/GRADIENT_BASED_OPTIMIZER/images')
    
    converted_count = 0
    total_gif_files = 0
    
    for filename in os.listdir('.'):
        if filename.endswith('.png'):
            # Проверяем реальный формат
            result = subprocess.run(['file', filename], capture_output=True, text=True)
            
            if 'GIF' in result.stdout:
                total_gif_files += 1
                print(f"🔄 Конвертирую {filename}...")
                
                try:
                    # Открываем GIF с помощью PIL
                    with Image.open(filename) as img:
                        # Конвертируем в RGB если нужно
                        if img.mode in ('RGBA', 'LA', 'P'):
                            # Для изображений с прозрачностью создаем белый фон
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Сохраняем как PNG
                        temp_filename = f"temp_{filename}"
                        img.save(temp_filename, 'PNG')
                        
                        # Заменяем оригинальный файл
                        os.remove(filename)
                        os.rename(temp_filename, filename)
                        
                        converted_count += 1
                        print(f"✅ {filename} успешно конвертирован")
                        
                except Exception as e:
                    print(f"❌ Ошибка конвертации {filename}: {e}")
    
    print(f"\n🎉 Результат конвертации:")
    print(f"   Найдено GIF файлов: {total_gif_files}")
    print(f"   Успешно конвертировано: {converted_count}")
    
    # Проверяем финальное состояние
    png_count = 0
    for filename in os.listdir('.'):
        if filename.endswith('.png'):
            result = subprocess.run(['file', filename], capture_output=True, text=True)
            if 'PNG' in result.stdout:
                png_count += 1
    
    print(f"   Корректных PNG файлов сейчас: {png_count}")

if __name__ == '__main__':
    convert_gif_to_png()