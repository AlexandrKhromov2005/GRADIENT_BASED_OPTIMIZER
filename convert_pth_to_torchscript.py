#!/usr/bin/env python3
"""
Утилита для конвертации PyTorch .pth модели в TorchScript .pt формат
для использования в C++ интеграции
"""

import torch
import torch.nn as nn
import argparse
import os

class SimpleCNN(nn.Module):
    """Простая CNN архитектура для классификации схем встраивания"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # После 3 pooling операций: 192/8 = 24
        self.fc1 = nn.Linear(128 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.conv1(x)))  # 96x96
        x = self.pool(self.relu(self.conv2(x)))  # 48x48  
        x = self.pool(self.relu(self.conv3(x)))  # 24x24
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def convert_pth_to_torchscript(pth_path, output_path, model_class=SimpleCNN, input_size=(1, 1, 192, 192)):
    """
    Конвертирует .pth модель в TorchScript .pt формат
    
    Args:
        pth_path: путь к .pth файлу
        output_path: путь для сохранения .pt файла
        model_class: класс модели
        input_size: размер входного тензора (batch, channels, height, width)
    """
    
    print(f"🔄 Loading model from {pth_path}")
    
    # Создаем экземпляр модели
    model = model_class(num_classes=2)
    
    # Загружаем веса
    try:
        # Пробуем загрузить как полный checkpoint
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            # Если это dict, ищем state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded from model_state_dict")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("✅ Loaded from state_dict")
            else:
                # Возможно сам checkpoint и есть state_dict
                model.load_state_dict(checkpoint)
                print("✅ Loaded checkpoint as state_dict")
        else:
            # Если это модель целиком
            model = checkpoint
            print("✅ Loaded complete model")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
        # Попробуем alternative loading
        try:
            state_dict = torch.load(pth_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ Loaded with alternative method")
        except Exception as e2:
            print(f"❌ Alternative loading failed: {e2}")
            return False
    
    # Переводим модель в eval режим
    model.eval()
    
    # Создаем пример входных данных
    example_input = torch.randn(input_size)
    
    print(f"📝 Creating TorchScript with input size: {input_size}")
    
    # Трассируем модель
    try:
        traced_model = torch.jit.trace(model, example_input)
        print("✅ Model traced successfully")
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        
        # Попробуем script mode
        try:
            traced_model = torch.jit.script(model)
            print("✅ Model scripted successfully")
        except Exception as e2:
            print(f"❌ Scripting also failed: {e2}")
            return False
    
    # Сохраняем TorchScript модель
    try:
        traced_model.save(output_path)
        print(f"💾 TorchScript model saved to: {output_path}")
        
        # Проверяем загрузку
        loaded_model = torch.jit.load(output_path)
        print("✅ Verification: TorchScript model loads correctly")
        
        # Тестируем предсказание
        with torch.no_grad():
            test_input = torch.randn(1, 1, 192, 192)
            original_output = model(test_input)
            torchscript_output = loaded_model(test_input)
            
            diff = torch.abs(original_output - torchscript_output).max().item()
            print(f"📊 Max difference between models: {diff}")
            
            if diff < 1e-6:
                print("✅ Models produce identical outputs")
            else:
                print("⚠️ Models have small differences (this might be acceptable)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save TorchScript model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch .pth model to TorchScript .pt format')
    parser.add_argument('input', help='Input .pth file path')
    parser.add_argument('output', help='Output .pt file path')
    parser.add_argument('--input-size', nargs=4, type=int, default=[1, 1, 192, 192],
                        help='Input tensor size: batch channels height width (default: 1 1 192 192)')
    
    args = parser.parse_args()
    
    # Проверяем существование входного файла
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return 1
    
    # Создаем выходную директорию если нужно
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"🚀 Converting {args.input} -> {args.output}")
    print(f"📏 Input size: {args.input_size}")
    
    success = convert_pth_to_torchscript(
        args.input, 
        args.output, 
        input_size=tuple(args.input_size)
    )
    
    if success:
        print("🎉 Conversion completed successfully!")
        return 0
    else:
        print("❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    exit(main())