import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from models.srcnn import SRCNN
from utils.metrics import calculate_metrics, save_image

def load_model(model_path, device):
    """加载训练好的模型"""
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, device):
    """预处理输入图像"""
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    
    # 转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 添加批次维度
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img.size

def inference(model_path, input_image_path, output_dir):
    """使用模型进行超分辨率重建"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 预处理输入图像
    input_tensor, original_size = preprocess_image(input_image_path, device)
    
    # 进行推理
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 保存结果
    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + '_sr.png'
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存超分辨率图像
    save_image(output_tensor[0], output_path)
    print(f'超分辨率重建结果已保存到: {output_path}')
    
    # 显示对比图
    input_img = cv2.imread(input_image_path)
    output_img = cv2.imread(output_path)
    
    # 调整输入图像大小以匹配输出
    input_img = cv2.resize(input_img, (output_img.shape[1], output_img.shape[0]))
    
    # 水平拼接图像
    comparison = np.hstack((input_img, output_img))
    
    # 保存对比图
    comparison_path = os.path.join(output_dir, f'comparison_{os.path.splitext(input_filename)[0]}.png')
    cv2.imwrite(comparison_path, comparison)
    print(f'对比图已保存到: {comparison_path}')

if __name__ == '__main__':
    # 设置路径
    # model_path = r"D:\resp\SRCNN\src\checkpoints\20250613_195443\best_model.pth"
    # input_image_path = r"C:\Users\kAVOY\Pictures\v2-862ff88bb9219695067965da144e9dbc_1440w.jpg"  # 输入图像路径
    # output_dir = r"D:\resp\SRCNN\.idea\TESToutput"  # 输出目录
    
    # 进行推理
    inference(model_path, input_image_path, output_dir) 