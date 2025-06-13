import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error
import cv2

def calculate_psnr(img1, img2):
    """计算PSNR"""
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    # 确保数据范围在[0,1]之间
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """计算SSIM"""
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    
    # 确保数据范围在[0,1]之间
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # 检查图像尺寸
    if img1.shape[1] < 3 or img1.shape[2] < 3:
        return 0.0  # 如果图像太小，返回0
    
    # 使用较小的窗口大小，确保小于图像尺寸
    win_size = min(7, min(img1.shape[1], img1.shape[2]) - 1)
    if win_size % 2 == 0:
        win_size -= 1  # 确保窗口大小为奇数
    if win_size < 3:  # 如果窗口太小，使用3x3
        win_size = 3
    
    try:
        # 对每个通道分别计算SSIM
        ssim_values = []
        for i in range(img1.shape[0]):
            ssim_value = ssim(img1[i], img2[i], data_range=1.0, win_size=win_size)
            ssim_values.append(ssim_value)
        return np.mean(ssim_values)  # 返回所有通道的平均SSIM值
    except Exception as e:
        print(f"SSIM计算错误: {str(e)}")
        return 0.0  # 如果计算失败，返回0

def calculate_mae(img1, img2):
    """计算MAE"""
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    # 确保数据范围在[0,1]之间
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    return mean_squared_error(img1, img2)

def denormalize(tensor):
    """反归一化图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def tensor_to_numpy(tensor):
    """将张量转换为numpy数组"""
    tensor = denormalize(tensor)
    tensor = tensor.clamp(0, 1)
    return tensor.cpu().numpy().transpose(1, 2, 0)

def save_image(tensor, filename):
    """保存图像"""
    img = tensor_to_numpy(tensor)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def calculate_metrics(sr_img, hr_img):
    """计算所有评估指标"""
    try:
        psnr_value = calculate_psnr(sr_img, hr_img)
        ssim_value = calculate_ssim(sr_img, hr_img)
        mae_value = calculate_mae(sr_img, hr_img)
        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'mae': mae_value
        }
    except Exception as e:
        print(f"指标计算错误: {str(e)}")
        return {
            'psnr': 0.0,
            'ssim': 0.0,
            'mae': 0.0
        } 