import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, scale=8, patch_size=33, augment=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch_size = patch_size  # 低分辨率图像块大小应该与高分辨率相同
        self.augment = augment
        
        # 获取所有高分辨率图像文件
        self.image_files = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 数据增强
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_name = self.image_files[idx]
        hr_path = os.path.join(self.hr_dir, hr_name)
        
        # 读取高分辨率图像
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.lr_dir:
            # 如果提供了低分辨率图像目录，构造对应的低分辨率图像文件名
            lr_name = hr_name.replace('.png', f'x{self.scale}.png')
            lr_path = os.path.join(self.lr_dir, lr_name)
            lr_img = Image.open(lr_path).convert('RGB')
        else:
            # 否则从高分辨率图像生成低分辨率图像
            w, h = hr_img.size
            lr_size = (w // self.scale, h // self.scale)
            lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        
        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                hr_img = self.augment_transform(hr_img)
                lr_img = self.augment_transform(lr_img)
        
        # 确保图像大小足够进行裁剪
        w, h = hr_img.size
        if w < self.patch_size or h < self.patch_size:
            hr_img = hr_img.resize((max(w, self.patch_size), max(h, self.patch_size)), Image.BICUBIC)
            lr_img = lr_img.resize((max(w//self.scale, self.lr_patch_size), max(h//self.scale, self.lr_patch_size)), Image.BICUBIC)
            w, h = hr_img.size
        
        # 随机裁剪
        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        hr_patch = hr_img.crop((left, top, left + self.patch_size, top + self.patch_size))
        lr_patch = lr_img.crop((left // self.scale, top // self.scale, 
                              (left + self.patch_size) // self.scale, 
                              (top + self.patch_size) // self.scale))
        
        # 确保裁剪后的图像块大小正确
        if hr_patch.size != (self.patch_size, self.patch_size):
            hr_patch = hr_patch.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        if lr_patch.size != (self.lr_patch_size, self.lr_patch_size):
            lr_patch = lr_patch.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        
        # 转换为张量
        hr_tensor = self.transform(hr_patch)
        lr_tensor = self.transform(lr_patch)
        
        return lr_tensor, hr_tensor 