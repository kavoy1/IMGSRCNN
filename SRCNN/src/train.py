import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from models.srcnn import SRCNN
from data.dataset import SRDataset
from utils.metrics import calculate_metrics, save_image

def train():
    # 设置路径
    # train_hr_dir = r"D:\resp\SRCNN\src\data\train\HR"
    # train_lr_dir = r"D:\resp\SRCNN\src\data\train\LR"
    # val_hr_dir = r"D:\resp\SRCNN\src\data\val\HR"
    # val_lr_dir = r"D:\resp\SRCNN\src\data\val\LR"

    save_dir = "checkpoints"
    log_dir = "logs"
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, timestamp)
    log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 创建数据加载器
    train_dataset = SRDataset(train_hr_dir, train_lr_dir, scale=8, augment=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dataset = SRDataset(val_hr_dir, val_lr_dir, scale=8, augment=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 创建tensorboard写入器
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    best_psnr = 0
    epochs = 100
    eval_interval = 5  # 每5个epoch评估一次
    save_interval = 10  # 每10个epoch保存一次模型
    vis_interval = 10  # 每10个epoch可视化一次
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # 验证阶段
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_metrics = {'psnr': 0, 'ssim': 0, 'mae': 0}
            
            with torch.no_grad():
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs = lr_imgs.to(device)
                    hr_imgs = hr_imgs.to(device)
                    
                    sr_imgs = model(lr_imgs)
                    metrics = calculate_metrics(sr_imgs, hr_imgs)
                    
                    for k, v in metrics.items():
                        val_metrics[k] += v
            
            # 计算平均指标
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)
                writer.add_scalar(f'Metrics/{k.upper()}', val_metrics[k], epoch)
            
            # 更新学习率
            scheduler.step(val_metrics['psnr'])
            
            # 保存最佳模型
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))
        
        # 可视化结果
        if (epoch + 1) % vis_interval == 0:
            visualize_results(model, val_loader, device, epoch, save_dir)
    
    writer.close()
    print(f'训练完成！最佳PSNR: {best_psnr:.2f}dB')

def visualize_results(model, val_loader, device, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
            if i >= 4:  # 只显示4张图片
                break
                
            lr_imgs = lr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            
            # 保存对比图像
            for j in range(min(4, lr_imgs.size(0))):
                save_image(lr_imgs[j], os.path.join(save_dir, f'epoch{epoch}_img{i}_lr.png'))
                save_image(sr_imgs[j], os.path.join(save_dir, f'epoch{epoch}_img{i}_sr.png'))
                save_image(hr_imgs[j], os.path.join(save_dir, f'epoch{epoch}_img{i}_hr.png'))

if __name__ == '__main__':
    train()
