# SRCNN超分辨率重建项目

这是一个基于SRCNN（Super-Resolution Convolutional Neural Network）的超分辨率重建项目，支持8倍超分辨率重建，适用于DIV2K数据集。

## 项目结构

```
SRCNN/
├── src/
│   ├── models/
│   │   └── srcnn.py
│   ├── data/
│   │   └── dataset.py
│   ├── utils/
│   │   └── metrics.py
│   ├── train.py
│   └── inference.py
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.7+
- PyTorch 1.7.0+
- CUDA（推荐，用于GPU加速）

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/SRCNN.git
cd SRCNN
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集准备

1. 下载DIV2K数据集
2. 将数据集解压到项目目录下，确保包含以下结构：
```
data/
├── train/
│   ├── HR/  # DIV2K_train_HR
│   └── LR/  # DIV2K_train_LR_x8
└── val/
    ├── HR/  # DIV2K_valid_HR
    └── LR/  # DIV2K_valid_LR_x8
```

## 训练

```bash
python src/train.py \
    --train_hr_dir data/train/HR \
    --train_lr_dir data/train/LR \
    --val_hr_dir data/val/HR \
    --val_lr_dir data/val/LR \
    --save_dir checkpoints \
    --log_dir logs \
    --scale 8 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4
```

## 推理

```bash
python src/inference.py \
    --input_dir test_images \
    --hr_dir ground_truth \
    --output_dir results \
    --model_path checkpoints/best_model.pth
```

## 主要特点

1. 支持彩色图像处理
2. 自动处理图像尺寸
3. 每20次迭代保存模型
4. 包含PSNR/SSIM/MAE三种评估指标
5. 支持可视化对比显示
6. 适配DIV2K数据集
7. 支持数据增强
8. 使用学习率调度器
9. 使用TensorBoard记录训练过程

## 评估指标

项目支持以下评估指标：
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)

## 可视化

- 训练过程中的损失和评估指标可以通过TensorBoard查看：
```bash
tensorboard --logdir logs
```

- 每5个epoch会生成一次对比图像，保存在checkpoints目录下

## 模型保存

- 每20次迭代保存一次检查点
- 保存最佳模型（基于验证集PSNR）
- 所有模型文件保存在checkpoints目录下

## 注意事项

1. 确保有足够的GPU内存（推荐至少8GB）
2. 训练时间取决于数据集大小和硬件配置
3. 可以根据需要调整batch_size和学习率
4. 建议使用SSD存储数据集以提高读取速度 