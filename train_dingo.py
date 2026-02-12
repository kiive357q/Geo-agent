#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 - 启动 MEF-Net 训练（重建任务）
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from dingo_core.engine.trainer import Trainer

def plot_reconstruction(original, reconstructed, save_path):
    """
    绘制原始波形与重建波形的对比图
    
    参数:
    - original: 原始波形
    - reconstructed: 重建波形
    - save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original, label='Original', linewidth=2)
    plt.plot(reconstructed, label='Reconstructed', linewidth=2, linestyle='--')
    plt.title('Waveform Reconstruction Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reconstruction plot saved to: {save_path}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Train MEF-Net on Dingo dataset (Reconstruction Task)')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=r"D:\本科论文\MEF-NSPC-RL\V2.0\MEF-NSPC-RL\3r14qbdhv648b2p83gjqby2fl8\ALL", help='Dingo dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=1024, help='Max waveform length')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--wave_eq_weight', type=float, default=0.1, help='Wave equation loss weight')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight')
    
    args = parser.parse_args()
    
    # 确保检查点目录存在
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 训练配置
    config = {
        'data_dir': args.data_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'train_val_split': args.train_val_split,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'wave_eq_weight': args.wave_eq_weight,
        'recon_weight': args.recon_weight,
        'task_type': 'reconstruction'  # 切换为重建任务
    }
    
    print("\n=== Training Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("===========================")
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()
    
    # 训练结束后，进行重建可视化
    print("\n=== Reconstruction Visualization ===")
    # 加载最佳模型
    best_model_path = os.path.join(args.checkpoint_dir, 'best_recon_model.pth')
    if os.path.exists(best_model_path):
        # 加载模型
        trainer.model.load_state_dict(torch.load(best_model_path))
        trainer.model.eval()
        
        # 获取验证数据的一个批次
        val_loader = trainer._get_val_loader()
        if val_loader:
            with torch.no_grad():
                for batch in val_loader:
                    wave = batch['wave'].to(trainer.device)
                    # 前向传播
                    reconstructed_wave = trainer.model.reconstruct(wave)
                    # 绘制对比图
                    plot_path = os.path.join(args.checkpoint_dir, 'reconstruction_example.png')
                    plot_reconstruction(
                        wave[0].cpu().numpy(),
                        reconstructed_wave[0].cpu().numpy(),
                        plot_path
                    )
                    break
    else:
        print("Best model not found. Skipping visualization.")
    
    # 复制最佳模型到指定路径
    best_model_path = os.path.join(args.checkpoint_dir, 'best_recon_model.pth')
    # 找到训练器创建的检查点目录
    import glob
    checkpoint_dirs = glob.glob(os.path.join(args.checkpoint_dir, '*'))
    if checkpoint_dirs:
        latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
        best_checkpoint = os.path.join(latest_dir, 'best_model.pth')
        if os.path.exists(best_checkpoint):
            import shutil
            shutil.copy(best_checkpoint, best_model_path)
            print(f"\nBest reconstruction model copied to: {best_model_path}")
        else:
            print(f"\nWarning: Best model not found in {latest_dir}")
    else:
        print(f"\nWarning: No checkpoint directories found in {args.checkpoint_dir}")

if __name__ == '__main__':
    main()
