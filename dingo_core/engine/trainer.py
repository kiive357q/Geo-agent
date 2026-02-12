#!/usr/bin/env python3
"""
训练器 - 实现信号重建任务和物理一致性训练
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
from datetime import datetime
import sys

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dingo_core.dataset.fast_loader import FastDingoDataset
from dingo_core.modeling.mef_net import MEFNet
from dingo_core.physics.physics_loss import PhysicsLoss

class Trainer:
    """
    训练器
    """
    
    def __init__(self, config):
        """
        初始化训练器
        
        参数:
        - config: 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用快速加载器
        processed_pt_file = os.path.join('data', 'processed_dataset.pt')
        
        # 检查 processed_dataset.pt 文件是否存在
        if not os.path.exists(processed_pt_file):
            raise FileNotFoundError(
                f"Error: {processed_pt_file} not found! "
                "Please run preprocess_data.py first to generate the processed dataset."
            )
        
        # 创建快速数据集
        self.dataset = FastDingoDataset(processed_pt_file)
        self.batch_size = config['batch_size']
        self.train_val_split = config['train_val_split']
        self.task_type = config.get('task_type', 'reconstruction')
        self.recon_weight = config.get('recon_weight', 1.0)
        
        # 创建模型
        self.model = MEFNet().to(self.device)
        
        # 创建损失函数
        self.physics_loss_fn = PhysicsLoss(wave_eq_weight=config['wave_eq_weight'])
        self.mse_loss_fn = nn.MSELoss()
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # 检查点目录
        self.checkpoint_dir = os.path.join(
            config['checkpoint_dir'],
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 日志
        self.logs = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_physics_loss': [],
            'val_physics_loss': [],
            'learning_rate': []
        }
        
        # 早停机制
        self.early_stop_patience = config.get('early_stop_patience', 10)
        self.early_stop_counter = 0
        self.best_loss = float('inf')
    
    def _get_dataloaders(self):
        """
        获取训练和验证数据加载器
        """
        # 分割数据集为训练和验证集
        dataset_size = len(self.dataset)
        train_size = int(self.train_val_split * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"数据集分割完成:")
        print(f"  总样本数: {dataset_size}")
        print(f"  训练样本数: {train_size}")
        print(f"  验证样本数: {val_size}")
        
        return train_loader, val_loader
    
    def _get_val_loader(self):
        """
        获取验证数据加载器
        """
        _, val_loader = self._get_dataloaders()
        return val_loader
    
    def train_epoch(self, train_loader):
        """
        训练一个 epoch
        
        参数:
        - train_loader: 训练数据加载器
        
        返回:
        - 平均训练损失
        - 平均重建损失
        - 平均物理损失
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_physics_loss = 0
        
        # 数据健全性检查
        first_batch = True
        
        for batch in tqdm(train_loader, desc='Training'):
            # 移动数据到设备
            wave = batch['wave'].to(self.device)
            
            # 数据健全性检查
            if first_batch:
                # 检查波形数据
                wave_max = wave.max().item()
                wave_min = wave.min().item()
                print(f"Wave Max: {wave_max}, Min: {wave_min}")
                # 改为警告信息，允许训练继续进行
                if wave_max > 1.001 or wave_min < -1.001:
                    print(f"   ⚠️  警告: 数值超出 [-1, 1] 范围，但训练将继续进行")
                    print(f"   最大值: {wave_max}, 最小值: {wave_min}")
                first_batch = False
            
            # 前向传播 - 重建任务
            reconstructed_wave = self.model.reconstruct(wave)
            
            # 计算损失
            # 1. 重建损失
            recon_loss = self.mse_loss_fn(reconstructed_wave, wave)
            
            # 2. 物理损失
            # 创建dummy输入用于原始前向传播
            batch_size = wave.shape[0]
            dummy_image = torch.zeros(batch_size, 1, 64, 64, device=self.device)
            dummy_param = torch.zeros(batch_size, 10, device=self.device)
            logits, defect_depth, _ = self.model(wave, dummy_image, dummy_param)
            _, _, _, physics_loss = self.physics_loss_fn(wave, logits, defect_depth, torch.full((batch_size,), -1, device=self.device))
            
            # 3. 总损失
            total_loss_val = self.recon_weight * recon_loss + 0.1 * physics_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_val.backward()
            self.optimizer.step()
            
            # 累加损失
            total_loss += total_loss_val.item()
            total_recon_loss += recon_loss.item()
            total_physics_loss += physics_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        
        return avg_loss, avg_recon_loss, avg_physics_loss
    
    def validate(self, val_loader):
        """
        验证模型
        
        参数:
        - val_loader: 验证数据加载器
        
        返回:
        - 平均验证损失
        - 平均重建损失
        - 平均物理损失
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_physics_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                # 移动数据到设备
                wave = batch['wave'].to(self.device)
                
                # 前向传播 - 重建任务
                reconstructed_wave = self.model.reconstruct(wave)
                
                # 计算损失
                # 1. 重建损失
                recon_loss = self.mse_loss_fn(reconstructed_wave, wave)
                
                # 2. 物理损失
                # 创建dummy输入用于原始前向传播
                batch_size = wave.shape[0]
                dummy_image = torch.zeros(batch_size, 1, 64, 64, device=self.device)
                dummy_param = torch.zeros(batch_size, 10, device=self.device)
                logits, defect_depth, _ = self.model(wave, dummy_image, dummy_param)
                _, _, _, physics_loss = self.physics_loss_fn(wave, logits, defect_depth, torch.full((batch_size,), -1, device=self.device))
                
                # 3. 总损失
                total_loss_val = self.recon_weight * recon_loss + 0.1 * physics_loss
                
                # 累加损失
                total_loss += total_loss_val.item()
                total_recon_loss += recon_loss.item()
                total_physics_loss += physics_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_physics_loss = total_physics_loss / len(val_loader)
        
        return avg_loss, avg_recon_loss, avg_physics_loss
    
    def train(self):
        """
        训练模型
        """
        # 分割数据集为训练和验证集
        train_loader, val_loader = self._get_dataloaders()
        
        # 启动检查
        print("Reconstruction Task Activated. Target: Minimize MSE + Physics Loss.")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 训练
            train_loss, train_recon_loss, train_physics_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_recon_loss, val_physics_loss = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            self.logs['train_loss'].append(train_loss)
            self.logs['val_loss'].append(val_loss)
            self.logs['train_recon_loss'].append(train_recon_loss)
            self.logs['val_recon_loss'].append(val_recon_loss)
            self.logs['train_physics_loss'].append(train_physics_loss)
            self.logs['val_physics_loss'].append(val_physics_loss)
            self.logs['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印损失
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Recon Loss: {train_recon_loss:.4f}, Val Recon Loss: {val_recon_loss:.4f}")
            print(f"Train Physics Loss: {train_physics_loss:.4f}, Val Physics Loss: {val_physics_loss:.4f}")
            
            # 保存检查点
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")
            
            # 早停检查
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_stop_counter = 0
                # 保存最佳模型
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved at: {best_model_path}")
            else:
                self.early_stop_counter += 1
                print(f"Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered! No improvement for {self.early_stop_patience} epochs.")
                    break
        
        # 保存日志
        self.save_logs()
        
        print("\nTraining completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_logs(self):
        """
        保存日志
        """
        log_path = os.path.join(self.checkpoint_dir, 'logs.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        print(f"Logs saved at: {log_path}")

if __name__ == "__main__":
    # 测试训练器
    test_config = {
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 32,
        'max_length': 1024,
        'train_val_split': 0.8,
        'epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'wave_eq_weight': 0.1,
        'recon_weight': 1.0,
        'task_type': 'reconstruction'
    }
    
    try:
        trainer = Trainer(test_config)
        print("Trainer created successfully!")
    except Exception as e:
        print(f"Error creating trainer: {e}")
