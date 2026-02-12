import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入新的DingoDataset
from dingo_loader import DingoDataset as NewDingoDataset

class DingoDataModule:
    """
    Dingo 数据模块
    负责创建训练和验证数据加载器
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, max_length: int = 1024, train_val_split: float = 0.8):
        """
        初始化数据模块
        
        参数:
        - data_dir: Dingo 数据集目录
        - batch_size: 批大小
        - max_length: 1D 信号固定长度
        - train_val_split: 训练集比例
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_val_split = train_val_split
    
    def setup(self, stage: str = None):
        """
        设置数据集
        
        参数:
        - stage: 阶段 (train, val, test)
        """
        # 创建完整数据集
        full_dataset = NewDingoDataset(self.data_dir)
        
        # 划分训练集和验证集
        train_size = int(len(full_dataset) * self.train_val_split)
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
    
    def train_dataloader(self) -> DataLoader:
        """
        创建训练数据加载器
        
        返回:
        - 训练数据加载器
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        创建验证数据加载器
        
        返回:
        - 验证数据加载器
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

# 测试代码
if __name__ == "__main__":
    # 测试新的DingoDataset
    data_dir = r"D:\本科论文\MEF-NSPC-RL\V2.0\MEF-NSPC-RL\3r14qbdhv648b2p83gjqby2fl8\ALL"
    dataset = NewDingoDataset(data_dir)
    
    print(f"数据集长度: {len(dataset)}")
    
    # 测试获取数据项
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"波形数据形状: {sample['wave'].shape}")
        print(f"桩长: {sample['length']}")
        print(f"采样间隔: {sample['dt']}")
        print(f"标签: {sample['label']}")
    
    # 测试 DataModule
    data_module = DingoDataModule(data_dir, batch_size=8)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"训练数据加载器批数: {len(train_loader)}")
    print(f"验证数据加载器批数: {len(val_loader)}")
    
    # 测试数据加载器
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"批波形数据形状: {batch['wave'].shape}")
        print(f"批桩长形状: {batch['length'].shape}")
        print(f"批采样间隔形状: {batch['dt'].shape}")
        print(f"批标签形状: {batch['label'].shape}")
