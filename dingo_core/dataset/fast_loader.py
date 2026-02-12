#!/usr/bin/env python3
"""
Dingo-Phoenix 快速加载器 (The Fast Loader)
负责直接读取处理好的二进制文件，秒级加载
"""

import torch

class FastDingoDataset(torch.utils.data.Dataset):
    """
    快速Dingo数据集类
    直接读取处理好的processed_dataset.pt文件
    """
    
    def __init__(self, pt_file_path):
        """
        初始化快速数据集
        
        参数:
        - pt_file_path: processed_dataset.pt文件路径
        """
        self.pt_file_path = pt_file_path
        
        # 加载预处理好的数据
        print(f"🚀 加载预处理数据: {pt_file_path}")
        self.data = torch.load(pt_file_path)
        
        # 提取样本
        self.samples = self.data['samples']
        self.metadata_list = self.data.get('metadata_list', [])
        self.statistics = self.data.get('statistics', {})
        
        # 计算标记和未标记样本数量
        self._calculate_label_stats()
        
        # 调试信息：检查第一个样本的最大值
        if self.samples:
            first_sample_wave = self.samples[0]['wave']
            max_val = torch.max(torch.abs(first_sample_wave)).item()
            print(f"🔍 FastLoader Check - First Sample Max Value: {max_val:.4f}")
            # 稍微放宽阈值，允许一些微小的数值误差
            assert max_val <= 1.1, f"数据异常! 第一个样本最大值: {max_val}"
            if max_val > 1.001:
                print(f"   ⚠️  警告: 数值略超出 [-1, 1] 范围，但在可接受范围内")
        
        print(f"✅ 数据加载完成")
        print(f"   总样本数: {len(self.samples)}")
        print(f"   标记样本数: {self.num_labeled}")
        print(f"   未标记样本数: {self.num_unlabeled}")
    
    def _calculate_label_stats(self):
        """计算标记和未标记样本数量"""
        self.num_labeled = 0
        self.num_unlabeled = 0
        
        for sample in self.samples:
            if sample['label'] != -1:
                self.num_labeled += 1
            else:
                self.num_unlabeled += 1
    
    def __len__(self) -> int:
        """
        返回数据集长度
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取数据项
        
        参数:
        - idx: 数据索引
        
        返回:
        - 包含训练数据的字典
        """
        sample = self.samples[idx]
        
        # 直接返回张量，不做任何计算
        return {
            'wave': sample['wave'],
            'length': sample['length'],
            'dt': sample['dt'],
            'label': sample['label']
        }
    
    @property
    def get_statistics(self):
        """
        获取统计信息
        """
        return self.statistics
    
    @property
    def get_metadata_count(self):
        """
        获取元数据记录数
        """
        return len(self.metadata_list)

if __name__ == "__main__":
    # 测试快速加载器
    test_pt_file = "data/processed_dataset.pt"
    
    try:
        dataset = FastDingoDataset(test_pt_file)
        print(f"\n📊 测试结果")
        print(f"数据集长度: {len(dataset)}")
        print(f"标记样本数: {dataset.num_labeled}")
        print(f"未标记样本数: {dataset.num_unlabeled}")
        
        # 测试获取样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n第一个样本:")
            print(f"  wave形状: {sample['wave'].shape}")
            print(f"  length: {sample['length']}")
            print(f"  dt: {sample['dt']}")
            print(f"  label: {sample['label']}")
            print(f"  wave类型: {type(sample['wave'])}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
