#!/usr/bin/env python3
"""
Geo-Agent 预测器模块
负责加载训练好的模型并进行推理
"""

import os
import numpy as np
import torch
from scipy.signal import resample
from dingo_core.modeling.mef_net import MEFNet

class GeoPredictor:
    """
    地理预测器类
    负责加载模型权重并进行推理
    """
    
    def __init__(self, checkpoint_dir=None):
        """
        初始化预测器
        
        参数:
        - checkpoint_dir: 检查点目录
        """
        # 如果没有提供 checkpoint_dir，使用相对于当前文件的路径
        if checkpoint_dir is None:
            # 获取当前文件的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上两级到 Geo-Agent-Dingo 目录，然后进入 checkpoints 目录
            self.checkpoint_dir = os.path.join(current_dir, '..', '..', 'checkpoints')
        else:
            self.checkpoint_dir = checkpoint_dir
        
        # 规范化路径
        self.checkpoint_dir = os.path.normpath(self.checkpoint_dir)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """
        加载最新的模型权重
        """
        print("🔧 加载模型权重...")
        
        # 查找最新的模型权重文件
        best_model_path = os.path.join(self.checkpoint_dir, 'best_recon_model.pth')
        
        if os.path.exists(best_model_path):
            # 加载模型
            self.model = MEFNet().to(self.device)
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ 模型加载成功: {best_model_path}")
        else:
            # 尝试查找其他模型文件
            checkpoint_files = []
            for root, _, files in os.walk(self.checkpoint_dir):
                for file in files:
                    if file.endswith('.pth'):
                        checkpoint_files.append(os.path.join(root, file))
            
            if checkpoint_files:
                # 选择最新的文件
                latest_file = max(checkpoint_files, key=os.path.getmtime)
                self.model = MEFNet().to(self.device)
                self.model.load_state_dict(torch.load(latest_file, map_location=self.device))
                self.model.eval()
                print(f"✅ 模型加载成功: {latest_file}")
            else:
                raise FileNotFoundError("❌ 未找到模型权重文件")
    
    def preprocess(self, raw_signal):
        """
        预处理原始波形
        
        参数:
        - raw_signal: 原始波形数据
        
        返回:
        - 预处理后的波形数据
        """
        # 归一化
        mean_val = np.mean(raw_signal)
        max_val = np.max(np.abs(raw_signal - mean_val))
        if max_val == 0:
            max_val = 1.0
        norm_signal = (raw_signal - mean_val) / max_val
        
        # 重采样到1024
        if len(norm_signal) != 1024:
            norm_signal = resample(norm_signal, 1024)
        
        return norm_signal
    
    def predict(self, raw_signal):
        """
        模型推理
        
        参数:
        - raw_signal: 原始波形数据
        
        返回:
        - 推理结果
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 预处理
        norm_signal = self.preprocess(raw_signal)
        
        # 转换为张量
        wave_tensor = torch.tensor(norm_signal, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            # 重建波形
            reconstructed_wave = self.model.reconstruct(wave_tensor)
            
            # 获取缺陷深度
            dummy_image = torch.zeros(1, 1, 64, 64, device=self.device)
            dummy_param = torch.zeros(1, 10, device=self.device)
            _, defect_depth, _ = self.model(wave_tensor, dummy_image, dummy_param)
        
        # 计算物理置信度
        mse = torch.mean((wave_tensor - reconstructed_wave) ** 2).item()
        confidence = 1.0 - mse
        
        return {
            'reconstructed_wave': reconstructed_wave.squeeze().cpu().numpy(),
            'defect_depth': defect_depth.item(),
            'confidence': confidence,
            'mse': mse,
            'input_wave': norm_signal
        }
    
    def get_model_info(self):
        """
        获取模型信息
        """
        if self.model is None:
            return "模型未加载"
        return f"GeoFormer 模型 (设备: {self.device.type})"

if __name__ == "__main__":
    # 测试预测器
    try:
        predictor = GeoPredictor()
        print(f"模型信息: {predictor.get_model_info()}")
        
        # 生成测试数据
        test_signal = np.random.randn(1024)
        result = predictor.predict(test_signal)
        print(f"推理结果: ")
        print(f"  缺陷深度: {result['defect_depth']:.2f}m")
        print(f"  物理置信度: {result['confidence']:.4f}")
        print(f"  MSE: {result['mse']:.4f}")
    except Exception as e:
        print(f"测试失败: {e}")
