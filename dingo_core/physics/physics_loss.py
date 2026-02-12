import torch
import torch.nn as nn
import numpy as np

class WaveEquationLoss(nn.Module):
    """
    波动方程残差 Loss
    计算波形信号与波动方程解之间的残差
    """
    
    def __init__(self, c=1.0, dx=0.1, dt=0.01):
        super(WaveEquationLoss, self).__init__()
        self.c = c  # 波速
        self.dx = dx  # 空间步长
        self.dt = dt  # 时间步长
        self.r = (c * dt / dx) ** 2  # 稳定性参数
    
    def forward(self, waveform):
        """
        计算波动方程残差
        
        参数:
        - waveform: 形状为 (batch_size, 1024) 的波形数据
        
        返回:
        - 残差 Loss
        """
        batch_size, length = waveform.shape
        
        # 计算二阶时间导数 (中心差分)
        d2u_dt2 = torch.zeros_like(waveform)
        d2u_dt2[:, 1:-1] = waveform[:, 2:] - 2 * waveform[:, 1:-1] + waveform[:, :-2]
        d2u_dt2[:, 0] = waveform[:, 1] - 2 * waveform[:, 0] + waveform[:, 0]
        d2u_dt2[:, -1] = waveform[:, -1] - 2 * waveform[:, -1] + waveform[:, -2]
        d2u_dt2 /= self.dt ** 2
        
        # 计算二阶空间导数 (中心差分)
        d2u_dx2 = torch.zeros_like(waveform)
        d2u_dx2[:, 1:-1] = waveform[:, 2:] - 2 * waveform[:, 1:-1] + waveform[:, :-2]
        d2u_dx2[:, 0] = waveform[:, 1] - 2 * waveform[:, 0] + waveform[:, 0]
        d2u_dx2[:, -1] = waveform[:, -1] - 2 * waveform[:, -1] + waveform[:, -2]
        d2u_dx2 /= self.dx ** 2
        
        # 波动方程: d²u/dt² = c² * d²u/dx²
        # 残差: d²u/dt² - c² * d²u/dx²
        residual = d2u_dt2 - (self.c ** 2) * d2u_dx2
        
        # 计算残差的均方误差
        loss = torch.mean(residual ** 2)
        
        return loss

class PhysicsConsistencyLoss(nn.Module):
    """
    物理一致性 Loss
    确保模型的预测符合波动方程
    """
    
    def __init__(self, c=4000.0, window_size=20):
        super(PhysicsConsistencyLoss, self).__init__()
        self.c = c  # 波速 (m/s)
        self.window_size = window_size  # 窗口大小
    
    def forward(self, waveform, pred_depth, pred_class):
        """
        计算物理一致性 Loss
        
        参数:
        - waveform: 原始波形数据 (batch_size, 1024)
        - pred_depth: 预测的缺陷深度 (batch_size, 1)
        - pred_class: 预测的类别 (batch_size,)
        
        返回:
        - 物理一致性 Loss
        """
        batch_size, length = waveform.shape
        
        # 计算理论反射时间 (单位: 秒)
        t_theory = 2 * pred_depth.squeeze(1) / self.c
        
        # 假设采样频率为 20000 Hz，将时间转换为采样点位置
        sample_rate = 20000
        t_positions = t_theory * sample_rate
        
        # 计算每个样本的窗口内能量（使用高斯窗口，保留梯度）
        energies = torch.zeros(batch_size, device=waveform.device)
        
        for i in range(batch_size):
            # 创建高斯窗口
            t = torch.linspace(0, length-1, length, device=waveform.device)
            sigma = self.window_size / 4
            gaussian_window = torch.exp(-0.5 * ((t - t_positions[i]) / sigma) ** 2)
            gaussian_window = gaussian_window / gaussian_window.sum()
            
            # 计算加权能量
            weighted_energy = torch.sum((waveform[i] ** 2) * gaussian_window)
            energies[i] = weighted_energy
        
        # 归一化能量
        energies = energies / (self.window_size * waveform.max())
        
        # 计算物理 Loss
        physics_loss = torch.tensor(0.0, device=waveform.device)
        
        for i in range(batch_size):
            if pred_class[i] in [2, 3]:  # 断桩/缺陷 (III 类桩或 IV 类桩)
                # 如果窗口内能量极低，惩罚
                if energies[i] < 0.1:
                    physics_loss += (0.1 - energies[i]) ** 2
            else:  # 完整桩 (I 类桩或 II 类桩)
                # 如果窗口内能量极高，惩罚
                if energies[i] > 0.5:
                    physics_loss += (energies[i] - 0.5) ** 2
        
        # 平均 Loss
        physics_loss = physics_loss / batch_size
        
        return physics_loss

class PhysicsLoss(nn.Module):
    """
    物理 Loss 组合
    """
    
    def __init__(self, wave_eq_weight=0.1, physics_consistency_weight=0.1):
        super(PhysicsLoss, self).__init__()
        self.wave_eq_loss = WaveEquationLoss()
        self.physics_consistency_loss = PhysicsConsistencyLoss()
        self.wave_eq_weight = wave_eq_weight
        self.physics_consistency_weight = physics_consistency_weight
    
    def forward(self, waveform, logits, defect_depth, labels):
        """
        计算总物理 Loss
        
        参数:
        - waveform: 波形数据
        - logits: 模型输出的分类 logits
        - defect_depth: 模型输出的缺陷深度
        - labels: 真实标签
        
        返回:
        - 总 Loss
        """
        # 处理标签为 -1 的样本
        mask = (labels != -1)
        
        # 分类 Loss
        if mask.sum() > 0:
            criterion = nn.CrossEntropyLoss()
            cls_loss = criterion(logits[mask], labels[mask])
        else:
            # 如果 Batch 中全是无标签数据，则 cls_loss = 0.0，仅计算 Physics Loss
            cls_loss = torch.tensor(0.0, device=waveform.device)
        
        # 波动方程残差 Loss
        wave_loss = self.wave_eq_loss(waveform)
        
        # 物理一致性 Loss
        pred_class = torch.argmax(logits, dim=1)
        physics_loss = self.physics_consistency_loss(waveform, defect_depth, pred_class)
        
        # 总 Loss
        total_loss = cls_loss + self.wave_eq_weight * wave_loss + self.physics_consistency_weight * physics_loss
        
        return total_loss, cls_loss, wave_loss, physics_loss

class JGJ106RuleEngine:
    """
    JGJ-106 规则引擎
    作为 Agent 的 Safety Valve
    """
    
    def __init__(self):
        # JGJ-106 规则阈值
        self.thresholds = {
            "class_I": 0.8,  # I 类桩阈值
            "class_II": 0.5,  # II 类桩阈值
            "class_III": 0.2,  # III 类桩阈值
            "class_IV": 0.0   # IV 类桩阈值
        }
    
    def validate(self, prediction, waveform):
        """
        根据 JGJ-106 规则验证预测结果
        
        参数:
        - prediction: 模型预测结果
        - waveform: 原始波形数据
        
        返回:
        - 验证后的预测结果
        """
        # 计算波形特征
        max_amplitude = torch.max(torch.abs(waveform), dim=1)[0]
        energy = torch.sum(waveform ** 2, dim=1)
        
        # 根据规则调整预测
        validated_prediction = prediction.clone()
        
        for i in range(len(prediction)):
            # 计算波形评分
            score = (energy[i] / (max_amplitude[i] + 1e-6)).item()
            
            # 根据 JGJ-106 规则调整
            if score > self.thresholds["class_I"]:
                validated_prediction[i] = 0  # I 类桩
            elif score > self.thresholds["class_II"]:
                validated_prediction[i] = 1  # II 类桩
            elif score > self.thresholds["class_III"]:
                validated_prediction[i] = 2  # III 类桩
            else:
                validated_prediction[i] = 3  # IV 类桩
        
        return validated_prediction

if __name__ == "__main__":
    # 测试物理 Loss
    loss_fn = PhysicsLoss()
    waveform = torch.randn(8, 1024)
    logits = torch.randn(8, 4)
    defect_depth = torch.randn(8, 1)
    labels = torch.randint(0, 4, (8,))
    
    total_loss, cls_loss, wave_loss, physics_loss = loss_fn(waveform, logits, defect_depth, labels)
    print(f"Total Loss: {total_loss.item()}")
    print(f"Classification Loss: {cls_loss.item()}")
    print(f"Wave Equation Loss: {wave_loss.item()}")
    print(f"Physics Consistency Loss: {physics_loss.item()}")
    
    # 测试 JGJ-106 规则引擎
    rule_engine = JGJ106RuleEngine()
    prediction = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    validated_prediction = rule_engine.validate(prediction, waveform)
    print(f"Original Prediction: {prediction.tolist()}")
    print(f"Validated Prediction: {validated_prediction.tolist()}")
