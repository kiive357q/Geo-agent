import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class ResNet1D(nn.Module):
    """
    ResNet-1D (3个 Residual Blocks)
    """
    
    def __init__(self, input_dim=1024, hidden_dim=64, num_blocks=3):
        super(ResNet1D, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        
        # ResNet 块
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim)
                )
            )
        
        # 全局池化和输出层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(hidden_dim, 256)
    
    def forward(self, x):
        # 输入形状: (batch_size, 1024)
        x = x.unsqueeze(1)  # 添加通道维度: (batch_size, 1, 1024)
        
        # 初始卷积
        x = self.initial_conv(x)
        
        # ResNet 块
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual
            x = F.relu(x)
        
        # 全局池化
        x = self.global_pool(x).squeeze(2)  # (batch_size, hidden_dim)
        
        # 输出特征
        x = self.output_fc(x)  # (batch_size, 256)
        
        return x

class MiniViT(nn.Module):
    """
    Mini-ViT (Vision Transformer)
    """
    
    def __init__(self, input_size=64, patch_size=8, num_channels=1, hidden_size=192, num_classes=256):
        super(MiniViT, self).__init__()
        
        # 配置 Mini-ViT
        config = ViTConfig(
            image_size=input_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_hidden_layers=6,  # 减少层数，实现 Mini-ViT
            num_attention_heads=3,
            intermediate_size=768,
            num_classes=num_classes
        )
        
        # 创建 ViT 模型
        self.vit = ViTModel(config)
        
        # 输出层
        self.output_fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 输入形状: (batch_size, 1, 64, 64)
        outputs = self.vit(pixel_values=x)
        
        # 获取 CLS token 输出
        x = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # 输出特征
        x = self.output_fc(x)  # (batch_size, 256)
        
        return x

class EntropyCalculator(nn.Module):
    """
    计算特征的香农熵
    """
    
    def forward(self, features):
        # 输入形状: (batch_size, feature_dim)
        # 计算每个特征的概率分布
        probabilities = F.softmax(features, dim=1)
        
        # 计算香农熵
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1, keepdim=True)
        
        return entropy

class GatingNetwork(nn.Module):
    """
    门控网络，根据特征和熵计算权重
    """
    
    def __init__(self, input_dim=256*2 + 1):
        super(GatingNetwork, self).__init__()
        
        # 小型 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出两个标量 α, β
            nn.Sigmoid()  # 确保输出在 0~1 之间
        )
    
    def forward(self, wave_features, image_features, entropy):
        # 拼接输入
        combined = torch.cat([wave_features, image_features, entropy], dim=1)
        
        # 计算权重
        weights = self.mlp(combined)  # (batch_size, 2)
        
        return weights

class MEFNet(nn.Module):
    """
    多专家融合网络
    """
    
    def __init__(self):
        super(MEFNet, self).__init__()
        
        # 专家网络
        self.wave_encoder = ResNet1D()  # Expert 1: Wave Encoder
        self.image_encoder = MiniViT()  # Expert 2: Image Encoder
        
        # 熵计算器
        self.entropy_calculator = EntropyCalculator()
        
        # 门控网络
        self.gating_network = GatingNetwork()
        
        # 决策头
        self.classifier = nn.Linear(256, 4)  # 分类头：4 类桩
        self.regression_head = nn.Linear(256, 1)  # 回归头：预测 defect_depth
        
        # 重建头
        self.reconstruction_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
    
    def forward(self, wave, image, param):
        # 专家网络前向传播
        wave_features = self.wave_encoder(wave)
        image_features = self.image_encoder(image)
        
        # 计算波特征的熵
        entropy = self.entropy_calculator(wave_features)
        
        # 门控网络计算权重
        weights = self.gating_network(wave_features, image_features, entropy)
        alpha, beta = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        
        # 融合特征
        final_feature = alpha * wave_features + beta * image_features
        
        # 分类和回归
        logits = self.classifier(final_feature)
        defect_depth = self.regression_head(final_feature)
        
        return logits, defect_depth, weights
    
    def reconstruct(self, wave):
        """
        重建波形
        
        参数:
        - wave: 输入波形 (batch_size, 1024)
        
        返回:
        - reconstructed_wave: 重建波形 (batch_size, 1024)
        """
        # 创建dummy输入
        batch_size = wave.shape[0]
        dummy_image = torch.zeros(batch_size, 1, 64, 64, device=wave.device)
        dummy_param = torch.zeros(batch_size, 10, device=wave.device)
        
        # 前向传播获取特征
        wave_features = self.wave_encoder(wave)
        image_features = self.image_encoder(dummy_image)
        entropy = self.entropy_calculator(wave_features)
        weights = self.gating_network(wave_features, image_features, entropy)
        alpha, beta = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        final_feature = alpha * wave_features + beta * image_features
        
        # 重建波形
        reconstructed_wave = self.reconstruction_head(final_feature)
        
        return reconstructed_wave

if __name__ == "__main__":
    # 测试模型
    model = MEFNet()
    
    # 生成测试数据
    wave = torch.randn(8, 1024)
    image = torch.randn(8, 1, 64, 64)
    param = torch.randn(8, 10)
    
    # 前向传播
    logits, defect_depth, weights = model(wave, image, param)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Defect Depth shape: {defect_depth.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights:\n{weights}")
    print(f"Alpha values: {weights[:, 0].tolist()}")
    print(f"Beta values: {weights[:, 1].tolist()}")
