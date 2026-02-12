import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from dingo_core.modeling.mef_net import MEFNet
from dingo_core.physics.physics_loss import JGJ106RuleEngine

class AgentBrain:
    """
    智能体大脑
    负责 CoT (思维链) 推理和用户对话
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化智能体大脑
        
        参数:
        - model_path: 模型权重路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = MEFNet().to(self.device)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from: {model_path}")
        self.model.eval()
        
        # JGJ-106 规则引擎
        self.rule_engine = JGJ106RuleEngine()
        
        # 桩类型映射
        self.pile_class_map = {
            0: "I 类桩",
            1: "II 类桩",
            2: "III 类桩",
            3: "IV 类桩"
        }
        
        # 桩类型描述
        self.pile_class_desc = {
            "I 类桩": "桩身完整，波形规则，无缺陷反射",
            "II 类桩": "桩身基本完整，波形基本规则，有轻微缺陷反射",
            "III 类桩": "桩身有明显缺陷，波形不规则，有严重缺陷反射",
            "IV 类桩": "桩身严重缺陷，波形严重不规则，有致命缺陷反射"
        }
    
    def preprocess_input(self, waveform: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预处理输入数据
        
        参数:
        - waveform: 原始波形数据
        
        返回:
        - 预处理后的多模态数据
        """
        # 归一化到 [-1, 1]
        max_abs = np.max(np.abs(waveform))
        if max_abs > 0:
            waveform = waveform / max_abs
        
        # 固定长度为 1024
        if len(waveform) > 1024:
            waveform = waveform[:1024]
        elif len(waveform) < 1024:
            waveform = np.pad(waveform, (0, 1024 - len(waveform)), 'constant')
        
        # 生成 STFT 时频图
        from scipy.signal import stft
        f, t, Zxx = stft(waveform, fs=20000, nperseg=128, noverlap=64)
        magnitude = np.abs(Zxx)
        magnitude = np.log1p(magnitude)
        
        # 调整大小到 (1, 64, 64)
        if magnitude.shape[0] > 64:
            magnitude = magnitude[:64, :]
        elif magnitude.shape[0] < 64:
            magnitude = np.pad(magnitude, ((0, 64 - magnitude.shape[0]), (0, 0)), 'constant')
        
        if magnitude.shape[1] > 64:
            magnitude = magnitude[:, :64]
        elif magnitude.shape[1] < 64:
            magnitude = np.pad(magnitude, ((0, 0), (0, 64 - magnitude.shape[1])), 'constant')
        
        stft_image = magnitude[np.newaxis, :, :]
        
        # 地质参数 (默认全零)
        param = np.zeros(10)
        
        # 转换为 Tensor
        wave_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        image_tensor = torch.tensor(stft_image, dtype=torch.float32).unsqueeze(0).to(self.device)
        param_tensor = torch.tensor(param, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return wave_tensor, image_tensor, param_tensor
    
    def cot_reasoning(self, waveform: np.ndarray) -> Dict:
        """
        思维链推理
        
        参数:
        - waveform: 原始波形数据
        
        返回:
        - 推理结果
        """
        # 预处理输入
        wave, image, param = self.preprocess_input(waveform)
        
        # 模型预测
        with torch.no_grad():
            logits, defect_depth, weights = self.model(wave, image, param)
            
        # 计算概率
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # 获取预测类别
        predicted_class = np.argmax(probs)
        predicted_pile_class = self.pile_class_map[predicted_class]
        
        # JGJ-106 规则验证
        validated_prediction = self.rule_engine.validate(
            torch.tensor([predicted_class]).to(self.device),
            wave.squeeze(0)
        )
        validated_pile_class = self.pile_class_map[validated_prediction.item()]
        
        # 思维链推理过程
        reasoning_steps = [
            f"Step 1: 分析时域波形特征 - 波形长度: {len(waveform)}, 最大值: {np.max(np.abs(waveform)):.4f}",
            f"Step 2: 生成频域时频图 - 分析频率分布特征",
            f"Step 3: 多专家融合预测 - Wave Expert 权重: {weights[0, 0]:.4f}, Image Expert 权重: {weights[0, 1]:.4f}",
            f"Step 4: 模型预测结果 - {predicted_pile_class} (概率: {probs[predicted_class]:.4f})",
            f"Step 5: 缺陷深度预测 - {defect_depth.item():.4f} m",
            f"Step 6: JGJ-106 规则验证 - 调整为: {validated_pile_class}",
            f"Step 7: 最终结论 - {self.pile_class_desc[validated_pile_class]}"
        ]
        
        # 生成结果
        result = {
            'predicted_class': predicted_class,
            'predicted_pile_class': predicted_pile_class,
            'validated_class': validated_prediction.item(),
            'validated_pile_class': validated_pile_class,
            'probabilities': probs.tolist(),
            'expert_weights': weights.squeeze(0).cpu().numpy().tolist(),
            'defect_depth': defect_depth.item(),
            'reasoning_steps': reasoning_steps,
            'pile_class_description': self.pile_class_desc[validated_pile_class]
        }
        
        return result
    
    def diagnose(self, waveform: np.ndarray) -> str:
        """
        桩身诊断
        
        参数:
        - waveform: 原始波形数据
        
        返回:
        - 诊断报告
        """
        # 思维链推理
        result = self.cot_reasoning(waveform)
        
        # 生成诊断报告
        report = f"# 桩身完整性诊断报告\n\n"
        report += f"## 诊断结果\n"
        report += f"- 预测桩类型: {result['predicted_pile_class']}\n"
        report += f"- 验证桩类型: {result['validated_pile_class']}\n"
        report += f"- 桩类型描述: {result['pile_class_description']}\n"
        report += f"- 缺陷深度预测: {result['defect_depth']:.4f} m\n\n"
        
        report += f"## 专家权重分析\n"
        report += f"- Wave Expert (时域): {result['expert_weights'][0]:.4f}\n"
        report += f"- Image Expert (频域): {result['expert_weights'][1]:.4f}\n\n"
        
        report += f"## 思维链推理\n"
        for i, step in enumerate(result['reasoning_steps'], 1):
            report += f"{i}. {step}\n"
        
        return report
    
    def converse(self, user_input: str, waveform: Optional[np.ndarray] = None) -> str:
        """
        用户对话
        
        参数:
        - user_input: 用户输入
        - waveform: 可选的波形数据
        
        返回:
        - 对话响应
        """
        user_input = user_input.lower()
        
        if '诊断' in user_input or '检测' in user_input or '分析' in user_input:
            if waveform is not None:
                return self.diagnose(waveform)
            else:
                return "请提供波形数据以进行诊断分析。"
        elif '帮助' in user_input or '使用' in user_input:
            return "我是 Geo-Agent Dingo-Phoenix，用于桩身完整性诊断。您可以提供波形数据，我会进行详细的分析和诊断。"
        elif '关于' in user_input:
            return "Geo-Agent Dingo-Phoenix 是基于 MEF-NSPC-RL 架构的工程级智能体，集成了多模态融合、熵权机制和物理约束，用于桩身完整性诊断。"
        else:
            return "我理解您需要桩身完整性诊断相关的帮助。请提供波形数据，我会为您进行详细分析。"

if __name__ == "__main__":
    # 测试 Agent Brain
    agent = AgentBrain()
    
    # 生成测试波形
    test_waveform = np.random.randn(1024)
    
    # 测试诊断
    report = agent.diagnose(test_waveform)
    print(report)
    
    # 测试对话
    response = agent.converse("帮我诊断这个桩的完整性", test_waveform)
    print("\n对话测试:")
    print(response)
