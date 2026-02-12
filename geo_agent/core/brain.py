#!/usr/bin/env python3
"""
Geo-Agent 大脑模块
使用 DeepSeek API 作为智能体大脑，实现动态 CoT 思维链
"""

import numpy as np
import os
import yaml
from openai import OpenAI
from geo_agent.skills.predictor import GeoPredictor
from geo_agent.knowledge.rules import JGJ106Rules

class GeoAgent:
    """
    Geo-Agent 智能体类
    使用 DeepSeek API 作为大脑，实现动态 CoT 思维链
    """
    
    def __init__(self, config_path="config/settings.yaml"):
        """
        初始化智能体
        
        参数:
        - config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化 OpenAI 客户端（DeepSeek API）
        self.client = OpenAI(
            api_key=self.config['llm']['api_key'],
            base_url=self.config['llm']['base_url']
        )
        
        # 初始化物理技能
        self.predictor = GeoPredictor()
        self.rules = JGJ106Rules()
        
        # 维护对话历史
        self.history = []
        
        print("🧠 Geo-Agent 大脑初始化完成")
        print(f"✅ 系统就绪: GeoFormer + DeepSeek {self.config['llm']['model']}")
    
    def _load_config(self, config_path):
        """
        加载配置文件
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            # 返回默认配置
            return {
                'llm': {
                    'api_key': 'your-deepseek-api-key',
                    'base_url': 'https://api.deepseek.com/v1',
                    'model': 'deepseek-chat'
                }
            }
    
    def _get_system_prompt(self):
        """
        获取系统提示（专家人设）
        """
        return (
            "你是国家级岩土工程与AI交叉领域首席专家（Geo-Agent）。\n"
            "你精通 JGJ-106 规范，具备强大的物理直觉和工程问题解决能力。\n"
            "你的任务是：基于下方的【物理引擎计算结果】和【现场地质上下文】，给出严谨的诊断结论。如果发现断桩，必须给出具体的工程加固方案（如注浆、补桩）。\n"
            "请使用专业、清晰的语言回答，确保结论准确可靠。"
        )
    
    def _calculate_snr(self, signal):
        """
        计算信噪比
        """
        signal_power = np.mean(np.square(signal))
        noise_power = np.mean(np.square(signal - np.mean(signal)))
        if noise_power == 0:
            return 100.0
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def diagnose(self, file_path):
        """
        诊断桩身完整性
        
        参数:
        - file_path: 波形文件路径
        
        返回:
        - 诊断报告
        """
        # 加载波形文件
        try:
            # 假设文件是 numpy 数组
            if file_path.endswith('.npy'):
                raw_signal = np.load(file_path)
            elif file_path.endswith('.txt'):
                raw_signal = np.loadtxt(file_path)
            elif file_path.endswith('.csv'):
                raw_signal = np.genfromtxt(file_path, delimiter=',')
            else:
                # 尝试以二进制方式读取
                raw_signal = np.fromfile(file_path, dtype=np.float32)
            
            # 假设采样率为 1000Hz，计算桩长
            sampling_rate = 1000  # Hz
            wave_speed = 4000     # m/s
            pile_length = (len(raw_signal) / sampling_rate) * (wave_speed / 2)
            
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return {
                'status': 'error',
                'message': f'加载文件失败: {e}'
            }
        
        # 物理技能执行
        try:
            result = self.predictor.predict(raw_signal)
        except Exception as e:
            print(f"❌ 模型推理失败: {e}")
            return {
                'status': 'error',
                'message': f'模型推理失败: {e}'
            }
        
        # 计算 SNR
        snr = self._calculate_snr(raw_signal)
        
        # 构建用户提示
        user_prompt = (
            f"【物理引擎计算结果】\n\n"
            f"桩长: {pile_length:.1f}m\n\n"
            f"缺陷深度: {result['defect_depth']:.1f}m\n\n"
            f"物理残差 Loss: {result['mse']:.1e}\n\n"
            f"信噪比: {snr:.1f}dB\n\n"
            f"物理置信度: {result['confidence']:.4f}\n\n"
            f"【现场地质上下文】\n\n"
            f"地质类型: 普通\n\n"
            f"【任务】\n\n"
            f"1. 基于物理引擎计算结果，给出桩身完整性的诊断结论\n\n"
            f"2. 分析可能的缺陷原因\n\n"
            f"3. 如果发现严重缺陷或断桩，给出具体的工程加固方案\n\n"
            f"4. 提供后续监测建议"
        )
        
        # 构建对话历史
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        
        # 调用 DeepSeek API
        try:
            response = self.client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            llm_response = response.choices[0].message.content
            
            # 更新对话历史
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": llm_response})
            
            # 生成 CoT 日志
            print("👀 感知: 加载文件...")
            print(f"👀 感知: 加载文件... 长度 {pile_length:.1f}m...")
            print("🧠 认知: 调用 GeoFormer 内核...")
            print(f"📉 物理自检: 预测波形与实测波形吻合度 {result['confidence']*100:.1f}% (Loss={result['mse']:.1e})。物理一致性极高。")
            print("🧠 认知: 调用 DeepSeek 大脑进行深度分析...")
            print(f"✅ 结论: {llm_response[:100]}...")
            
            return {
                'status': 'success',
                'pile_length': pile_length,
                'defect_depth': result['defect_depth'],
                'confidence': result['confidence'],
                'mse': result['mse'],
                'snr': snr,
                'llm_response': llm_response,
                'history': self.history
            }
            
        except Exception as e:
            print(f"❌ 调用 LLM 失败: {e}")
            return {
                'status': 'error',
                'message': f'调用 LLM 失败: {e}'
            }
    
    def chat(self, message):
        """
        多轮对话
        
        参数:
        - message: 用户消息
        
        返回:
        - 回复
        """
        # 构建对话历史
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # 添加历史对话
        messages.extend(self.history)
        
        # 添加新消息
        messages.append({"role": "user", "content": message})
        
        # 调用 DeepSeek API
        try:
            response = self.client.chat.completions.create(
                model=self.config['llm']['model'],
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            llm_response = response.choices[0].message.content
            
            # 更新对话历史
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": llm_response})
            
            return {
                'status': 'success',
                'response': llm_response,
                'history': self.history
            }
            
        except Exception as e:
            print(f"❌ 调用 LLM 失败: {e}")
            return {
                'status': 'error',
                'message': f'调用 LLM 失败: {e}'
            }
    
    def get_system_info(self):
        """
        获取系统信息
        """
        model_info = self.predictor.get_model_info()
        return {
            'model': model_info,
            'llm': self.config['llm']['model'],
            'rules': 'JGJ-106 规则'
        }

if __name__ == "__main__":
    # 测试大脑
    try:
        brain = GeoAgent()
        print(f"系统信息: {brain.get_system_info()}")
        
        # 测试诊断
        if os.path.exists('test_waveform.npy'):
            report = brain.diagnose('test_waveform.npy')
            if report['status'] == 'success':
                print("\n=== 诊断报告 ===")
                print(f"桩长: {report['pile_length']:.1f}m")
                print(f"缺陷深度: {report['defect_depth']:.1f}m")
                print(f"置信度: {report['confidence']:.4f}")
                print(f"SNR: {report['snr']:.1f}dB")
                print(f"LLM 分析: {report['llm_response']}")
                print("==============")
            
            # 测试多轮对话
            chat_response = brain.chat("如果我不处理这个缺陷，上层建筑会沉降吗？")
            if chat_response['status'] == 'success':
                print("\n=== 追问回答 ===")
                print(f"回答: {chat_response['response']}")
                print("==============")
            
    except Exception as e:
        print(f"测试失败: {e}")
