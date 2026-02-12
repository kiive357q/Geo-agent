#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geo-Agent Dingo-Phoenix 统一入口
"""

import argparse
import os
import numpy as np
import torch
from dingo_core.engine.trainer import Trainer
from geo_agent.core.brain import GeoBrain
from dingo_core.dataset.dingo_dataset import DingoDataModule

def train(args):
    """
    训练模型
    """
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
        'wave_eq_weight': args.wave_eq_weight
    }
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train()

def infer(args):
    """
    推理预测
    """
    # 创建 Geo Brain
    agent = GeoBrain()
    
    # 加载测试波形数据
    if os.path.exists(args.waveform_path):
        # 诊断
        report = agent.diagnose(args.waveform_path)
        
        if report['status'] == 'success':
            print("\n=== 诊断报告 ===")
            print(f"桩长: {report['pile_length']:.1f}m")
            print(f"缺陷深度: {report['defect_depth']:.1f}m")
            print(f"置信度: {report['confidence']:.4f}")
            print(f"完整性等级: {report['integrity_level']}")
            print(f"描述: {report['description']}")
            print(f"结论: {report['conclusion']}")
            print(f"建议: {report['recommendation']}")
            print(f"地质类型: {report['geo_type']}")
            print("==============\n")
        else:
            print(f"错误: {report['message']}")
        
        # 保存报告
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if report['status'] == 'success':
                    f.write("=== 诊断报告 ===\n")
                    f.write(f"桩长: {report['pile_length']:.1f}m\n")
                    f.write(f"缺陷深度: {report['defect_depth']:.1f}m\n")
                    f.write(f"置信度: {report['confidence']:.4f}\n")
                    f.write(f"完整性等级: {report['integrity_level']}\n")
                    f.write(f"描述: {report['description']}\n")
                    f.write(f"结论: {report['conclusion']}\n")
                    f.write(f"建议: {report['recommendation']}\n")
                    f.write(f"地质类型: {report['geo_type']}\n")
                    f.write("==============\n")
                else:
                    f.write(f"错误: {report['message']}\n")
            print(f"Report saved to: {args.output}")
    else:
        print(f"Waveform file not found: {args.waveform_path}")

def test_dataset(args):
    """
    测试数据集
    """
    # 测试 DataModule
    data_module = DingoDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_val_split=args.train_val_split
    )
    
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # 测试数据加载器
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"Batch wave shape: {batch['wave'].shape}")
        if 'length' in batch:
            print(f"Batch length shape: {batch['length'].shape}")
        if 'dt' in batch:
            print(f"Batch dt shape: {batch['dt'].shape}")
        if 'label' in batch:
            print(f"Batch label shape: {batch['label'].shape}")

def chat(args):
    """
    CLI 交互模式
    """
    print("🚀 Geo-Agent 对话模式启动")
    print("=====================================")
    print("欢迎使用 Geo-Agent！我是一个具备 RAG 和 CoT 能力的对话式 Agent。")
    print("我可以帮助您分析桩身完整性，检测可能的缺陷。")
    print("请输入波形文件路径，或输入 'exit' 退出。")
    print("=====================================")
    
    # 创建 Geo Brain
    try:
        agent = GeoBrain()
        print(f"✅ 系统就绪: {agent.get_system_info()['model']}")
        print(f"✅ 规则引擎: {agent.get_system_info()['rules']}")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 交互循环
    while True:
        try:
            # 获取用户输入
            file_path = input("\n请输入波形文件路径: ").strip()
            
            # 检查是否退出
            if file_path.lower() == 'exit':
                print("👋 再见！")
                break
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
            
            # 诊断
            report = agent.diagnose(file_path)
            
            if report['status'] == 'success':
                print("\n=== 诊断报告 ===")
                print(f"桩长: {report['pile_length']:.1f}m")
                print(f"缺陷深度: {report['defect_depth']:.1f}m")
                print(f"置信度: {report['confidence']:.4f}")
                print(f"完整性等级: {report['integrity_level']}")
                print(f"描述: {report['description']}")
                print(f"结论: {report['conclusion']}")
                print(f"建议: {report['recommendation']}")
                print(f"地质类型: {report['geo_type']}")
                print("==============")
            else:
                print(f"❌ 诊断失败: {report['message']}")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            continue

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Geo-Agent Dingo-Phoenix')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--max_length', type=int, default=1024, help='Max waveform length')
    train_parser.add_argument('--train_val_split', type=float, default=0.8, help='Train/val split ratio')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    train_parser.add_argument('--wave_eq_weight', type=float, default=0.1, help='Wave equation loss weight')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='Inference with the model')
    infer_parser.add_argument('--waveform_path', type=str, required=True, help='Waveform file path')
    infer_parser.add_argument('--output', type=str, help='Output report path')
    
    # 对话模式命令
    chat_parser = subparsers.add_parser('chat', help='CLI interactive mode')
    
    # 测试数据集命令
    test_parser = subparsers.add_parser('test_dataset', help='Test the dataset')
    test_parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    test_parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    test_parser.add_argument('--max_length', type=int, default=1024, help='Max waveform length')
    test_parser.add_argument('--train_val_split', type=float, default=0.8, help='Train/val split ratio')
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行命令
    if args.command == 'train':
        train(args)
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'chat':
        chat(args)
    elif args.command == 'test_dataset':
        test_dataset(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
