# Geo-Agent 智能桩基检测系统

## 项目简介

Geo-Agent 是一个以"领域物理计算"为大脑，以"LLM"为语音交互插件的垂直类工业智能体。该系统专为桩基检测设计，融合了物理信息神经网络（PINN）、强化学习（RL）熵权、Transformer-CNN 混合模型以及 JGJ-106 规则引擎，实现了对桩基质量的智能检测与评估。

## 系统架构

### 三层核心架构

1. **第一层：物理感知与检索层 (Perception & RAG Layer)**
   - 组件: OpenClaw + RAG + PINN/RL Entropy
   - 功能: 数据加载、预处理、熵权计算

2. **第二层：主体深度认知层 (Deep Cognitive Layer)**
   - 组件: Transformer + CNN 双流骨干网络
   - 功能: 多维特征提取与拟合

3. **第三层：符号执行与安全层 (Symbolic Skills Layer)**
   - 组件: JGJ-106 Rule Engine
   - 功能: 四分类判定与安全验证

## 核心功能

- **异构数据兼容**: 支持国内外不同格式的桩基检测数据
- **智能信号处理**: 自动去直流、重采样、归一化
- **物理信息融合**: 结合 PINN 与 RL 熵权计算
- **深度特征提取**: Transformer-CNN 双流网络架构
- **规范符合性**: 严格按照 JGJ-106 规范进行判定
- **批量处理能力**: 支持大规模数据的批处理分析
- **LLM 交互**: 提供自然语言交互接口

## 项目结构

```
Geo-Agent/
├── checkpoints/          # 模型训练 checkpoint
├── data/                 # 数据集
├── dingo_core/           # 核心算法模块
│   ├── dataset/          # 数据处理
│   ├── engine/           # 训练引擎
│   ├── modeling/         # 模型定义
│   └── physics/          # 物理计算
├── geo_agent/            # 智能体核心
│   ├── core/             # 大脑模块
│   ├── knowledge/        # 知识库
│   └── skills/           # 技能模块
├── create_test_waveform.py  # 测试波形生成
├── main.py               # 主入口
├── test_waveform.npy     # 测试波形文件
└── train_dingo.py        # 训练脚本
```

## 环境依赖

- Python 3.10+
- PyTorch 2.0+
- NumPy
- SciPy
- FastAPI
- Uvicorn
- DeepSeek API (可选，用于 LLM 交互)

## 安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

### 单文件测试

```bash
python main.py --file test_waveform.npy
```

### 批处理测试

```bash
python main.py --batch --directory data/
```

### 模型训练

```bash
python train_dingo.py --epochs 100 --batch_size 32
```

## 数据格式

支持以下数据格式：
- 标准 CSV 文件
- Dingo 格式 CSV 文件
- TXT 文件
- NPY 文件

## 输出结果

系统输出包括：
- 四分类结果 (I/II/III/IV)
- 临界波速
- 临界波幅衰减
- 功率谱密度
- 综合评分

## 技术特点

1. **物理信息神经网络 (PINN)**: 融合物理定律与数据驱动方法
2. **强化学习熵权**: 动态调整信号可信度
3. **Transformer-CNN 混合模型**: 同时捕获全局与局部特征
4. **JGJ-106 规则引擎**: 确保判定符合国家规范
5. **OpenClaw 数据适配器**: 统一处理不同格式数据

## 应用场景

- 建筑工程桩基质量检测
- 桥梁桩基健康评估
- 土木工程质量控制
- 桩基检测数据分析

## 许可证

本项目为私有项目，仅供内部使用。

## 联系信息

如有问题，请联系项目维护人员。
