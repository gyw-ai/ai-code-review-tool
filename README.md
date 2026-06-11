# Topology Experiment Platform

## 快速开始

```bash
# 1. 查看可用本地模型
python run_experiment.py --list-models

# 2. 文件审查（CPU可用）
python run_experiment.py --review demo.py

# 3. 路径对比
python run_experiment.py --compare src/agents/gpt_agent.py
```

## GPU 加速流水线

连接显卡后，完整运行：

```bash
# 阶段一：教师自动标注（100个源文件）
python run_experiment.py --label 100

# 阶段二：蒸馏训练
python run_experiment.py --train teacher_labels.json --epochs 10

# 阶段三：对比验证
python run_experiment.py --compare src/core/result_merger.py
```

## 架构说明

```
src/topology_experiment/
├── paths/
│   ├── cnn_path.py          1D Conv → 局部模式
│   ├── rnn_path.py          BiLSTM → 执行流
│   ├── transformer_path.py  Ollama LLM → 全局上下文
│   └── gnn_path.py          GCN on AST → 结构关系
├── router.py                拓扑路由网关
├── fusion.py                Cross-attention 融合层
├── experiment.py            实验编排器
└── data/
    ├── labeling.py          教师标注 + 蒸馏训练
    └── code_parser.py       代码解析 + AST 构建
```

## CUDA 支持

代码自动检测 GPU，无需手动指定设备。

## Ollama 要求

GPU 机器上建议使用更大的模型以提高标签质量：

```bash
ollama pull deepseek-coder-v2:16b
# 或
ollama pull qwen3.6:27b
```

然后在运行前设置环境变量：

```bash
set OLLAMA_MODEL=deepseek-coder-v2:16b
```
