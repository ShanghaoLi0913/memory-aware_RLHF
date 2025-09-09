# Memory-Aware RLHF for Interactive Memory Tasks

## 研究目标 (Research Objectives)

本研究提出一个memory-aware的RLHF机制，使大语言模型在交互式记忆任务中能够：

1. **避免过度拒答 (over-refusal)** - 当上下文中已有明确证据时，模型应该回答而不是拒答
2. **保证更新一致性 (update consistency)** - 当历史事实被更新时，模型应该引用最新事实

## 研究问题 (Research Questions)

### RQ2: RLHF过度保守性分析
- **问题**: RLHF是否在可回答的记忆检索场景中过于保守，导致错误拒答？
- **目标**: 验证RLHF存在拒答边界右移的现象

### RQ4: RLHF更新一致性分析  
- **问题**: RLHF是否在知识更新场景中缺乏一致性，表现为更倾向于回答旧信息或拒答？
- **目标**: 验证RLHF的更新一致性缺陷

## 项目结构

```
memory-aware_RLHF/
├── data/                    # 数据集和测试用例
├── models/                  # 模型定义和训练代码
├── experiments/             # 实验脚本和配置
├── evaluation/              # 评估框架和指标
├── utils/                   # 工具函数
└── results/                 # 实验结果和分析
```

## 安装和使用

### 环境要求
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- WSL环境 (推荐)

### 数据集设置
请将LongMemEval数据集下载并解压到：
```
/mnt/d/datasets/longmemeval_data/
```

确保目录包含以下文件：
- longmemeval_oracle.json
- longmemeval_s.json  
- longmemeval_m.json

### 安装依赖
```bash
# 激活MARLHF环境
conda activate MARLHF

# 自动安装核心依赖
python install_core_deps.py

# 或手动安装
pip install -r requirements.txt
```

## 实验流程

1. **RQ2实验**: 测试RLHF在记忆检索任务中的过度拒答现象
2. **RQ4实验**: 测试RLHF在知识更新任务中的一致性问题
3. **Memory-Aware RLHF**: 实现改进的训练机制
4. **综合评估**: 对比改进前后的性能表现
