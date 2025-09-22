# Memory-Aware RLHF: 过度拒答现象分析 🧠

## 🎯 研究目标 (Research Objectives)

本研究专注于分析**RLHF在交互式记忆任务中的过度拒答现象**，主要解决以下核心问题：

1. **🚫 过度拒答检测 (Over-Refusal Detection)** - 验证RLHF模型是否在有明确证据时仍然错误拒答
2. **📊 拒答边界分析 (Refusal Boundary Analysis)** - 量化RLHF导致的拒答行为偏移
3. **🔍 统计显著性验证** - 通过严格的统计检验证明过度拒答现象的存在

## 🔬 研究问题 (Research Questions)

### **RQ2: RLHF过度保守性分析** ⭐ *[已完成实现]*
- **问题**: RLHF是否在可回答的记忆检索场景中过于保守，导致错误拒答？
- **方法**: 对比Base模型vs RLHF模型在IE(Information Extraction)和ABS(Abstention)子集上的拒答行为
- **指标**: 
  - **ORR (Over-Refusal Rate)**: IE子集上的拒答率
  - **ABS-LegitRefuse**: ABS子集上的合法拒答率  
  - **McNemar统计检验**: 配对比较的显著性检验
- **状态**: ✅ **完整实现，可直接运行**

### **RQ4: RLHF更新一致性分析** 
- **问题**: RLHF是否在知识更新场景中缺乏一致性？
- **状态**: 🚧 *待实现*

## 📁 项目结构

```
memory-aware_RLHF/
├── 📊 data/                     # 数据处理和统计分析
│   ├── longmemeval_loader.py    # LongMemEval数据集加载器 ✅
│   └── analyze_dataset_stats.py # 数据集统计分析工具 ✅
├── 🧪 experiments/              # RQ2实验核心模块
│   ├── config.py               # 模型和实验配置 ✅
│   └── rq2_over_refusal.py     # RQ2实验引擎 ✅
├── 🚀 run_rq2_experiment.py    # 实验启动器 ✅
├── 🧩 tune_refusal_detector.py # 拒答检测器调优 ✅
├── 🧪 test_rq2_environment.py   # 环境检查和验证 ✅
└── 📁 results/                 # 实验结果输出目录
```

## ⚡ 快速开始

### **🔧 环境要求**
- **Python**: 3.8+
- **GPU**: RTX 4070 (12GB VRAM) 或更高
- **系统内存**: 16GB+
- **平台**: Windows + WSL2 (推荐)

### **📦 安装依赖**
```bash
# 1. 激活环境
conda activate MARLHF

# 2. 自动安装核心依赖
python install_core_deps.py

# 3. 验证环境
python test_rq2_environment.py --framework-only
```

### **📂 数据集设置**
将LongMemEval数据集放置在项目目录下：
```
memory-aware_RLHF/
└── data/
    └── longmemeval_data/
        └── longmemeval_oracle.json  # 核心数据文件
```

### **🎯 运行RQ2实验**

#### **选项1: 快速验证实验 (推荐新手)**
```bash
# 运行10个样本的快速测试 (~5-10分钟)
python run_rq2_experiment.py --model-pair qwen2.5-3b --quick-test
```

#### **选项2: 完整单模型实验**
```bash
# Qwen2.5-3B完整实验 (~2-3小时)
python run_rq2_experiment.py --model-pair qwen2.5-3b

# Llama-3.2-3B完整实验
python run_rq2_experiment.py --model-pair llama3.2-3b

# Mistral-7B完整实验 (需要更多VRAM)
python run_rq2_experiment.py --model-pair mistral-7b
```

#### **选项3: 多模型综合对比**
```bash
# 运行所有模型对的综合实验 (~6-8小时)
python run_rq2_experiment.py --comprehensive
```

## 📊 实验输出说明

### **✅ 成功的实验输出示例**
```
📊 RQ2实验结果摘要: RLHF过度拒答现象分析
================================================================
🏷️  实验配置:
   基础模型: Qwen/Qwen2.5-3B
   RLHF模型: Qwen/Qwen2.5-3B-Instruct  
   IE实例数: 150 (应该回答)
   ABS实例数: 30 (应该拒答)

📈 ORR (Over-Refusal Rate) 分析:
   Base模型 IE拒答率: 5.3% (8/150)
   RLHF模型 IE拒答率: 12.7% (19/150)  
   拒答率变化: +7.4% (RLHF更保守)

🚫 ABS (Abstention) 合法拒答分析:
   Base模型 ABS拒答率: 76.7% (23/30)
   RLHF模型 ABS拒答率: 86.7% (26/30)
   合法拒答率变化: +10.0%

📊 统计显著性检验 (McNemar Test):
   检验统计量: 4.167
   P值: 0.041
   是否显著 (p<0.05): 是
   结论: RLHF显著更保守

🎯 RQ2核心发现:
   RLHF模型在IE上拒答率增加 7.4%
   过度拒答证据: 发现
   RLHF在ABS上合法拒答率: 86.7%
```

### **📁 输出文件结构**
```
results/rq2_qwen2.5_3b/
├── rq2_base_ie_responses_20240922_143020.json    # Base模型IE响应
├── rq2_rlhf_ie_responses_20240922_143020.json    # RLHF模型IE响应  
├── rq2_base_abs_responses_20240922_143020.json   # Base模型ABS响应
├── rq2_rlhf_abs_responses_20240922_143020.json   # RLHF模型ABS响应
└── rq2_analysis_20240922_143020.json             # 完整统计分析结果
```

## 🛠️ 高级功能

### **📈 数据集统计分析**
```bash
# 查看LongMemEval数据集的详细统计信息
python data/analyze_dataset_stats.py
```

### **🔧 拒答检测器调优**
```bash
# 在ABS子集上测试和调优拒答检测算法
python tune_refusal_detector.py --model base --num-test 50
```

### **🧪 环境诊断**
```bash
# 检查完整的实验环境
python test_rq2_environment.py

# 仅检查配置
python test_rq2_environment.py --config-only
```

## 🎯 核心实验结论预期

基于初步实验，RQ2实验预期能验证以下假设：

1. **✅ 过度拒答现象存在**: RLHF模型在IE子集(有证据问题)上显著高于Base模型的拒答率
2. **✅ 统计显著性**: McNemar检验p<0.05，证明差异具有统计显著性  
3. **✅ 合法拒答能力保持**: RLHF在ABS子集(无证据问题)上拒答率仍然较高，说明其拒答能力本身是合理的

## 🚨 常见问题排除

### **GPU内存不足**
```bash
# 如果出现CUDA OOM，尝试更小的模型
python run_rq2_experiment.py --model-pair qwen2.5-3b --quick-test
```

### **模型下载失败**  
```bash
# 检查网络连接，或使用代理
export HF_ENDPOINT=https://hf-mirror.com
python run_rq2_experiment.py --model-pair qwen2.5-3b --quick-test
```

### **实验中断后恢复**
```bash
# 检查results/目录下的部分结果
ls results/rq2_*/
```
