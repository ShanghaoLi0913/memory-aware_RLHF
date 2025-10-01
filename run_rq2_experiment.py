#!/usr/bin/env python3
"""
RQ2实验启动器: RLHF过度拒答现象分析
==========================================

功能概述:
    本脚本是RQ2实验的用户友好前端界面，用于运行Base vs RLHF模型的
    拒答行为对比实验。支持灵活的模型选择和分层实验策略。

🎯 核心研究问题 (RQ2):
    RLHF是否在可回答的记忆检索场景中过于保守，导致错误拒答？

🔬 实验方法:
    - IE子集(Information Extraction): 150个有证据的问题，应该回答
    - ABS子集(Abstention): 30个无证据的问题，应该拒答  
    - 计算ORR(Over-Refusal Rate)和统计显著性(McNemar检验)

📊 支持的模型配置:

1. **qwen2.5-3b**: Qwen2.5-3B Base vs Instruct (推荐RTX 4070)
2. **llama3.2-3b**: Llama-3.2-3B Base vs Instruct  
3. **mistral-7b**: Mistral-7B-v0.3 Base vs Instruct
4. **long_context**: 多模型综合对比实验
5. **cloud_full_context**: 云端28K完整上下文实验

🚀 快速开始命令:

## 推荐流程: 分层实验策略

### 阶段1: 本地RTX 4070验证 (8K上下文，71.6%数据覆盖)
```bash
# 1. 环境检查
python test_rq2_environment.py

# 2. 快速验证实验逻辑 (10个样本, ~5-10分钟)  
python run_rq2_experiment.py --model-pair qwen2.5-3b --quick-test

# 3. 完整单模型实验 (~2-3小时)
python run_rq2_experiment.py --model-pair qwen2.5-3b

# 4. 多模型对比实验 (~6-8小时)
python run_rq2_experiment.py --comprehensive
```

### 阶段2: 云端RTX 4090完整实验 (28K上下文，100%数据覆盖) 
```bash
# 云端完整上下文实验 - 覆盖所有27K最长数据
python run_rq2_experiment.py --config cloud_full_context
```

🔧 所有支持的命令行选项:

```bash
# 基础实验命令
python run_rq2_experiment.py --model-pair MODEL_NAME [--quick-test]
python run_rq2_experiment.py --config CONFIG_NAME
python run_rq2_experiment.py --comprehensive

# 自定义模型对 (高级用法)
python run_rq2_experiment.py --model-pair "base_model,instruct_model"

# 实用工具命令
python run_rq2_experiment.py --list-configs    # 查看所有可用配置
python run_rq2_experiment.py --test-only       # 仅测试环境，不运行模型
```

📋 命令行参数详解:

--model-pair, -m:  指定模型对
  格式1: 预设配置名 (如 'qwen2.5-3b', 'llama3.2-3b', 'mistral-7b')
  格式2: 自定义模型对 ('base_model,instruct_model')
  
--config, -c:      指定实验配置名称
  可选: qwen2.5-3b, llama3.2-3b, mistral-7b, long_context, cloud_full_context
  
--quick-test:      快速测试模式，限制10个IE实例 + 3个ABS实例
--comprehensive:   运行所有3个模型对的综合对比实验
--test-only:       仅运行环境测试，不执行实际模型推理
--list-configs:    列出所有可用的实验配置

⚙️ 硬件要求和性能优化:

RTX 4070 (12GB VRAM):
  ✅ Qwen2.5-3B: 最佳选择，稳定运行
  ✅ Llama-3.2-3B: 良好支持  
  ⚠️ Mistral-7B: 接近显存极限，建议云端运行

RTX 4090 (24GB VRAM):
  ✅ 所有模型: 完美支持 
  ✅ 28K完整上下文: 推荐配置

上下文长度策略:
  - 本地RTX 4070: 8K tokens (覆盖71.6%数据)
  - 云端RTX 4090: 28K tokens (覆盖100%数据，包括最长27K样本)

📊 预期实验输出:

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

📁 输出文件结构:
```
results/rq2_MODEL_NAME/
├── rq2_base_ie_responses_TIMESTAMP.json     # Base模型IE响应
├── rq2_rlhf_ie_responses_TIMESTAMP.json     # RLHF模型IE响应
├── rq2_base_abs_responses_TIMESTAMP.json    # Base模型ABS响应  
├── rq2_rlhf_abs_responses_TIMESTAMP.json    # RLHF模型ABS响应
└── rq2_analysis_TIMESTAMP.json              # 完整统计分析结果
```

🚨 故障排除:

CUDA OOM错误:
  → 使用更小模型: --model-pair qwen2.5-3b
  → 启用快速测试: --quick-test
  
模型下载失败:
  → 设置镜像: export HF_ENDPOINT=https://hf-mirror.com
  → 检查网络连接
  
实验中断:
  → 检查 results/ 目录下的部分结果
  → 重新运行会自动覆盖

💡 最佳实践:

1. 首次使用建议运行: python run_rq2_experiment.py --model-pair qwen2.5-3b --quick-test
2. 环境验证: python test_rq2_environment.py  
3. 监控GPU: nvidia-smi (实验过程中)
4. 学术发表: 使用云端28K完整实验结果

依赖文件:
    - experiments/rq2_over_refusal.py (核心实验引擎)
    - experiments/config.py (配置管理)  
    - data/longmemeval_loader.py (数据加载)
    - utils/refusal_detector.py (拒答检测算法)
"""
import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.config import get_rq2_config, get_model_config, list_available_configs
from experiments.rq2_over_refusal import RQ2ExperimentConfig, run_rq2_experiment


def create_rq2_config_from_preset(preset_name: str) -> RQ2ExperimentConfig:
    """从预设配置创建RQ2实验配置"""
    preset = get_rq2_config(preset_name)
    
    # 获取模型配置
    if "model_pairs" in preset:
        # 多模型对比实验，使用第一对作为默认
        base_model_name, rlhf_model_name = preset["model_pairs"][0]
    else:
        base_model_name = preset["base_model"]
        rlhf_model_name = preset["rlhf_model"]
    
    base_model = get_model_config(base_model_name)
    rlhf_model = get_model_config(rlhf_model_name)
    
    # 构建配置
    config = RQ2ExperimentConfig(
        base_model_name=base_model.path,
        rlhf_model_name=rlhf_model.path,
        longmemeval_path=f"{preset['data_config'].longmemeval_path}/longmemeval_{preset['data_config'].dataset_variant}.json",
        max_sessions=preset['data_config'].max_sessions,
        max_tokens=preset['data_config'].max_context_length,
        temperature=preset['generation_config'].temperature,
        top_p=preset['generation_config'].top_p,
        max_new_tokens=preset['generation_config'].max_new_tokens,
        output_dir=preset['experiment_config'].output_dir,
        save_responses=preset['experiment_config'].save_responses
    )
    
    return config


def run_comprehensive_rq2():
    """运行综合RQ2实验（多个模型对比）"""
    print("🚀 开始综合RQ2实验...")
    
    preset = get_rq2_config("comprehensive")
    model_pairs = preset["model_pairs"]
    
    all_results = {}
    
    for i, (base_model_name, rlhf_model_name) in enumerate(model_pairs, 1):
        print(f"\n{'='*60}")
        print(f"实验 {i}/{len(model_pairs)}: {base_model_name} vs {rlhf_model_name}")
        print(f"{'='*60}")
        
        try:
            # 创建配置
            base_model = get_model_config(base_model_name)
            rlhf_model = get_model_config(rlhf_model_name)
            
            config = RQ2ExperimentConfig(
                base_model_name=base_model.path,
                rlhf_model_name=rlhf_model.path,
                longmemeval_path=f"{preset['data_config'].longmemeval_path}/longmemeval_{preset['data_config'].dataset_variant}.json",
                max_sessions=preset['data_config'].max_sessions,
                max_tokens=preset['data_config'].max_context_length,
                temperature=preset['generation_config'].temperature,
                top_p=preset['generation_config'].top_p,
                max_new_tokens=preset['generation_config'].max_new_tokens,
                output_dir=f"{preset['experiment_config'].output_dir}/{base_model_name}_vs_{rlhf_model_name}",
                save_responses=preset['experiment_config'].save_responses
            )
            
            # 运行实验
            result = run_rq2_experiment(config)
            all_results[f"{base_model_name}_vs_{rlhf_model_name}"] = result
            
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            continue
    
    # 汇总所有结果
    print_comprehensive_summary(all_results)
    
    return all_results


def print_comprehensive_summary(all_results):
    """打印综合实验结果摘要"""
    print("\n" + "="*80)
    print("🏆 综合RQ2实验结果汇总")
    print("="*80)
    
    for pair_name, result in all_results.items():
        overall = result["overall_metrics"]
        conclusion = result["rq2_conclusion"]
        
        print(f"\n📊 {pair_name}:")
        print(f"  拒答率增加: {overall['refusal_rate_increase']:+.3f}")
        print(f"  置信度增加: {overall['confidence_increase']:+.3f}")
        print(f"  检测到过度拒答: {'是' if conclusion['over_refusal_detected'] else '否'}")
        print(f"  统计显著性: {'是' if conclusion['statistical_significance']['significant_at_0.05'] else '否'}")
    
    # 计算平均效应
    refusal_increases = [r["overall_metrics"]["refusal_rate_increase"] for r in all_results.values()]
    avg_increase = sum(refusal_increases) / len(refusal_increases)
    
    print(f"\n📈 平均拒答率增加: {avg_increase:+.3f}")
    
    over_refusal_count = sum(1 for r in all_results.values() if r["rq2_conclusion"]["over_refusal_detected"])
    print(f"🎯 检测到过度拒答的模型对: {over_refusal_count}/{len(all_results)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQ2实验: RLHF过度拒答现象分析")
    parser.add_argument("--config", "-c", default="qwen2.5-3b", 
                       help="实验配置名称 (qwen2.5-3b, llama3.2-3b, mistral-7b, long_context)")
    parser.add_argument("--test-only", "-t", action="store_true",
                       help="只运行框架测试，不执行实际实验")
    parser.add_argument("--list-configs", "-l", action="store_true",
                       help="列出所有可用配置")
    parser.add_argument("--model-pair", "-m",
                       help="指定模型对 (格式1: 预设配置名如'qwen2.5-3b'; 格式2: 自定义'base_model,instruct_model')")
    parser.add_argument("--quick-test", action="store_true",
                       help="快速测试模式 (只用前10个实例)")
    parser.add_argument("--comprehensive", action="store_true",
                       help="运行多模型综合实验")
    parser.add_argument("--abs-only", action="store_true",
                       help="仅评估ABS子集（应该拒答），忽略IE 子集")
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_available_configs()
        return
    
    if args.test_only:
        print("🧪 运行RQ2完整测试...")
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "test_rq2_complete.py"], capture_output=False)
        if result.returncode == 0:
            print("✅ 测试通过，可以运行完整实验")
        else:
            print("❌ 测试失败，请检查配置")
        return
    
    if args.comprehensive:
        run_comprehensive_rq2()
        return
    
    # 处理模型对参数
    if args.model_pair:
        # 检查是否是预设配置名还是自定义模型对
        if ',' in args.model_pair:
            # 自定义模型对格式: "base_model,instruct_model"
            base_model, instruct_model = args.model_pair.split(',')
            print(f"🎯 运行自定义模型对实验: {base_model} vs {instruct_model}")
            
            config = RQ2ExperimentConfig(
                base_model_name=base_model.strip(),
                rlhf_model_name=instruct_model.strip(),
                longmemeval_path="data/longmemeval_data/longmemeval_oracle.json",
                max_sessions=None if not args.quick_test else 5,
                output_dir=f"results/rq2_custom/{base_model}_{instruct_model}"
            )
        else:
            # 预设配置名格式: "qwen2.5-3b"
            preset_name = args.model_pair
            print(f"🎯 运行预设实验: {preset_name}")
            # 打印评审后端信息（当前使用 evaluate_qa.py 的 LLM-as-a-judge）
            print("🧪 评审后端: evaluate_qa.py (LLM-as-a-judge)")
            
            config = create_rq2_config_from_preset(preset_name)
            
            # 应用快速测试设置
            if args.quick_test:
                config.max_instances = 10
                config.max_sessions = 5
                # 更新输出目录名称
                config.output_dir = config.output_dir.replace("rq2_", "rq2_").rstrip("/") + "_quick_test"
                print(f"⚡ 快速测试模式：限制10个实例，5个会话")
            if args.abs_only:
                config.abs_only = True
                print("⚙️ 启用 --abs-only：只评估 ABS 子集")
        
        result = run_rq2_experiment(config)
        return
    
    # 运行单个配置的实验
    print(f"🚀 运行RQ2实验，配置: {args.config}")
    # 打印评审后端信息（当前使用 evaluate_qa.py 的 LLM-as-a-judge）
    print("🧪 评审后端: evaluate_qa.py (LLM-as-a-judge)")
    
    try:
        config = create_rq2_config_from_preset(args.config)
        
        # 快速测试模式修改配置
        if args.quick_test:
            print("⚡ 快速测试模式: 只处理前10个实例")
            config.max_instances = 10  # 限制实例数量
            config.max_sessions = 5   # 限制会话数量
            config.output_dir = f"{config.output_dir}_quick_test"
        if args.abs_only:
            config.abs_only = True
            print("⚙️ 启用 --abs-only：只评估 ABS 子集")
        
        result = run_rq2_experiment(config)
        
        print("\n🎉 实验完成！")
        
        # 简要结论
        conclusion = result["rq2_conclusion"]
        if conclusion["over_refusal_detected"]:
            print("🔍 实验结论: 检测到RLHF过度拒答现象")
        else:
            print("🔍 实验结论: 未检测到明显的过度拒答现象")
            
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
