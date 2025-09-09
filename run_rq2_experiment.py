#!/usr/bin/env python3
"""
RQ2实验启动器: RLHF过度拒答现象分析
==========================================

功能概述:
    本脚本是RQ2实验的用户友好前端界面，提供命令行接口来运行各种
    预配置的实验。它连接用户输入与核心实验引擎(rq2_over_refusal.py)，
    简化了复杂实验的启动和管理。

主要功能:

1. **命令行界面管理**
   - 解析用户输入的实验参数
   - 提供多种预设实验配置
   - 支持灵活的实验定制选项

2. **配置管理**
   - create_rq2_config_from_preset(): 从预设创建实验配置
   - 支持的预设配置:
     * default: 标准7B模型对比实验
     * lightweight: 轻量级快速实验 (较少会话数)
     * comprehensive: 多模型全面对比实验

3. **实验启动模式**
   - 单配置实验: 运行指定的预设配置
   - 多模型综合实验: 批量对比多对模型
   - 测试模式: 验证环境和配置而不运行大模型
   - 配置列表: 显示所有可用的实验配置

4. **实验执行与管理**
   - run_comprehensive_rq2(): 执行多模型对比实验
   - print_comprehensive_summary(): 汇总多实验结果
   - 自动结果收集和错误处理
   - 实验进度显示和状态报告

5. **结果汇总与报告**
   - 单实验结果展示
   - 多实验对比分析
   - 统计显著性汇总
   - 实验结论自动生成

命令行选项:
    --config, -c [配置名]     指定实验配置 (default/lightweight/comprehensive)
    --test-only, -t          仅运行测试，不执行实际模型推理
    --list-configs, -l       列出所有可用的实验配置
    --comprehensive          运行多模型综合对比实验

使用示例:
    ```bash
    # 运行默认实验 (Llama-2 7B基础vs聊天模型)
    python run_rq2_experiment.py --config default
    
    # 运行轻量级快速实验 (适合测试)
    python run_rq2_experiment.py --config lightweight
    
    # 运行多模型综合实验 (Llama-2 + Mistral)
    python run_rq2_experiment.py --comprehensive
    
    # 测试模式 (验证环境配置)
    python run_rq2_experiment.py --test-only
    
    # 查看所有可用配置
    python run_rq2_experiment.py --list-configs
    ```

支持的实验配置:

1. **default 配置**
   - 模型: Llama-2-7b-hf vs Llama-2-7b-chat-hf
   - 会话数: 20个最大会话
   - 生成参数: temperature=0.1, max_tokens=512
   - 适用: 标准研究实验

2. **lightweight 配置**
   - 模型: 同default
   - 会话数: 10个会话 (减少计算量)
   - 生成参数: max_tokens=256
   - 适用: 快速验证和测试

3. **comprehensive 配置**
   - 模型组合:
     * Llama-2-7b vs Llama-2-7b-chat
     * Llama-2-13b vs Llama-2-13b-chat  
     * Mistral-7B vs Mistral-7B-Instruct
   - 会话数: 30个会话
   - 生成参数: temperature=0.0 (更确定性)
   - 适用: 全面的学术研究

实验流程:
    1. 参数解析 → 解析命令行输入
    2. 配置加载 → 从config.py加载预设配置
    3. 环境检查 → 验证数据和模型可用性
    4. 实验执行 → 调用rq2_over_refusal.py执行核心逻辑
    5. 结果展示 → 显示实验结果和统计分析
    6. 文件保存 → 自动保存详细结果到results/目录

输出内容:
    - 实验进度实时显示
    - 拒答率对比结果
    - 统计显著性检验
    - 按问题类型的详细分析
    - 实验结论和建议

错误处理:
    - 模型加载失败时的graceful降级
    - 网络连接问题的重试机制
    - 实验中断后的状态恢复
    - 详细的错误信息和解决建议

依赖文件:
    - experiments/rq2_over_refusal.py (核心实验引擎)
    - experiments/config.py (配置管理)
    - data/longmemeval_loader.py (数据加载)

资源要求:
    - GPU内存: 推荐8GB+ (7B模型) / 16GB+ (13B模型)
    - 系统内存: 16GB+
    - 存储空间: 模型缓存需要20-50GB
    - 运行时间: 轻量级15-30分钟 / 综合实验2-6小时

注意事项:
    - 首次运行会自动下载模型 (需要网络连接)
    - 大模型推理占用大量GPU资源
    - 实验结果保存在results/目录下
    - 支持实验中断后的手动分析
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
                       help="指定模型对 (格式: base_model,instruct_model)")
    parser.add_argument("--quick-test", action="store_true",
                       help="快速测试模式 (只用前10个实例)")
    parser.add_argument("--comprehensive", action="store_true",
                       help="运行多模型综合实验")
    
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
    
    # 处理自定义模型对
    if args.model_pair:
        base_model, instruct_model = args.model_pair.split(',')
        print(f"🎯 运行自定义模型对实验: {base_model} vs {instruct_model}")
        
        config = RQ2ExperimentConfig(
            base_model_name=base_model.strip(),
            rlhf_model_name=instruct_model.strip(),
            longmemeval_path="/mnt/d/datasets/longmemeval_data/longmemeval_oracle.json",
            max_sessions=None if not args.quick_test else 5,
            output_dir=f"results/rq2_custom/{base_model}_{instruct_model}"
        )
        
        result = run_rq2_experiment(config)
        return
    
    # 运行单个配置的实验
    print(f"🚀 运行RQ2实验，配置: {args.config}")
    
    try:
        config = create_rq2_config_from_preset(args.config)
        
        # 快速测试模式修改配置
        if args.quick_test:
            print("⚡ 快速测试模式: 只处理前10个实例")
            config.max_instances = 10  # 限制实例数量
            config.max_sessions = 5   # 限制会话数量
            config.output_dir = f"{config.output_dir}_quick_test"
        
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
