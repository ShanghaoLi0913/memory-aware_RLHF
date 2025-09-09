#!/usr/bin/env python3
"""
RQ2实验完整测试套件
====================================

功能概述:
    集成所有RQ2实验相关的测试功能，包括框架测试、配置测试、数据验证等。
    提供一站式的实验前检查，确保所有组件正常工作。

主要测试模块:
1. **框架基础测试**
   - 数据加载和处理
   - 拒答检测算法
   - 提示生成逻辑

2. **配置和模型测试**
   - 实验配置加载
   - 模型配置验证
   - 数据集可用性检查

3. **实验就绪性检查**
   - 环境依赖验证
   - GPU可用性检测
   - 命令行使用指南

使用方法:
    ```bash
    # 完整测试
    python test_rq2_complete.py
    
    # 只测试特定模块
    python test_rq2_complete.py --framework-only
    python test_rq2_complete.py --config-only
    ```

注意: 这是一个轻量级测试，不会加载大模型，适合快速验证。
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 依赖检查
DEPS_AVAILABLE = True
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from tqdm import tqdm
except ImportError as e:
    print(f"⚠️ 缺少依赖包: {e}")
    DEPS_AVAILABLE = False


def test_environment():
    """测试环境配置"""
    print("🌍 环境配置测试...")
    
    # Python版本
    print(f"  Python版本: {sys.version}")
    
    # 依赖包检查
    if DEPS_AVAILABLE:
        print("  ✅ 核心依赖包可用")
        
        # GPU检查
        if torch.cuda.is_available():
            print(f"  ✅ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"  📊 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  ⚠️ GPU不可用，将使用CPU运行")
    else:
        print("  ❌ 缺少必要依赖包")
        return False
    
    return True


def test_data_loading():
    """测试数据加载"""
    print("\n📁 数据加载测试...")
    
    try:
        from data.longmemeval_loader import LongMemEvalLoader
        
        # 测试数据路径
        data_path = "/mnt/d/datasets/longmemeval_data/longmemeval_oracle.json"
        
        if not os.path.exists(data_path):
            print(f"  ❌ 数据文件不存在: {data_path}")
            return False
        
        # 加载数据
        loader = LongMemEvalLoader(data_path)
        all_instances = loader.load_data()
        rq2_instances = loader.get_rq2_instances()
        abs_instances = loader.get_abstention_instances()
        
        print(f"  ✅ 数据加载成功:")
        print(f"    总实例数: {len(all_instances)}")
        print(f"    RQ2实例数: {len(rq2_instances)}")
        print(f"    拒答实例数: {len(abs_instances)}")
        
        # 问题类型统计
        question_types = {}
        for instance in all_instances:
            q_type = instance.question_type
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        print(f"  📊 问题类型分布:")
        for q_type, count in sorted(question_types.items()):
            print(f"    {q_type}: {count}")
        
        # 测试格式化功能
        if rq2_instances:
            sample = rq2_instances[0]
            formatted = loader.format_conversation_history(sample, max_sessions=2)
            print(f"  📝 样例问题: {sample.question[:50]}...")
            print(f"    格式化历史长度: {len(formatted)} 字符")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return False


def test_refusal_detection():
    """测试拒答检测"""
    print("\n🔍 拒答检测测试...")
    
    if not DEPS_AVAILABLE:
        print("  ⚠️ 依赖不完整，跳过拒答检测测试")
        return False
    
    try:
        from utils.refusal_detector import RefusalDetector
        
        detector = RefusalDetector()
        
        # 测试案例
        test_cases = [
            ("I don't know the answer to that question.", True),    # 明确拒答
            ("I'm not sure about this information.", True),         # 不确定
            ("The weather is sunny today.", False),                 # 正常回答
            ("Based on the data, the result is 42.", False),       # 正常回答
            ("I cannot find this information.", True),              # 拒答
            ("There is no mention of this in the context.", True), # 拒答
        ]
        
        correct = 0
        total = len(test_cases)
        
        print(f"  📋 测试 {total} 个案例:")
        for i, (text, expected) in enumerate(test_cases, 1):
            is_refusal, confidence = detector.detect_refusal(text)
            is_correct = (is_refusal == expected)
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"    {i}. {status} 检测: {is_refusal}, 置信度: {confidence:.3f}")
            print(f"       文本: {text[:40]}...")
        
        accuracy = correct / total
        print(f"  🎯 准确率: {accuracy:.1%} ({correct}/{total})")
        
        return accuracy > 0.7
        
    except Exception as e:
        print(f"  ❌ 拒答检测测试失败: {e}")
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n⚙️ 配置加载测试...")
    
    try:
        from experiments.config import get_rq2_config, list_available_configs, MODELS
        
        # 测试模型配置
        print(f"  📊 可用模型数量: {len(MODELS)}")
        for model_name, model_config in MODELS.items():
            print(f"    {model_name}: {model_config.path}")
        
        # 测试RQ2配置
        configs_to_test = ["qwen2.5-3b", "llama3.2-3b", "mistral-7b", "long_context"]
        success_count = 0
        
        for config_name in configs_to_test:
            try:
                config = get_rq2_config(config_name)
                print(f"  ✅ {config_name}: {config['description']}")
                print(f"    模型对数: {len(config['model_pairs'])}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ {config_name}: {e}")
        
        print(f"  🎯 配置成功率: {success_count}/{len(configs_to_test)}")
        return success_count == len(configs_to_test)
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False


def test_prompt_generation():
    """测试提示生成"""
    print("\n📝 提示生成测试...")
    
    try:
        from data.longmemeval_loader import LongMemEvalLoader
        
        data_path = "/mnt/d/datasets/longmemeval_data/longmemeval_oracle.json"
        if not os.path.exists(data_path):
            print("  ⚠️ 跳过：数据文件不存在")
            return True
        
        loader = LongMemEvalLoader(data_path)
        rq2_instances = loader.get_rq2_instances()
        
        if not rq2_instances:
            print("  ⚠️ 跳过：没有RQ2实例")
            return True
        
        # 测试提示生成
        sample = rq2_instances[0]
        
        # 测试不同的会话数限制
        for max_sessions in [2, 5, None]:
            history = loader.format_conversation_history(sample, max_sessions)
            
            # 简单的提示模板
            prompt = f"""Based on the conversation history, please answer: {sample.question}

Conversation History:
{history}

Answer:"""
            
            prompt_len = len(prompt)
            print(f"  📏 max_sessions={max_sessions}: {prompt_len} 字符")
            
            if max_sessions == 2:
                print(f"    样例片段: {prompt[:100]}...")
        
        print("  ✅ 提示生成正常")
        return True
        
    except Exception as e:
        print(f"  ❌ 提示生成测试失败: {e}")
        return False


def show_usage_guide():
    """显示使用指南"""
    print("\n" + "="*60)
    print("📖 RQ2实验使用指南")
    print("="*60)
    
    print("""
🚀 推荐的实验流程:

1. 快速测试 Qwen2.5-3B:
   python run_rq2_experiment.py --config qwen2.5-3b --quick-test

2. 完整 Qwen2.5-3B 实验:
   python run_rq2_experiment.py --config qwen2.5-3b

3. 测试其他模型:
   python run_rq2_experiment.py --config llama3.2-3b
   python run_rq2_experiment.py --config mistral-7b  # 云上运行

4. 综合实验 (所有模型):
   python run_rq2_experiment.py --comprehensive

5. 自定义模型对:
   python run_rq2_experiment.py --model-pair "Qwen/Qwen2.5-3B,Qwen/Qwen2.5-3B-Instruct"

📋 其他有用命令:

- 查看配置: python run_rq2_experiment.py --list-configs
- 框架测试: python run_rq2_experiment.py --test-only
- 查看帮助: python run_rq2_experiment.py --help
""")


def run_complete_test():
    """运行完整测试"""
    print("🧪 RQ2实验完整测试套件")
    print("="*50)
    
    test_results = []
    
    # 1. 环境测试
    test_results.append(("环境配置", test_environment()))
    
    # 2. 数据加载测试
    test_results.append(("数据加载", test_data_loading()))
    
    # 3. 拒答检测测试
    test_results.append(("拒答检测", test_refusal_detection()))
    
    # 4. 配置加载测试
    test_results.append(("配置加载", test_config_loading()))
    
    # 5. 提示生成测试
    test_results.append(("提示生成", test_prompt_generation()))
    
    # 总结结果
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体通过率: {passed}/{total} ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 所有测试通过！RQ2实验已准备就绪。")
        show_usage_guide()
        return True
    else:
        print(f"\n⚠️ {total-passed} 项测试失败，请检查配置后重试。")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQ2实验完整测试套件")
    parser.add_argument("--framework-only", action="store_true",
                       help="只运行框架测试（数据、拒答检测、提示生成）")
    parser.add_argument("--config-only", action="store_true",
                       help="只运行配置测试（环境、配置加载）")
    parser.add_argument("--usage", action="store_true",
                       help="只显示使用指南")
    
    args = parser.parse_args()
    
    if args.usage:
        show_usage_guide()
        return
    
    if args.framework_only:
        print("🔧 运行框架测试...")
        results = [
            test_data_loading(),
            test_refusal_detection(),
            test_prompt_generation()
        ]
        success = all(results)
    elif args.config_only:
        print("⚙️ 运行配置测试...")
        results = [
            test_environment(),
            test_config_loading()
        ]
        success = all(results)
    else:
        success = run_complete_test()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
