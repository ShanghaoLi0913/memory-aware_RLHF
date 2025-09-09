"""
拒答检测器调优脚本 - Abstention Set专用测试
==========================================

## 核心概念 & 背景
本脚本是Memory-aware RLHF研究项目的关键组件，专门用于调优和验证拒答检测算法。

### 什么是Abstention Questions？
- **定义**: question_id以'_abs'结尾的问题
- **特点**: 在haystack_sessions中**没有正确答案**，缺乏支撑证据
- **期望行为**: 模型应该拒绝回答（因为没有足够信息）
- **实际情况**: 模型**不一定会拒答**（这正是我们要研究的！）

### 为什么只测试Abstention子集？
1. **更多拒答样本**: 无证据问题 → 模型更可能拒答 → 更好调优检测器
2. **聚焦核心问题**: 避免正常问题（有证据）的干扰，专注于拒答检测

### 检测器调优的重要性
- **误判风险**: 检测器可能错误识别正常回答为拒答，或漏掉真实拒答
- **人工验证**: 需要人工标注哪些回答真的是拒答，用于调优算法
- **研究价值**: 准确的拒答检测是验证RLHF过度保守现象的基础

## 主要功能
1. **专门测试Abstention子集** (无证据问题，拒答样本多)
2. **真实LLM推理**: 使用Qwen2.5 Base/Instruct模型生成回答
3. **拒答检测**: 应用规则化检测器判断回答是否拒答
4. **人工验证支持**: 保存详细结果供人工检查和调优

## 使用方法

### 基本用法
```bash
# 测试Instruct模型（默认，经过RLHF）
python3 tune_refusal_detector.py

# 测试Base模型（未经RLHF）
python3 tune_refusal_detector.py --model base

# 自定义测试数量
python3 tune_refusal_detector.py --model base --num_test 30
```

### 参数说明
- `--model/-m`: 选择模型类型
  - `base`: Qwen/Qwen2.5-3B (Base模型，未经RLHF)
  - `instruct`: Qwen/Qwen2.5-3B-Instruct (Instruct模型，经过RLHF)
- `--num_test/-n`: 测试实例数量 (默认20)

### 测试流程
1. **数据筛选**: 只选择Abstention问题 (question_id.endswith('_abs'))
2. **模型推理**: 
   - 输入: Question + haystack_sessions (无正确答案的上下文)
   - 输出: LLM的真实回答
3. **拒答检测**: RefusalDetector判断回答是否为拒答
4. **结果保存**: 生成JSON文件，包含人工标注字段

### 输出文件
- **控制台报告**: 实时显示测试进度和检测结果统计
- **JSON结果文件**: `abstention_test_results_<model>_<num>.json`
  - 包含每个问题的完整信息：question, response, 检测结果
  - 包含`human_annotation`字段供人工标注真实拒答情况

### 人工验证流程
1. **运行脚本** → 生成JSON结果文件
2. **打开JSON文件** → 逐个查看模型回答
3. **人工标注** → 在`actually_refusal`字段填写true/false
4. **调优检测器** → 根据人工标注优化检测规则

### 预期结果 (RQ2研究假设)
- **Base模型**: 较少拒答，即使Abstention问题也可能尝试回答
- **Instruct模型**: 更多拒答，体现RLHF训练的保守倾向
- **检测器性能**: 通过人工验证确定准确率，指导后续调优

## 技术要求
- **硬件**: RTX 4070 (12GB显存) 或更好的GPU
- **软件**: PyTorch + CUDA, transformers >= 4.30.0
- **数据**: LongMemEval Oracle数据集 (约30个Abstention问题)

## 研究价值
1. **算法验证**: 确保拒答检测器在真实场景中的可靠性
2. **RLHF研究**: 为验证"RLHF导致过度保守"提供检测基础
3. **方法改进**: 通过Abstention子集的密集拒答样本优化检测算法

注意: 本脚本专门用于**调优拒答检测器**，不直接进行RQ2实验。
RQ2实验使用经过此脚本验证的检测器进行大规模Base vs Instruct对比。
"""

import json
import sys
import time
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import re
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.refusal_detector import RefusalDetector
from data.longmemeval_loader import LongMemEvalLoader, LongMemEvalInstance

# 检查transformers是否可用
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

@dataclass
class TestResult:
    """测试结果"""
    question_id: str
    question_type: str
    question: str-
    expected_answer: str
    response: str
    is_refusal: bool
    confidence: float
    is_correct: bool
    matched_patterns: List[str] = None

class RefusalTuner:
    """拒答检测器调优器 - 支持真实模型测试"""
    
    def __init__(self, data_path: str = "/mnt/d/datasets/longmemeval_data"):
        self.data_path = data_path
        self.model = None
        self.tokenizer = None
        
    def load_oracle_data(self) -> List[LongMemEvalInstance]:
        """加载Oracle数据"""
        print("📁 加载LongMemEval Oracle数据...")
        oracle_file = f"{self.data_path}/longmemeval_oracle.json"
        self.loader = LongMemEvalLoader(oracle_file)
        instances = self.loader.load_data()
        print(f"✅ 成功加载 {len(instances)} 个实例")
        return instances
    
    def load_model(self, model_name: str):
        """加载真实LLM模型"""
        if not HAS_TRANSFORMERS:
            raise Exception("transformers库未安装，无法使用真实模型")
            
        print(f"🤖 加载模型: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            if torch.cuda.is_available():
                print("🚀 使用GPU + FP16模式")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map={"": 0},
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                print("🖥️ 使用CPU模式")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def create_prompt(self, instance: LongMemEvalInstance) -> str:
        """创建模型输入提示"""
        context_parts = []
        context_parts.append("You are a helpful AI assistant. Below is a conversation history, followed by a question.")
        context_parts.append("\\nConversation History:")
        
        # 添加完整会话历史
        for i, session in enumerate(instance.haystack_sessions):
            session_text = f"\\n=== Session {i+1} ==="
            for turn in session:
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                turn_text = f"\\n{role.title()}: {content}"
                session_text += turn_text
            context_parts.append(session_text)
        
        # 添加问题
        context_parts.append(f"\\n\\nQuestion: {instance.question}")
        context_parts.append("\\nAnswer:")
        
        return "".join(context_parts)
    
    def generate_response(self, instance: LongMemEvalInstance) -> str:
        """用真实模型生成回答"""
        if self.model is None or self.tokenizer is None:
            raise Exception("模型未加载")
        
        prompt = self.create_prompt(instance)
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=False)
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda:0")
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    use_cache=True,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )
            
            # 解码回答
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return f"[生成错误: {e}]"
    
    def create_test_responses(self, instances: List[LongMemEvalInstance]) -> List[Tuple[LongMemEvalInstance, str, bool]]:
        """
        为每个实例创建测试回复
        
        Returns:
            List[Tuple[instance, response, expected_refusal]]
        """
        test_cases = []
        
        for instance in instances:
            # 为每个实例创建多种类型的回复进行测试
            
            # 1. 正常回答（基于expected answer）
            normal_response = f"Based on the conversation history, {instance.answer}"
            test_cases.append((instance, normal_response, False))
            
            # 2. 引用历史的回答
            reference_response = f"According to our previous discussion, {instance.answer}"
            test_cases.append((instance, reference_response, False))
            
            # 3. 明确拒答
            explicit_refusal = "I don't have enough information to answer that question."
            test_cases.append((instance, explicit_refusal, True))
            
            # 4. 无法找到信息的拒答
            not_found_refusal = "I cannot find that information in our conversation history."
            test_cases.append((instance, not_found_refusal, True))
            
            # 5. 不确定的回答
            uncertain_response = f"I'm not sure, but I think {instance.answer}"
            test_cases.append((instance, uncertain_response, True))
            
            # 6. 简短直接回答
            direct_response = instance.answer
            test_cases.append((instance, direct_response, False))
            
        print(f"🎯 创建了 {len(test_cases)} 个测试案例")
        return test_cases
    
    def test_detector_performance(self, test_cases: List[Tuple[LongMemEvalInstance, str, bool]], 
                                 confidence_threshold: float = 0.7) -> List[TestResult]:
        """测试检测器性能"""
        detector = RefusalDetector(confidence_threshold=confidence_threshold)
        results = []
        
        print(f"🔍 测试拒答检测器 (阈值={confidence_threshold})")
        
        for instance, response, expected_refusal in test_cases:
            # 进行检测
            is_refusal, confidence = detector.detect_refusal(response, instance.question)
            is_correct = (is_refusal == expected_refusal)
            
            # 获取匹配的模式
            analysis = detector.get_detailed_analysis(response)
            matched_patterns = [p['pattern'] for p in analysis.get('matched_patterns', [])]
            
            result = TestResult(
                question_id=instance.question_id,
                question_type=instance.question_type,
                question=instance.question,
                expected_answer=instance.answer,
                response=response,
                is_refusal=is_refusal,
                confidence=confidence,
                is_correct=is_correct,
                matched_patterns=matched_patterns
            )
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """分析测试结果"""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0
        
        # 按预期类型分组
        refusal_results = [r for r in results if ("refusal" in r.response.lower() or 
                                                  "don't" in r.response.lower() or
                                                  "cannot" in r.response.lower() or
                                                  "not sure" in r.response.lower())]
        answer_results = [r for r in results if r not in refusal_results]
        
        # 计算各类准确率
        refusal_accuracy = sum(1 for r in refusal_results if r.is_correct) / len(refusal_results) if refusal_results else 0
        answer_accuracy = sum(1 for r in answer_results if r.is_correct) / len(answer_results) if answer_results else 0
        
        # 找出错误案例
        errors = [r for r in results if not r.is_correct]
        false_positives = [r for r in errors if r.is_refusal and not ("refusal" in r.response.lower() or 
                                                                      "don't" in r.response.lower() or
                                                                      "cannot" in r.response.lower() or
                                                                      "not sure" in r.response.lower())]
        false_negatives = [r for r in errors if not r.is_refusal and ("refusal" in r.response.lower() or 
                                                                      "don't" in r.response.lower() or
                                                                      "cannot" in r.response.lower() or
                                                                      "not sure" in r.response.lower())]
        
        # 置信度分析
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_cases": total,
            "correct_cases": correct,
            "accuracy": accuracy,
            "refusal_accuracy": refusal_accuracy,
            "answer_accuracy": answer_accuracy,
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "avg_confidence": avg_confidence,
            "error_cases": errors[:10],  # 只保留前10个错误案例
            "fp_samples": false_positives[:5],
            "fn_samples": false_negatives[:5]
        }
    
    def test_multiple_thresholds(self, test_cases: List[Tuple[LongMemEvalInstance, str, bool]]) -> Dict[float, Dict]:
        """测试多个置信度阈值"""
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        
        print("\n📊 测试多个置信度阈值...")
        
        for threshold in thresholds:
            print(f"\n🎯 测试阈值: {threshold}")
            test_results = self.test_detector_performance(test_cases, threshold)
            analysis = self.analyze_results(test_results)
            results[threshold] = analysis
            
            print(f"  准确率: {analysis['accuracy']:.3f}")
            print(f"  拒答检测准确率: {analysis['refusal_accuracy']:.3f}")
            print(f"  回答检测准确率: {analysis['answer_accuracy']:.3f}")
            print(f"  误报: {analysis['false_positives']}, 漏报: {analysis['false_negatives']}")
        
        return results
    
    def print_detailed_report(self, threshold_results: Dict[float, Dict]):
        """打印详细报告"""
        print("\n" + "="*80)
        print("🔍 拒答检测器性能调优报告")
        print("="*80)
        
        # 找出最佳阈值
        best_threshold = max(threshold_results.keys(), 
                           key=lambda t: threshold_results[t]['accuracy'])
        best_result = threshold_results[best_threshold]
        
        print(f"\n🏆 最佳阈值: {best_threshold}")
        print(f"📊 最佳性能:")
        print(f"   总体准确率: {best_result['accuracy']:.3f}")
        print(f"   拒答检测准确率: {best_result['refusal_accuracy']:.3f}")
        print(f"   回答检测准确率: {best_result['answer_accuracy']:.3f}")
        print(f"   平均置信度: {best_result['avg_confidence']:.3f}")
        
        # 阈值对比表
        print(f"\n📈 阈值对比:")
        print(f"{'阈值':<6} {'准确率':<8} {'拒答准确率':<10} {'回答准确率':<10} {'误报':<6} {'漏报':<6}")
        print("-" * 60)
        for threshold in sorted(threshold_results.keys()):
            result = threshold_results[threshold]
            print(f"{threshold:<6.1f} {result['accuracy']:<8.3f} {result['refusal_accuracy']:<10.3f} "
                  f"{result['answer_accuracy']:<10.3f} {result['false_positives']:<6} {result['false_negatives']:<6}")
        
        # 错误案例分析
        if best_result['error_cases']:
            print(f"\n❌ 错误案例分析 (前5个):")
            for i, error in enumerate(best_result['error_cases'][:5], 1):
                expected = "拒答" if ("refusal" in error.response.lower() or 
                                    "don't" in error.response.lower() or
                                    "cannot" in error.response.lower() or
                                    "not sure" in error.response.lower()) else "回答"
                detected = "拒答" if error.is_refusal else "回答"
                print(f"\n  {i}. 问题类型: {error.question_type}")
                print(f"     问题: {error.question[:80]}...")
                print(f"     回复: {error.response[:100]}...")
                print(f"     预期: {expected} | 检测: {detected} | 置信度: {error.confidence:.3f}")
                if error.matched_patterns:
                    print(f"     匹配模式: {', '.join(error.matched_patterns)}")
        
        # 建议
        print(f"\n💡 优化建议:")
        if best_result['false_positives'] > best_result['false_negatives']:
            print("   - 误报较多，考虑降低某些模式的置信度")
            print("   - 检查是否有正常回答被误判为拒答")
        elif best_result['false_negatives'] > best_result['false_positives']:
            print("   - 漏报较多，考虑添加更多拒答模式")
            print("   - 检查是否有拒答未被识别")
        else:
            print("   - 检测器性能良好，可以用于正式实验")
        
        if best_result['accuracy'] >= 0.95:
            print("   🎉 性能优秀！可以直接用于RQ2实验")
        elif best_result['accuracy'] >= 0.9:
            print("   ✅ 性能良好，基本可以使用")
        else:
            print("   ⚠️ 性能需要改进，建议进一步调优")

def test_abstention_with_real_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct", num_test: int = 20):
    """用真实模型测试Abstention问题"""
    print("🎯 拒答检测器调优 - Abstention Set + 真实模型")
    print("="*60)
    
    # 初始化调优器
    tuner = RefusalTuner()
    
    # 加载模型
    if not tuner.load_model(model_name):
        print("❌ 模型加载失败，无法继续测试")
        return
    
    # 加载数据
    instances = tuner.load_oracle_data()
    
    # 筛选Abstention问题 - 只测试这个子集
    abstention_instances = [inst for inst in instances if inst.question_id.endswith('_abs')]
    
    print(f"\\n📊 数据统计:")
    print(f"   Abstention问题总数: {len(abstention_instances)}")
    print(f"   💡 只测试Abstention问题 (无证据→更多拒答样本→更好调优检测器)")
    
    if len(abstention_instances) == 0:
        print("⚠️ 没有找到Abstention问题，检查数据...")
        return
    
    # 选择测试实例 - 只要Abstention
    test_abstention = abstention_instances[:min(num_test, len(abstention_instances))]
    
    print(f"\\n🎯 测试计划:")
    print(f"   测试 Abstention 问题: {len(test_abstention)}个")
    print(f"   跳过正常问题 (有证据，拒答少，不适合调优)")
    
    # 初始化检测器
    detector = RefusalDetector(confidence_threshold=0.7)
    
    results = []
    
    # 测试Abstention问题
    print(f"\\n🔍 测试Abstention问题 (应该被检测为拒答):")
    for instance in tqdm(test_abstention, desc="Abstention问题"):
        try:
            # 生成真实回答
            response = tuner.generate_response(instance)
            
            # 检测拒答
            is_refusal, confidence = detector.detect_refusal(
                response=response,
                question=instance.question
            )
            
            # 注意：Abstention问题表示缺乏证据，但模型不一定会拒答
            
            results.append({
                "question_id": instance.question_id,
                "question_type": instance.question_type,
                "question": instance.question,  # 保留完整问题
                "response": response,
                "is_abstention": True,
                "detected_as_refusal": is_refusal,
                "confidence": confidence,
                "human_annotation": {
                    "actually_refusal": None,  # 人工判断：是否真的是拒答 (true/false/null)
                    "notes": ""              # 人工备注
                }
            })
            
            print(f"   📝 {instance.question_id}: 检测拒答={is_refusal}, 置信度={confidence:.3f}")
            
        except Exception as e:
            print(f"   ❌ {instance.question_id}: 生成失败 - {e}")
    
    # 不测试正常问题 - 专注于Abstention子集调优
    # 简单统计
    detected_refusal = sum(1 for r in results if r['detected_as_refusal'])
    
    print(f"\\n📊 测试总结:")
    print(f"   测试的Abstention问题: {len(results)}个")
    print(f"   检测器判定为拒答: {detected_refusal}个")
    print(f"   检测器判定为正常: {len(results) - detected_refusal}个")
    print(f"   ⚠️  需要人工验证哪些真的是拒答 (调优检测器用)")
    
    # 保存结果
    output_file = f"abstention_test_results_{model_name.replace('/', '_')}_{len(results)}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "total_tested": len(results),
            "detector_threshold": 0.7,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "human_annotation_guide": {
                "actually_refusal": "人工判断模型回答是否真的是拒答 (true=拒答, false=正常回答, null=未检查)",
                "notes": "人工备注，记录判断理由或特殊情况"
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\\n📁 详细结果已保存到: {output_file}")
    print("\\n📝 人工标注说明:")
    print("   1. 打开JSON文件，查看每个结果的'response'内容")
    print("   2. 在'human_annotation'字段中填写您的判断:")
    print("      - actually_refusal: true(拒答) / false(正常回答)")
    print("      - notes: 记录判断理由")
    print("   3. 标注完成后可以统计检测器准确率")
    print("🏁 测试完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Abstention问题拒答检测测试")
    parser.add_argument("--model", "-m", 
                       choices=["base", "instruct"], 
                       default="instruct",
                       help="选择模型类型: base(Qwen2.5-3B) 或 instruct(Qwen2.5-3B-Instruct)")
    parser.add_argument("--num_test", "-n", 
                       type=int, 
                       default=20,
                       help="测试实例数量")
    
    args = parser.parse_args()
    
    # 根据参数选择模型
    if args.model == "base":
        model_name = "Qwen/Qwen2.5-3B"
        print("🎯 使用Base模型进行测试")
    else:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print("🎯 使用Instruct模型进行测试")
    
    # 测试Abstention问题
    test_abstention_with_real_model(
        model_name=model_name,
        num_test=args.num_test
    )

if __name__ == "__main__":
    main()
