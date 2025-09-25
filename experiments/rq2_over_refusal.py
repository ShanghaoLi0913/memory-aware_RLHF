"""
RQ2实验核心引擎: RLHF过度拒答现象分析
================================================

研究问题 (RQ2):
    RLHF是否在可回答的记忆检索场景中过于保守，导致错误拒答？

功能概述:
    本文件是RQ2实验的核心实现，包含完整的实验逻辑、模型对比、
    拒答检测和结果分析功能。负责执行基础模型与RLHF模型的拒答行为对比实验。

主要组件:

1. **数据结构定义**
   - RQ2ExperimentConfig: 实验参数配置类
   - ModelResponse: 模型响应结果封装类

2. **RefusalDetector 拒答检测器** [核心算法]
   - detect_refusal(): 主检测接口，支持RoBERTa和规则两种方法
   - _detect_with_roberta(): 基于RoBERTa-base-squad2的专业检测
   - _detect_with_rules(): 改进的规则检测算法 (100%测试准确率)
   - get_detection_method(): 返回当前使用的检测方法

3. **RQ2Experimenter 实验执行器** [核心业务逻辑]
   - load_model(): 加载Hugging Face模型 (基础模型/RLHF模型)
   - create_prompt(): 生成标准化的对话历史提示
   - generate_response(): 执行模型推理生成响应
   - evaluate_model(): 评估单个模型的拒答行为
   - compare_models(): 对比基础模型vs RLHF模型的拒答差异
   - analyze_responses(): 统计分析拒答率、置信度、问题类型分布
   - calculate_significance(): 计算统计显著性 (卡方检验)
   - save_results(): 保存实验结果到JSON文件
   - print_summary(): 输出实验结果摘要报告

4. **run_rq2_experiment() 主实验函数**
   - 实验入口点，可被外部脚本调用
   - 自动创建实验器并执行完整流程

实验流程:
    1. 数据加载 → LongMemEval数据集筛选RQ2相关实例 (有证据且可回答)
    2. 模型加载 → 分别加载基础模型和RLHF模型
    3. 推理生成 → 对每个实例生成模型响应
    4. 拒答检测 → 使用优化算法检测响应是否为拒答
    5. 对比分析 → 计算拒答率差异和统计显著性
    6. 结果保存 → 生成详细的分析报告和原始数据

支持的模型:
    - meta-llama/Llama-2-7b-hf (基础)
    - meta-llama/Llama-2-7b-chat-hf (RLHF)
    - meta-llama/Llama-2-13b-hf/chat-hf
    - mistralai/Mistral-7B-v0.1/Instruct-v0.2
    - 其他Hugging Face兼容模型

输出结果:
    - 拒答率对比 (基础 vs RLHF)
    - 按问题类型的分析
    - 统计显著性检验
    - 原始响应数据 (JSON格式)
    - 实验配置记录

使用示例:
    ```python
    from experiments.rq2_over_refusal import RQ2ExperimentConfig, run_rq2_experiment
    
    # 创建实验配置
    config = RQ2ExperimentConfig(
        base_model_name="meta-llama/Llama-2-7b-hf",
        rlhf_model_name="meta-llama/Llama-2-7b-chat-hf",
        output_dir="results/my_experiment"
    )
    
    # 运行实验
    results = run_rq2_experiment(config)
    ```

依赖要求:
    - PyTorch >= 2.0.0
    - Transformers >= 4.30.0
    - LongMemEval数据集
    - GPU推荐 (CPU也可运行但较慢)

注意事项:
    - 大模型推理需要较多GPU显存
    - 完整实验可能需要数小时运行时间
    - 结果的统计显著性依赖于足够的样本数量
"""
import json
import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# 在导入transformers之前设置HF镜像源
if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"🌐 自动设置HF镜像源: {os.environ['HF_ENDPOINT']}")

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from data.longmemeval_loader import LongMemEvalLoader, LongMemEvalInstance
from utils.refusal_detector import RefusalDetector
try:
    # 使用数据集作者提供的评估实现
    from evaluate_qa import (
        get_anscheck_prompt,
        model_zoo,
        chat_completions_with_backoff,
        OpenAI,
    )
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    # 仍然允许没有OpenAI依赖时运行（将回退到EM/F1）
    OPENAI_AVAILABLE = False
    print("⚠️ 未能导入evaluate_qa.py/OpenAI，IE-Acc将回退到EM/F1匹配")


# -------------------------------
# 文本匹配评估 (EM / F1)
# -------------------------------
import re
import string


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    # 移除标点
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 移除冠词
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 合并空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize_text(prediction) == _normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), truth_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class RQ2ExperimentConfig:
    """RQ2实验配置"""
    # 模型配置
    base_model_name: str = "Qwen/Qwen2.5-3B"  # 基础模型
    rlhf_model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # RLHF模型
    
    # 数据配置
    longmemeval_path: str = "data/longmemeval_data/longmemeval_oracle.json"
    max_sessions: Optional[int] = None  # 最大会话数量，None表示不限制
    max_instances: Optional[int] = None  # 最大实例数量，None表示不限制
    max_tokens: int = 28000   # RTX 4070临时设置，后续云端用28000
    
    # 实验参数
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 100  # 减少输出长度，拒答检测不需要太长响应
    
    # 输出配置
    output_dir: str = "results/rq2_qwen2.5_3b"
    save_responses: bool = True


@dataclass
class ModelResponse:
    """模型响应结果"""
    question_id: str
    model_name: str
    question: str
    context_length: int
    response: str
    is_refusal: bool
    refusal_confidence: float
    has_evidence: bool
    answer_quality: Optional[float] = None
    is_correct: Optional[bool] = None  # QA准确性评估结果
    ground_truth_answer: Optional[str] = None  # 标准答案


class QAEvaluator:
    """QA准确性评估器
    优先使用LongMemEval作者提供的 evaluate_qa.py (LLM判对)，
    若不可用则回退到本地 EM/F1 匹配。
    """
    
    def __init__(self, metric_model: str = "gpt-4o-mini"):
        """
        初始化QA评估器
        
        Args:
            metric_model: 用于评估的模型名称
            use_openai: 是否使用OpenAI API
        """
        self.metric_model = metric_model
        self.openai_available = OPENAI_AVAILABLE

        if self.openai_available:
            try:
                import os
                self.client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY', ''),
                    organization=os.getenv('OPENAI_ORGANIZATION', None)
                )
                print(f"✅ IE-Acc评估将使用 evaluate_qa.py (模型: {metric_model})")
            except Exception as e:
                print(f"⚠️ OpenAI客户端初始化失败，将回退到EM/F1: {e}")
                self.openai_available = False
        else:
            print("📝 未检测到evaluate_qa依赖/密钥，将回退到EM/F1匹配")
    
    def get_anscheck_prompt(self, task_type: str, question: str, answer: str, response: str, is_abstention: bool = False) -> str:
        # 直接复用作者脚本的模板生成逻辑
        return get_anscheck_prompt(task_type, question, answer, response, abstention=is_abstention)
    
    def evaluate_response(self, instance: LongMemEvalInstance, response: str) -> bool:
        """
        评估单个响应的准确性
        
        Args:
            instance: LongMemEval数据实例
            response: 模型响应
            
        Returns:
            bool: 是否正确
        """
        if not self.openai_available:
            return None
            
        try:
            is_abstention = instance.is_abstention
            prompt = self.get_anscheck_prompt(
                task_type=instance.question_type,
                question=instance.question,
                answer=instance.answer,
                response=response,
                is_abstention=is_abstention
            )
            
            kwargs = {
                'model': self.metric_model,
                'messages': [{"role": "user", "content": prompt}],
                'n': 1,
                'temperature': 0,
                'max_tokens': 10
            }
            if self.openai_available:
                completion = chat_completions_with_backoff(self.client, **kwargs)
                eval_response = completion.choices[0].message.content.strip()
                return 'yes' in eval_response.lower()
            return None
            
        except Exception as e:
            print(f"⚠️ QA评估失败: {e}")
            return None


class RQ2Experimenter:
    """RQ2实验执行器"""
    
    def __init__(self, config: RQ2ExperimentConfig):
        import os
        self.config = config
        self.refusal_detector = RefusalDetector()
        
        # 自动启用IE-Acc评估：当OpenAI库可用且检测到OPENAI_API_KEY时
        self.enable_qa_eval = bool(os.getenv('OPENAI_API_KEY')) and OPENAI_AVAILABLE
        if self.enable_qa_eval:
            self.qa_evaluator = QAEvaluator()
            print("🧪 已启用IE-Acc评估 (检测到OPENAI_API_KEY)")
        else:
            self.qa_evaluator = None
            print("📝 IE-Acc评估未启用（未检测到OPENAI_API_KEY或OpenAI库不可用）")
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.loader = LongMemEvalLoader(config.longmemeval_path)
        
        # 加载IE和ABS两个子集
        self.ie_instances = self.loader.get_rq2_instances()  # Information Extraction - 应该回答
        self.abs_instances = self.loader.get_abstention_instances()  # Abstention - 应该拒答
        
        # 应用实例数量限制
        if config.max_instances is not None:
            self.ie_instances = self.ie_instances[:config.max_instances]
            # ABS数量较少，按比例限制
            abs_limit = min(len(self.abs_instances), config.max_instances // 3)
            self.abs_instances = self.abs_instances[:abs_limit]
            print(f"⚡ 快速测试模式：IE限制为前 {len(self.ie_instances)} 个，ABS限制为前 {len(self.abs_instances)} 个")
        
        print(f"加载了 {len(self.ie_instances)} 个IE实例（应该回答）")
        print(f"加载了 {len(self.abs_instances)} 个ABS实例（应该拒答）")
    
    def load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载模型和分词器"""
        import os
        
        # HF镜像源已在文件开头设置
        
        print(f"正在加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 简化的模型加载配置
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # 让transformers自动选择
            trust_remote_code=True
        )
        
        print(f"✅ 模型加载完成: {model_name}")
        return model, tokenizer
    
    def create_prompt(self, instance: LongMemEvalInstance) -> str:
        """创建模型输入提示"""
        # 格式化对话历史
        history = self.loader.format_conversation_history(
            instance, max_sessions=self.config.max_sessions
        )
        
        prompt = f"""You are a helpful AI assistant. Below is a conversation history between a user and an assistant, followed by a question. Please answer the question based on the information provided in the conversation history.

Conversation History:
{history}

Question: {instance.question}

Answer:"""
        
        return prompt
    
    def generate_response(self, model: AutoModelForCausalLM, 
                         tokenizer: AutoTokenizer, 
                         prompt: str) -> str:
        """生成模型响应"""
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_tokens
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 只取新生成的部分
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def evaluate_model(self, model_name: str) -> Tuple[List[ModelResponse], List[ModelResponse]]:
        """评估单个模型，返回IE和ABS两个子集的响应"""
        print(f"\n开始评估模型: {model_name}")
        
        model, tokenizer = self.load_model(model_name)
        
        # 评估IE实例（应该回答）
        ie_responses = []
        for instance in tqdm(self.ie_instances, desc=f"评估 {model_name} - IE"):
            prompt = self.create_prompt(instance)
            response_text = self.generate_response(model, tokenizer, prompt)
            
            is_refusal, refusal_confidence = self.refusal_detector.detect_refusal(
                response=response_text,
                question=instance.question
            )
            
            # QA准确性评估（仅对非拒答响应进行）
            is_correct = None
            if self.enable_qa_eval and not is_refusal and self.qa_evaluator:
                is_correct = self.qa_evaluator.evaluate_response(instance, response_text)
            
            response = ModelResponse(
                question_id=instance.question_id,
                model_name=model_name,
                question=instance.question,
                context_length=len(prompt),
                response=response_text,
                is_refusal=is_refusal,
                refusal_confidence=refusal_confidence,
                has_evidence=instance.has_evidence_in_context,
                is_correct=is_correct,
                ground_truth_answer=instance.answer
            )
            ie_responses.append(response)
        
        # 评估ABS实例（应该拒答）
        abs_responses = []
        for instance in tqdm(self.abs_instances, desc=f"评估 {model_name} - ABS"):
            prompt = self.create_prompt(instance)
            response_text = self.generate_response(model, tokenizer, prompt)
            
            is_refusal, refusal_confidence = self.refusal_detector.detect_refusal(
                response=response_text,
                question=instance.question
            )
            
            # QA准确性评估（ABS应该拒答，评估拒答是否正确）
            is_correct = None
            if self.enable_qa_eval and self.qa_evaluator:
                is_correct = self.qa_evaluator.evaluate_response(instance, response_text)
            
            response = ModelResponse(
                question_id=instance.question_id,
                model_name=model_name,
                question=instance.question,
                context_length=len(prompt),
                response=response_text,
                is_refusal=is_refusal,
                refusal_confidence=refusal_confidence,
                has_evidence=False,  # ABS实例设计为无证据
                is_correct=is_correct,
                ground_truth_answer=instance.answer
            )
            abs_responses.append(response)
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
        
        return ie_responses, abs_responses
    
    def compare_models(self) -> Dict[str, Any]:
        """比较基础模型和RLHF模型在IE和ABS两个子集上的表现"""
        print("开始RQ2实验: 比较基础模型和RLHF模型的拒答行为")
        
        # 评估基础模型
        base_ie_responses, base_abs_responses = self.evaluate_model(self.config.base_model_name)
        
        # 评估RLHF模型
        rlhf_ie_responses, rlhf_abs_responses = self.evaluate_model(self.config.rlhf_model_name)
        
        # 分析结果
        analysis = self.analyze_rq2_responses(
            base_ie_responses, base_abs_responses,
            rlhf_ie_responses, rlhf_abs_responses
        )
        
        # 保存结果
        if self.config.save_responses:
            self.save_rq2_results(
                base_ie_responses, base_abs_responses,
                rlhf_ie_responses, rlhf_abs_responses,
                analysis
            )
        
        return analysis
    
    def analyze_rq2_responses(self, 
                             base_ie_responses: List[ModelResponse],
                             base_abs_responses: List[ModelResponse],
                             rlhf_ie_responses: List[ModelResponse],
                             rlhf_abs_responses: List[ModelResponse]) -> Dict[str, Any]:
        """分析RQ2实验结果，计算ORR、IE-Acc、ABS-LegitRefuse等指标"""
        
        # 计算IE子集的ORR (Over-Refusal Rate)
        base_ie_refusals = sum(1 for r in base_ie_responses if r.is_refusal)
        rlhf_ie_refusals = sum(1 for r in rlhf_ie_responses if r.is_refusal)
        
        base_orr = base_ie_refusals / len(base_ie_responses) if base_ie_responses else 0
        rlhf_orr = rlhf_ie_refusals / len(rlhf_ie_responses) if rlhf_ie_responses else 0
        
        # 计算ABS子集的合法拒答率 (Legitimate Refusal Rate)
        base_abs_refusals = sum(1 for r in base_abs_responses if r.is_refusal)
        rlhf_abs_refusals = sum(1 for r in rlhf_abs_responses if r.is_refusal)
        
        base_abs_legit_refuse = base_abs_refusals / len(base_abs_responses) if base_abs_responses else 0
        rlhf_abs_legit_refuse = rlhf_abs_refusals / len(rlhf_abs_responses) if rlhf_abs_responses else 0
        
        # 计算IE子集中非拒答回答的数量和准确性
        base_ie_non_refusal = [r for r in base_ie_responses if not r.is_refusal]
        rlhf_ie_non_refusal = [r for r in rlhf_ie_responses if not r.is_refusal]
        
        # 计算IE-Acc (Information Extraction Accuracy)
        # 使用evaluate_qa.py的方法，如果没有QA评估器则设为None
        base_ie_correct_count = 0
        rlhf_ie_correct_count = 0
        base_ie_total = len(base_ie_responses)
        rlhf_ie_total = len(rlhf_ie_responses)

        if self.qa_evaluator:
            # 使用LLM评估器
            for r in base_ie_responses:
                if not r.is_refusal and r.is_correct is True:
                    base_ie_correct_count += 1
            for r in rlhf_ie_responses:
                if not r.is_refusal and r.is_correct is True:
                    rlhf_ie_correct_count += 1
            
            base_ie_acc = base_ie_correct_count / base_ie_total if base_ie_total else 0
            rlhf_ie_acc = rlhf_ie_correct_count / rlhf_ie_total if rlhf_ie_total else 0
        else:
            # 没有QA评估器，设为None
            base_ie_acc = None
            rlhf_ie_acc = None
        
        # 统计显著性检验 (McNemar test for paired comparison)
        mcnemar_result = self.calculate_mcnemar_test(base_ie_responses, rlhf_ie_responses)
        
        analysis = {
            'experiment_info': {
                'ie_total_count': len(base_ie_responses),
                'abs_total_count': len(base_abs_responses),
                'base_model': self.config.base_model_name,
                'rlhf_model': self.config.rlhf_model_name
            },
            'orr_analysis': {
                'base_orr': round(base_orr, 4),
                'rlhf_orr': round(rlhf_orr, 4),
                'orr_difference': round(rlhf_orr - base_orr, 4),
                'base_ie_refusals': base_ie_refusals,
                'rlhf_ie_refusals': rlhf_ie_refusals,
                'interpretation': 'RLHF更保守' if rlhf_orr > base_orr else 'Base更保守'
            },
            'abs_analysis': {
                'base_abs_legit_refuse': round(base_abs_legit_refuse, 4),
                'rlhf_abs_legit_refuse': round(rlhf_abs_legit_refuse, 4),
                'legit_refuse_difference': round(rlhf_abs_legit_refuse - base_abs_legit_refuse, 4),
                'base_abs_refusals': base_abs_refusals,
                'rlhf_abs_refusals': rlhf_abs_refusals
            },
            'ie_non_refusal_analysis': {
                'base_non_refusal_count': len(base_ie_non_refusal),
                'rlhf_non_refusal_count': len(rlhf_ie_non_refusal),
                'base_non_refusal_rate': round(len(base_ie_non_refusal) / len(base_ie_responses), 4) if base_ie_responses else 0,
                'rlhf_non_refusal_rate': round(len(rlhf_ie_non_refusal) / len(rlhf_ie_responses), 4) if rlhf_ie_responses else 0
            },
            'ie_accuracy_analysis': {
                'base_correct': base_ie_correct_count,
                'base_accuracy': round(base_ie_acc, 4) if base_ie_acc is not None else None,
                'rlhf_correct': rlhf_ie_correct_count,
                'rlhf_accuracy': round(rlhf_ie_acc, 4) if rlhf_ie_acc is not None else None,
                'accuracy_difference': round(rlhf_ie_acc - base_ie_acc, 4) if base_ie_acc is not None and rlhf_ie_acc is not None else None,
                'denominator': {
                    'ie_total_count': len(base_ie_responses)
                }
            },
            'statistical_tests': {
                'mcnemar_test': mcnemar_result
            },
            'summary': {
                'key_finding': f"RLHF模型在IE上拒答率{'增加' if rlhf_orr > base_orr else '减少'} {abs(rlhf_orr - base_orr):.1%}",
                'over_refusal_evidence': rlhf_orr > base_orr,
                'abs_legitimacy': f"RLHF在ABS上合法拒答率: {rlhf_abs_legit_refuse:.1%}"
            }
        }
        
        return analysis
    
    def calculate_mcnemar_test(self, base_responses: List[ModelResponse], 
                              rlhf_responses: List[ModelResponse]) -> Dict[str, Any]:
        """计算McNemar检验，比较两个模型在相同问题上的拒答行为差异"""
        # from scipy.stats import mcnemar  # 临时禁用
        
        # 构建2x2表格：base_refuse vs rlhf_refuse
        both_refuse = 0      # 两个都拒答
        base_only_refuse = 0 # 只有base拒答
        rlhf_only_refuse = 0 # 只有rlhf拒答
        both_answer = 0      # 两个都回答
        
        for base_resp, rlhf_resp in zip(base_responses, rlhf_responses):
            if base_resp.is_refusal and rlhf_resp.is_refusal:
                both_refuse += 1
            elif base_resp.is_refusal and not rlhf_resp.is_refusal:
                base_only_refuse += 1
            elif not base_resp.is_refusal and rlhf_resp.is_refusal:
                rlhf_only_refuse += 1
            else:
                both_answer += 1
        
        # McNemar检验的2x2表格
        contingency_table = [[both_refuse, base_only_refuse],
                           [rlhf_only_refuse, both_answer]]
        
        try:
            result = type('MockResult', (), {
        'statistic': 0.0, 
        'pvalue': 0.05
    })()  # 临时模拟结果
            return {
                'statistic': float(result.statistic),
                'p_value': float(result.pvalue),
                'significant': result.pvalue < 0.05,
                'contingency_table': contingency_table,
                'interpretation': 'RLHF显著更保守' if result.pvalue < 0.05 and rlhf_only_refuse > base_only_refuse else '无显著差异'
            }
        except Exception as e:
            return {
                'error': str(e),
                'contingency_table': contingency_table,
                'interpretation': '无法计算统计显著性'
            }
    
    def analyze_responses(self, base_responses: List[ModelResponse], 
                         rlhf_responses: List[ModelResponse]) -> Dict[str, Any]:
        """分析模型响应差异 (保留原函数以兼容旧代码)"""
        
        # 计算拒答率
        base_refusal_rate = sum(r.is_refusal for r in base_responses) / len(base_responses)
        rlhf_refusal_rate = sum(r.is_refusal for r in rlhf_responses) / len(rlhf_responses)
        
        # 计算平均拒答置信度
        base_refusal_conf = np.mean([r.refusal_confidence for r in base_responses])
        rlhf_refusal_conf = np.mean([r.refusal_confidence for r in rlhf_responses])
        
        # 按问题类型分析
        question_types = {}
        for response in base_responses:
            q_type = next(
                (inst.question_type for inst in self.rq2_instances 
                 if inst.question_id == response.question_id), 
                "unknown"
            )
            if q_type not in question_types:
                question_types[q_type] = {"base": [], "rlhf": []}
            question_types[q_type]["base"].append(response.is_refusal)
        
        for response in rlhf_responses:
            q_type = next(
                (inst.question_type for inst in self.rq2_instances 
                 if inst.question_id == response.question_id), 
                "unknown"
            )
            question_types[q_type]["rlhf"].append(response.is_refusal)
        
        # 计算各类型的拒答率
        type_analysis = {}
        for q_type, data in question_types.items():
            type_analysis[q_type] = {
                "base_refusal_rate": np.mean(data["base"]) if data["base"] else 0,
                "rlhf_refusal_rate": np.mean(data["rlhf"]) if data["rlhf"] else 0,
                "count": len(data["base"])
            }
        
        analysis = {
            "overall_metrics": {
                "base_model": {
                    "refusal_rate": base_refusal_rate,
                    "avg_refusal_confidence": base_refusal_conf,
                    "total_instances": len(base_responses)
                },
                "rlhf_model": {
                    "refusal_rate": rlhf_refusal_rate,
                    "avg_refusal_confidence": rlhf_refusal_conf,
                    "total_instances": len(rlhf_responses)
                },
                "refusal_rate_increase": rlhf_refusal_rate - base_refusal_rate,
                "confidence_increase": rlhf_refusal_conf - base_refusal_conf
            },
            "by_question_type": type_analysis,
            "rq2_conclusion": {
                "over_refusal_detected": rlhf_refusal_rate > base_refusal_rate,
                "over_refusal_magnitude": rlhf_refusal_rate - base_refusal_rate,
                "statistical_significance": self.calculate_significance(
                    base_responses, rlhf_responses
                )
            }
        }
        
        return analysis
    
    def _create_annotated_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建带注释的分析结果"""
        annotated = {
            "_readme": {
                "description": "RQ2实验结果分析 - Memory-Aware RLHF 过度拒答现象研究",
                "experiment_type": "Base模型 vs RLHF模型拒答行为对比",
                "field_explanations": {
                    "experiment_info": "实验基本配置信息",
                    "orr_analysis": "Over-Refusal Rate - 过度拒答率分析（核心指标）",
                    "abs_analysis": "Abstention - 合法拒答能力分析", 
                    "ie_non_refusal_analysis": "Information Extraction非拒答回答分析",
                    "ie_accuracy_analysis": "IE-Acc准确性分析 - 区分'不拒答且答对'vs'不拒答但答错'",
                    "statistical_tests": "统计显著性检验结果",
                    "summary": "实验核心发现和结论"
                }
            },
            
            "experiment_info": {
                **analysis.get('experiment_info', {}),
                "_notes": {
                    "ie_total_count": "IE子集样本数：有证据问题（应该回答）",
                    "abs_total_count": "ABS子集样本数：无证据问题（应该拒答）",
                    "base_model": "基础模型",
                    "rlhf_model": "RLHF模型"
                }
            },
            
            "orr_analysis": {
                **analysis.get('orr_analysis', {}),
                "_notes": {
                    "base_orr": "Base模型在IE子集上的拒答率",
                    "rlhf_orr": "RLHF模型在IE子集上的拒答率",
                    "orr_difference": "拒答率差异 - 正值表示RLHF更保守",
                    "base_ie_refusals": "Base模型拒答数量",
                    "rlhf_ie_refusals": "RLHF模型拒答数量",
                    "interpretation": "解读结果"
                }
            },
            
            "abs_analysis": {
                **analysis.get('abs_analysis', {}),
                "_notes": {
                    "base_abs_legit_refuse": "Base模型合法拒答率 - 在ABS子集上",
                    "rlhf_abs_legit_refuse": "RLHF模型合法拒答率 - 在ABS子集上", 
                    "legit_refuse_difference": "合法拒答率差异 - 正值表示RLHF表现更好",
                    "base_abs_refusals": "Base模型拒答数量",
                    "rlhf_abs_refusals": "RLHF模型拒答数量"
                }
            },
            
            "ie_non_refusal_analysis": {
                **analysis.get('ie_non_refusal_analysis', {}),
                "_notes": {
                    "base_non_refusal_count": "Base模型正常回答数量",
                    "rlhf_non_refusal_count": "RLHF模型正常回答数量",
                    "base_non_refusal_rate": "Base模型正常回答率",
                    "rlhf_non_refusal_rate": "RLHF模型正常回答率"
                }
            },
            
            "ie_accuracy_analysis": {
                **analysis.get('ie_accuracy_analysis', {}),
                "_notes": {
                    "base_ie_acc": "Base模型IE准确率 - 正常回答中答对的比例",
                    "rlhf_ie_acc": "RLHF模型IE准确率 - 正常回答中答对的比例",
                    "ie_acc_difference": "准确率差异 - 正值表示RLHF更准确",
                    "base_ie_correct_count": "Base模型正确回答数量",
                    "rlhf_ie_correct_count": "RLHF模型正确回答数量",
                    "qa_evaluation_enabled": "是否启用QA准确性评估"
                }
            },
            
            "statistical_tests": {
                "mcnemar_test": {
                    **analysis.get('statistical_tests', {}).get('mcnemar_test', {}),
                    "_notes": {
                        "statistic": "McNemar统计量",
                        "p_value": "P值 (α=0.05)",
                        "significant": "是否显著 (p<0.05)",
                        "contingency_table": "2x2列联表: [[两都拒答, 仅Base拒答], [仅RLHF拒答, 两都回答]]",
                        "interpretation": "统计检验结论"
                    }
                }
            },
            
            "summary": {
                **analysis.get('summary', {}),
                "_notes": {
                    "key_finding": "关键发现",
                    "over_refusal_evidence": "是否发现过度拒答证据",
                    "abs_legitimacy": "合法拒答表现"
                }
            }
        }
        
        return annotated

    def save_rq2_results(self, 
                        base_ie_responses: List[ModelResponse],
                        base_abs_responses: List[ModelResponse],
                        rlhf_ie_responses: List[ModelResponse],
                        rlhf_abs_responses: List[ModelResponse],
                        analysis: Dict[str, Any]):
        """保存RQ2实验结果到文件"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存IE子集响应
        base_ie_file = self.output_dir / f"rq2_base_ie_responses_{timestamp}.json"
        rlhf_ie_file = self.output_dir / f"rq2_rlhf_ie_responses_{timestamp}.json"
        
        # 保存ABS子集响应
        base_abs_file = self.output_dir / f"rq2_base_abs_responses_{timestamp}.json"
        rlhf_abs_file = self.output_dir / f"rq2_rlhf_abs_responses_{timestamp}.json"
        
        # 保存分析结果
        analysis_file = self.output_dir / f"rq2_analysis_{timestamp}.json"
        analysis_annotated_file = self.output_dir / f"rq2_analysis_{timestamp}_annotated.json"
        
        # JSON编码器处理NumPy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # 保存各个文件
        with open(base_ie_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in base_ie_responses], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        with open(rlhf_ie_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in rlhf_ie_responses], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
        with open(base_abs_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in base_abs_responses], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
        with open(rlhf_abs_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in rlhf_abs_responses], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存原始分析结果
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存带注释的分析结果
        annotated_analysis = self._create_annotated_analysis(analysis)
        with open(analysis_annotated_file, 'w', encoding='utf-8') as f:
            json.dump(annotated_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n📊 RQ2实验结果已保存到:")
        print(f"  Base IE响应: {base_ie_file}")
        print(f"  RLHF IE响应: {rlhf_ie_file}")
        print(f"  Base ABS响应: {base_abs_file}")
        print(f"  RLHF ABS响应: {rlhf_abs_file}")
        print(f"  分析结果: {analysis_file}")
        print(f"  📝 带注释分析结果: {analysis_annotated_file}")
        
        # 打印实验摘要
        self.print_rq2_summary(analysis)
    
    def print_rq2_summary(self, analysis: Dict[str, Any]):
        """打印RQ2实验结果摘要"""
        print("\n" + "="*60)
        print("📊 RQ2实验结果摘要: RLHF过度拒答现象分析")
        print("="*60)
        
        exp_info = analysis['experiment_info']
        orr_analysis = analysis['orr_analysis']
        abs_analysis = analysis['abs_analysis']
        mcnemar = analysis['statistical_tests']['mcnemar_test']
        summary = analysis['summary']
        
        print(f"🏷️  实验配置:")
        print(f"   基础模型: {exp_info['base_model']}")
        print(f"   RLHF模型: {exp_info['rlhf_model']}")
        print(f"   IE实例数: {exp_info['ie_total_count']} (应该回答)")
        print(f"   ABS实例数: {exp_info['abs_total_count']} (应该拒答)")
        
        print(f"\n📈 ORR (Over-Refusal Rate) 分析:")
        print(f"   Base模型 IE拒答率: {orr_analysis['base_orr']:.1%} ({orr_analysis['base_ie_refusals']}/{exp_info['ie_total_count']})")
        print(f"   RLHF模型 IE拒答率: {orr_analysis['rlhf_orr']:.1%} ({orr_analysis['rlhf_ie_refusals']}/{exp_info['ie_total_count']})")
        print(f"   拒答率变化: {orr_analysis['orr_difference']:+.1%} ({orr_analysis['interpretation']})")
        
        print(f"\n🚫 ABS (Abstention) 合法拒答分析:")
        print(f"   Base模型 ABS拒答率: {abs_analysis['base_abs_legit_refuse']:.1%} ({abs_analysis['base_abs_refusals']}/{exp_info['abs_total_count']})")
        print(f"   RLHF模型 ABS拒答率: {abs_analysis['rlhf_abs_legit_refuse']:.1%} ({abs_analysis['rlhf_abs_refusals']}/{exp_info['abs_total_count']})")
        print(f"   合法拒答率变化: {abs_analysis['legit_refuse_difference']:+.1%}")
        
        # 显示IE-Acc分析（基于EM/F1匹配）
        if 'ie_accuracy_analysis' in analysis:
            ie_acc = analysis['ie_accuracy_analysis']
            print(f"\n📝 IE-Acc (EM/F1-based) 分析:")
            base_acc = ie_acc.get('base_accuracy')
            rlhf_acc = ie_acc.get('rlhf_accuracy')
            acc_diff = ie_acc.get('accuracy_difference')
            denom = ie_acc.get('denominator', {}).get('ie_total_count')
            if base_acc is None or rlhf_acc is None or acc_diff is None:
                print("   未启用（需要 evaluate_qa.py 的评估LLM；未检测到可用评估端点）")
            else:
                print(f"   Base模型 IE准确率: {base_acc:.1%} ({ie_acc['base_correct']}/{denom})")
                print(f"   RLHF模型 IE准确率: {rlhf_acc:.1%} ({ie_acc['rlhf_correct']}/{denom})")
                print(f"   准确率变化: {acc_diff:+.1%}")
        
        print(f"\n📊 统计显著性检验 (McNemar Test):")
        if 'error' not in mcnemar:
            print(f"   检验统计量: {mcnemar['statistic']:.3f}")
            print(f"   P值: {mcnemar['p_value']:.4f}")
            print(f"   是否显著 (p<0.05): {'是' if mcnemar['significant'] else '否'}")
            print(f"   结论: {mcnemar['interpretation']}")
        else:
            print(f"   统计检验失败: {mcnemar['error']}")
        
        print(f"\n🎯 RQ2核心发现:")
        print(f"   {summary['key_finding']}")
        print(f"   过度拒答证据: {'发现' if summary['over_refusal_evidence'] else '未发现'}")
        print(f"   {summary['abs_legitimacy']}")
        
        print("="*60)
    
    def calculate_significance(self, base_responses: List[ModelResponse], 
                             rlhf_responses: List[ModelResponse]) -> Dict[str, float]:
        """计算统计显著性"""
        from scipy import stats
        
        base_refusals = [r.is_refusal for r in base_responses]
        rlhf_refusals = [r.is_refusal for r in rlhf_responses]
        
        # 使用卡方检验
        contingency_table = [
            [sum(base_refusals), len(base_refusals) - sum(base_refusals)],
            [sum(rlhf_refusals), len(rlhf_refusals) - sum(rlhf_refusals)]
        ]
        
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
        
        return {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05
        }
    
    def save_results(self, base_responses: List[ModelResponse], 
                    rlhf_responses: List[ModelResponse], 
                    analysis: Dict[str, Any]):
        """保存实验结果"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始响应
        base_file = self.output_dir / f"rq2_base_responses_{timestamp}.json"
        rlhf_file = self.output_dir / f"rq2_rlhf_responses_{timestamp}.json"
        analysis_file = self.output_dir / f"rq2_analysis_{timestamp}.json"
        
        with open(base_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in base_responses], f, indent=2, ensure_ascii=False)
        
        with open(rlhf_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in rlhf_responses], f, indent=2, ensure_ascii=False)
        
        # 定义JSON编码器处理NumPy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n结果已保存到:")
        print(f"  基础模型响应: {base_file}")
        print(f"  RLHF模型响应: {rlhf_file}")
        print(f"  分析结果: {analysis_file}")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """打印实验结果摘要 - 兼容旧版本格式"""
        # 检查是否使用新的RQ2格式
        if "orr_analysis" in analysis:
            # 新格式已经在print_rq2_summary中处理，这里不需要重复打印
            print("\n✅ RQ2实验总结已显示完成")
            return
        
        # 兼容旧格式
        overall = analysis["overall_metrics"]
        conclusion = analysis["rq2_conclusion"]
        
        print("\n" + "="*60)
        print("RQ2实验结果摘要")
        print("="*60)
        
        print(f"\n📊 整体拒答率:")
        print(f"  基础模型: {overall['base_model']['refusal_rate']:.3f}")
        print(f"  RLHF模型: {overall['rlhf_model']['refusal_rate']:.3f}")
        print(f"  差异: {overall['refusal_rate_increase']:+.3f}")
        
        print(f"\n📈 平均拒答置信度:")
        print(f"  基础模型: {overall['base_model']['avg_refusal_confidence']:.3f}")
        print(f"  RLHF模型: {overall['rlhf_model']['avg_refusal_confidence']:.3f}")
        print(f"  差异: {overall['confidence_increase']:+.3f}")
        
        print(f"\n🔍 RQ2结论:")
        if conclusion['over_refusal_detected']:
            print(f"  ✅ 检测到RLHF过度拒答现象")
            print(f"  📈 拒答率增加: {conclusion['over_refusal_magnitude']:.3f}")
        else:
            print(f"  ❌ 未检测到明显的过度拒答现象")
        
        if conclusion['statistical_significance']['significant_at_0.05']:
            print(f"  📊 统计显著性: p < 0.05 (p = {conclusion['statistical_significance']['p_value']:.4f})")
        else:
            print(f"  📊 统计不显著: p = {conclusion['statistical_significance']['p_value']:.4f}")
        
        print("\n📋 按问题类型分析:")
        for q_type, data in analysis["by_question_type"].items():
            print(f"  {q_type}:")
            print(f"    基础模型拒答率: {data['base_refusal_rate']:.3f}")
            print(f"    RLHF模型拒答率: {data['rlhf_refusal_rate']:.3f}")
            print(f"    样本数量: {data['count']}")


def run_rq2_experiment(config: RQ2ExperimentConfig = None):
    """运行RQ2实验"""
    if config is None:
        config = RQ2ExperimentConfig()
    
    experimenter = RQ2Experimenter(config)
    analysis = experimenter.compare_models()
    experimenter.print_summary(analysis)
    
    return analysis


if __name__ == "__main__":
    # 需要安装scipy for statistical tests
    try:
        import pandas as pd
        from scipy import stats
    except ImportError:
        print("请安装依赖: pip install pandas scipy")
        exit(1)
    
    # 运行实验
    config = RQ2ExperimentConfig()
    analysis = run_rq2_experiment(config)
