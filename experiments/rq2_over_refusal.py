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
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from data.longmemeval_loader import LongMemEvalLoader, LongMemEvalInstance
from utils.refusal_detector import RefusalDetector


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



class RQ2Experimenter:
    """RQ2实验执行器"""
    
    def __init__(self, config: RQ2ExperimentConfig):
        self.config = config
        self.refusal_detector = RefusalDetector()
        
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
            
            response = ModelResponse(
                question_id=instance.question_id,
                model_name=model_name,
                question=instance.question,
                context_length=len(prompt),
                response=response_text,
                is_refusal=is_refusal,
                refusal_confidence=refusal_confidence,
                has_evidence=instance.has_evidence_in_context
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
            
            response = ModelResponse(
                question_id=instance.question_id,
                model_name=model_name,
                question=instance.question,
                context_length=len(prompt),
                response=response_text,
                is_refusal=is_refusal,
                refusal_confidence=refusal_confidence,
                has_evidence=False  # ABS实例设计为无证据
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
        
        # 计算IE子集中非拒答回答的数量 (为后续准确率计算准备)
        base_ie_non_refusal = [r for r in base_ie_responses if not r.is_refusal]
        rlhf_ie_non_refusal = [r for r in rlhf_ie_responses if not r.is_refusal]
        
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
        from scipy.stats import mcnemar
        
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
            result = mcnemar(contingency_table, exact=True)
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
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n📊 RQ2实验结果已保存到:")
        print(f"  Base IE响应: {base_ie_file}")
        print(f"  RLHF IE响应: {rlhf_ie_file}")
        print(f"  Base ABS响应: {base_abs_file}")
        print(f"  RLHF ABS响应: {rlhf_abs_file}")
        print(f"  分析结果: {analysis_file}")
        
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
        """打印实验结果摘要"""
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
