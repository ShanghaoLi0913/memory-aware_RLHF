#!/usr/bin/env python3
"""
LongMemEval数据集统计分析工具
=====================================

功能概述:
    本脚本用于分析LongMemEval数据集的详细统计信息，包括问题类型分布、
    数据规模、对话历史长度等关键指标。

主要统计功能:
1. **问题类型分布** (Question Type Distribution)
   - 统计每种question_type的数量和占比
   - 区分普通问题和Abstention问题(_abs结尾)

2. **数据规模统计** (Data Scale Statistics)
   - 总问题数量
   - 平均对话历史长度
   - 最长/最短对话历史
   - Token数量估算

3. **时间分布分析** (Temporal Distribution)
   - 问题日期分布
   - 历史会话时间跨度

4. **会话统计** (Session Statistics)
   - 平均会话数量
   - 证据会话比例
   - 会话长度分布

使用方法:
    ```bash
    # 分析默认数据集
    python analyze_dataset_stats.py

    # 分析指定数据集
    python analyze_dataset_stats.py --data-file data/longmemeval_data/longmemeval_s.json

    # 保存详细报告
    python analyze_dataset_stats.py --save-report
    ```

输出结果:
- 控制台显示统计摘要
- 可选保存详细JSON报告到results/目录
"""

import json
import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import sys

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.longmemeval_loader import LongMemEvalLoader

class DatasetAnalyzer:
    """LongMemEval数据集统计分析器"""
    
    def __init__(self, data_file: str):
        """
        初始化分析器
        
        Args:
            data_file: 数据集文件路径
        """
        self.data_file = data_file
        self.loader = LongMemEvalLoader(data_file)
        self.instances = self.loader.load_data()
        
    def analyze_question_types(self) -> Dict[str, Any]:
        """分析问题类型分布"""
        print("📊 分析问题类型分布...")
        
        # 统计问题类型
        question_types = Counter()
        abstention_types = Counter()
        
        for instance in self.instances:
            q_type = instance.question_type
            question_types[q_type] += 1
            
            # 检查是否为Abstention问题
            if instance.question_id.endswith('_abs'):
                abstention_types[q_type] += 1
        
        total_questions = len(self.instances)
        total_abstention = sum(abstention_types.values())
        
        # 计算比例
        type_stats = {}
        for q_type, count in question_types.items():
            abs_count = abstention_types.get(q_type, 0)
            regular_count = count - abs_count
            
            type_stats[q_type] = {
                'total': count,
                'regular': regular_count,
                'abstention': abs_count,
                'percentage': round(count / total_questions * 100, 2)
            }
        
        return {
            'total_questions': total_questions,
            'total_abstention': total_abstention,
            'regular_questions': total_questions - total_abstention,
            'question_types': dict(question_types),
            'abstention_types': dict(abstention_types),
            'detailed_stats': type_stats
        }
    
    def analyze_session_statistics(self) -> Dict[str, Any]:
        """分析会话统计信息"""
        print("📊 分析会话统计信息...")
        
        session_counts = []
        evidence_session_counts = []
        session_lengths = []  # 每个会话的对话轮数
        haystack_session_counts = []  # 每个问题的haystack_sessions数量
        haystack_token_lengths = []  # 每个问题的haystack_sessions总token长度
        
        for instance in self.instances:
            # 会话数量
            num_sessions = len(instance.haystack_sessions)
            session_counts.append(num_sessions)
            
            # haystack_sessions数量（每个问题的会话数量）
            haystack_session_counts.append(num_sessions)
            
            # 计算haystack_sessions的总token长度
            total_haystack_chars = 0
            for session in instance.haystack_sessions:
                for turn in session:
                    content = str(turn.get('content', ''))
                    total_haystack_chars += len(content)
            
            # 估算token数量 (1 token ≈ 4 characters)
            estimated_tokens = total_haystack_chars // 4
            haystack_token_lengths.append(estimated_tokens)
            
            # 证据会话数量
            num_evidence = len(instance.answer_session_ids)
            evidence_session_counts.append(num_evidence)
            
            # 会话长度（对话轮数）
            for session in instance.haystack_sessions:
                session_lengths.append(len(session))
        
        return {
            'sessions_per_question': {
                'min': min(session_counts),
                'max': max(session_counts),
                'avg': round(sum(session_counts) / len(session_counts), 2),
                'total_sessions': sum(session_counts)
            },
            'haystack_sessions_count': {
                'min': min(haystack_session_counts),
                'max': max(haystack_session_counts),
                'avg': round(sum(haystack_session_counts) / len(haystack_session_counts), 2),
                'median': sorted(haystack_session_counts)[len(haystack_session_counts)//2],
                'total_questions': len(haystack_session_counts)
            },
            'haystack_sessions_token_length': {
                'min': min(haystack_token_lengths),
                'max': max(haystack_token_lengths),
                'avg': round(sum(haystack_token_lengths) / len(haystack_token_lengths), 2),
                'median': sorted(haystack_token_lengths)[len(haystack_token_lengths)//2],
                'percentile_75': sorted(haystack_token_lengths)[int(len(haystack_token_lengths) * 0.75)],
                'percentile_90': sorted(haystack_token_lengths)[int(len(haystack_token_lengths) * 0.9)],
                'total_tokens': sum(haystack_token_lengths)
            },
            'evidence_sessions_per_question': {
                'min': min(evidence_session_counts),
                'max': max(evidence_session_counts),
                'avg': round(sum(evidence_session_counts) / len(evidence_session_counts), 2),
                'total_evidence_sessions': sum(evidence_session_counts)
            },
            'turns_per_session': {
                'min': min(session_lengths),
                'max': max(session_lengths),
                'avg': round(sum(session_lengths) / len(session_lengths), 2),
                'total_turns': sum(session_lengths)
            }
        }
    
    def analyze_text_statistics(self) -> Dict[str, Any]:
        """分析文本统计信息（长度、Token估算等）"""
        print("📊 分析文本统计信息...")
        
        question_lengths = []
        answer_lengths = []
        context_lengths = []  # 整个对话历史的长度
        
        for instance in self.instances:
            # 问题长度
            question_lengths.append(len(str(instance.question)))
            
            # 答案长度 - 添加类型检查
            answer_text = str(instance.answer) if instance.answer is not None else ""
            answer_lengths.append(len(answer_text))
            
            # 对话历史长度（所有会话的所有消息）
            total_context_length = 0
            for session in instance.haystack_sessions:
                for turn in session:
                    content = str(turn.get('content', ''))
                    total_context_length += len(content)
            context_lengths.append(total_context_length)
        
        # 简单的Token估算（1 token ≈ 4 characters for Chinese/English mix）
        def estimate_tokens(char_length):
            return char_length // 4
        
        return {
            'question_lengths': {
                'min_chars': min(question_lengths),
                'max_chars': max(question_lengths),
                'avg_chars': round(sum(question_lengths) / len(question_lengths), 2),
                'avg_tokens_est': round(sum(question_lengths) / len(question_lengths) / 4, 2)
            },
            'answer_lengths': {
                'min_chars': min(answer_lengths),
                'max_chars': max(answer_lengths),
                'avg_chars': round(sum(answer_lengths) / len(answer_lengths), 2),
                'avg_tokens_est': round(sum(answer_lengths) / len(answer_lengths) / 4, 2)
            },
            'context_lengths': {
                'min_chars': min(context_lengths),
                'max_chars': max(context_lengths),
                'avg_chars': round(sum(context_lengths) / len(context_lengths), 2),
                'avg_tokens_est': round(sum(context_lengths) / len(context_lengths) / 4, 2)
            }
        }
    
    def analyze_temporal_distribution(self) -> Dict[str, Any]:
        """分析时间分布"""
        print("📊 分析时间分布...")
        
        from dateutil import parser
        
        question_dates = []
        session_date_ranges = []
        
        for instance in self.instances:
            try:
                # 问题日期
                if instance.question_date:
                    try:
                        parsed_date = parser.parse(instance.question_date)
                        question_dates.append(parsed_date)
                    except:
                        pass  # 跳过无法解析的日期
                
                # 会话时间跨度
                if instance.haystack_dates and len(instance.haystack_dates) > 1:
                    try:
                        parsed_dates = []
                        for date_str in instance.haystack_dates:
                            if date_str:
                                parsed_dates.append(parser.parse(date_str))
                        
                        if len(parsed_dates) > 1:
                            parsed_dates.sort()
                            time_span = (parsed_dates[-1] - parsed_dates[0]).days
                            session_date_ranges.append(time_span)
                    except:
                        pass  # 跳过无法解析的日期
            except:
                continue  # 跳过有问题的实例
        
        return {
            'question_date_range': {
                'earliest': min(question_dates).isoformat() if question_dates else None,
                'latest': max(question_dates).isoformat() if question_dates else None,
                'total_days': (max(question_dates) - min(question_dates)).days if len(question_dates) > 1 else 0
            },
            'session_time_spans': {
                'min_days': min(session_date_ranges) if session_date_ranges else 0,
                'max_days': max(session_date_ranges) if session_date_ranges else 0,
                'avg_days': round(sum(session_date_ranges) / len(session_date_ranges), 2) if session_date_ranges else 0,
                'total_valid_spans': len(session_date_ranges)
            }
        }
    
    def generate_full_report(self) -> Dict[str, Any]:
        """生成完整的统计报告"""
        print(f"\n🔍 开始分析数据集: {self.data_file}")
        print(f"📁 加载了 {len(self.instances)} 个实例")
        
        report = {
            'dataset_info': {
                'file_path': self.data_file,
                'total_instances': len(self.instances),
                'analysis_time': datetime.now().isoformat()
            },
            'question_type_analysis': self.analyze_question_types(),
            'session_statistics': self.analyze_session_statistics(),
            'text_statistics': self.analyze_text_statistics(),
            'temporal_distribution': self.analyze_temporal_distribution()
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("📊 LongMemEval数据集统计摘要")
        print("="*60)
        
        # 基本信息
        dataset_info = report['dataset_info']
        print(f"📁 数据集文件: {dataset_info['file_path']}")
        print(f"📊 总实例数: {dataset_info['total_instances']}")
        
        # 问题类型分布
        qt_analysis = report['question_type_analysis']
        print(f"\n🏷️  问题类型分布:")
        print(f"   总问题数: {qt_analysis['total_questions']}")
        print(f"   普通问题: {qt_analysis['regular_questions']}")
        print(f"   Abstention问题: {qt_analysis['total_abstention']}")
        
        print(f"\n   各类型详细分布:")
        for q_type, stats in qt_analysis['detailed_stats'].items():
            print(f"   • {q_type}: {stats['total']} ({stats['percentage']}%)")
            print(f"     - 普通: {stats['regular']}, Abstention: {stats['abstention']}")
        
        # 会话统计
        session_stats = report['session_statistics']
        print(f"\n💬 会话统计:")
        print(f"   平均会话数/问题: {session_stats['sessions_per_question']['avg']}")
        print(f"   平均证据会话数/问题: {session_stats['evidence_sessions_per_question']['avg']}")
        print(f"   平均对话轮数/会话: {session_stats['turns_per_session']['avg']}")
        
        # haystack_sessions数量分布
        haystack_count_stats = session_stats['haystack_sessions_count']
        print(f"\n📚 Haystack Sessions数量分布:")
        print(f"   最少会话数: {haystack_count_stats['min']}")
        print(f"   最多会话数: {haystack_count_stats['max']}")
        print(f"   平均会话数: {haystack_count_stats['avg']}")
        print(f"   中位数会话数: {haystack_count_stats['median']}")
        
        # haystack_sessions token长度分布
        haystack_token_stats = session_stats['haystack_sessions_token_length']
        print(f"\n🔢 Haystack Sessions Token长度分布:")
        print(f"   最短token长度: {haystack_token_stats['min']:,}")
        print(f"   最长token长度: {haystack_token_stats['max']:,}")
        print(f"   平均token长度: {haystack_token_stats['avg']:,.2f}")
        print(f"   中位数token长度: {haystack_token_stats['median']:,}")
        print(f"   75%分位数: {haystack_token_stats['percentile_75']:,}")
        print(f"   90%分位数: {haystack_token_stats['percentile_90']:,}")
        
        # 文本统计
        text_stats = report['text_statistics']
        print(f"\n📝 文本统计:")
        print(f"   平均问题长度: {text_stats['question_lengths']['avg_chars']} 字符 (~{text_stats['question_lengths']['avg_tokens_est']} tokens)")
        print(f"   平均答案长度: {text_stats['answer_lengths']['avg_chars']} 字符 (~{text_stats['answer_lengths']['avg_tokens_est']} tokens)")
        print(f"   平均上下文长度: {text_stats['context_lengths']['avg_chars']} 字符 (~{text_stats['context_lengths']['avg_tokens_est']} tokens)")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="LongMemEval数据集统计分析")
    parser.add_argument(
        "--data-file", 
        default="data/longmemeval_data/longmemeval_oracle.json",
        help="数据集文件路径"
    )
    parser.add_argument(
        "--save-report", 
        action="store_true",
        help="保存详细报告到JSON文件"
    )
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"❌ 数据文件不存在: {args.data_file}")
        sys.exit(1)
    
    try:
        # 创建分析器并生成报告
        analyzer = DatasetAnalyzer(args.data_file)
        report = analyzer.generate_full_report()
        
        # 打印摘要
        analyzer.print_summary(report)
        
        # 可选保存详细报告
        if args.save_report:
            # 创建results目录
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = Path(args.data_file).stem
            report_file = results_dir / f"dataset_analysis_{dataset_name}_{timestamp}.json"
            
            # 保存报告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 详细报告已保存到: {report_file}")
        
        print(f"\n✅ 数据集分析完成!")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
