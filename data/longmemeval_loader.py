"""
LongMemEval数据集加载器
用于加载和处理LongMemEval数据集，支持RQ2和RQ4实验
"""
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LongMemEvalInstance:
    """LongMemEval数据实例"""
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_session_ids: List[str]
    haystack_dates: List[str]
    haystack_sessions: List[List[Dict[str, Any]]]
    answer_session_ids: List[str]
    
    @property
    def is_abstention(self) -> bool:
        """判断是否为拒答问题"""
        return self.question_id.endswith('_abs')
    
    @property
    def is_knowledge_update(self) -> bool:
        """判断是否为知识更新问题"""
        return self.question_type == 'knowledge-update'
    
    @property
    def has_evidence_in_context(self) -> bool:
        """检查上下文中是否包含证据"""
        for session in self.haystack_sessions:
            for turn in session:
                if turn.get('has_answer', False):
                    return True
        return False


class LongMemEvalLoader:
    """LongMemEval数据集加载器"""
    
    def __init__(self, data_path: str):
        """
        初始化加载器
        
        Args:
            data_path: LongMemEval数据文件路径
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    def load_data(self) -> List[LongMemEvalInstance]:
        """加载所有数据实例"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        instances = []
        for item in data:
            instance = LongMemEvalInstance(
                question_id=item['question_id'],
                question_type=item['question_type'],
                question=item['question'],
                answer=item['answer'],
                question_date=item['question_date'],
                haystack_session_ids=item['haystack_session_ids'],
                haystack_dates=item['haystack_dates'],
                haystack_sessions=item['haystack_sessions'],
                answer_session_ids=item['answer_session_ids']
            )
            instances.append(instance)
        
        return instances
    
    def get_rq2_instances(self) -> List[LongMemEvalInstance]:
        """
        获取RQ2实验相关的数据实例
        RQ2关注过度拒答：在有明确证据的情况下模型是否错误拒答
        """
        instances = self.load_data()
        
        # 筛选有证据且非拒答问题的实例
        rq2_instances = []
        for instance in instances:
            # 排除设计为拒答的问题
            if instance.is_abstention:
                continue
            
            # 只包含上下文中有证据的问题
            if instance.has_evidence_in_context:
                rq2_instances.append(instance)
        
        return rq2_instances
    
    def get_rq4_instances(self) -> List[LongMemEvalInstance]:
        """
        获取RQ4实验相关的数据实例
        RQ4关注知识更新一致性
        """
        instances = self.load_data()
        
        # 筛选知识更新相关的实例
        rq4_instances = [
            instance for instance in instances 
            if instance.is_knowledge_update
        ]
        
        return rq4_instances
    
    def get_abstention_instances(self) -> List[LongMemEvalInstance]:
        """获取设计为拒答的问题实例"""
        instances = self.load_data()
        
        abstention_instances = [
            instance for instance in instances 
            if instance.is_abstention
        ]
        
        return abstention_instances
    
    def format_conversation_history(self, instance: LongMemEvalInstance, 
                                  max_sessions: Optional[int] = None) -> str:
        """
        格式化对话历史为模型输入
        
        Args:
            instance: 数据实例
            max_sessions: 最大会话数量限制
            
        Returns:
            格式化的对话历史字符串
        """
        sessions = instance.haystack_sessions
        if max_sessions:
            sessions = sessions[:max_sessions]
        
        formatted_history = []
        for i, session in enumerate(sessions):
            session_date = instance.haystack_dates[i] if i < len(instance.haystack_dates) else "Unknown"
            formatted_history.append(f"=== Session {i+1} ({session_date}) ===")
            
            for turn in session:
                role = turn['role']
                content = turn['content']
                formatted_history.append(f"{role.title()}: {content}")
            
            formatted_history.append("")  # 空行分隔会话
        
        return "\n".join(formatted_history)


def load_longmemeval_datasets(base_path: str) -> Dict[str, LongMemEvalLoader]:
    """
    加载所有LongMemEval数据集变体
    
    Args:
        base_path: LongMemEval数据目录路径
        
    Returns:
        包含各个数据集变体的字典
    """
    base_path = Path(base_path)
    loaders = {}
    
    # 各个数据集文件
    dataset_files = {
        'small': 'longmemeval_s.json',
        'medium': 'longmemeval_m.json', 
        'oracle': 'longmemeval_oracle.json'
    }
    
    for variant, filename in dataset_files.items():
        file_path = base_path / filename
        if file_path.exists():
            loaders[variant] = LongMemEvalLoader(str(file_path))
        else:
            print(f"警告: 数据文件不存在 {file_path}")
    
    return loaders


if __name__ == "__main__":
    # 测试代码
    data_path = "data/longmemeval_data"  # WSL路径
    loaders = load_longmemeval_datasets(data_path)
    
    if 'oracle' in loaders:
        loader = loaders['oracle']
        
        # 测试RQ2数据
        rq2_instances = loader.get_rq2_instances()
        print(f"RQ2实验数据: {len(rq2_instances)}个实例")
        
        # 测试RQ4数据
        rq4_instances = loader.get_rq4_instances()
        print(f"RQ4实验数据: {len(rq4_instances)}个实例")
        
        # 测试拒答数据
        abs_instances = loader.get_abstention_instances()
        print(f"拒答问题: {len(abs_instances)}个实例")
