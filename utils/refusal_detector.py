"""
拒答检测模块 (Rule-Based)
========================

功能概述:
    基于规则的intentional refusal检测算法，专门用于识别LLM回复中的有意拒答行为。

核心设计:
    专注于检测intentional refusal - 即模型明确表示"不知道"、"无法回答"等拒绝回答的情况。
    这与"问题可回答性"不同 - 这里分析的是模型实际回复的意图和内容。

主要算法:
**基于规则的模式匹配**
   - 基于常见拒答模式的关键词匹配
   - 识别各种拒答表达方式
   - 快速准确的轻量级检测

**检测模式分类**
   - 明确否定: "I don't know", "I cannot"
   - 信息不足: "not enough information", "insufficient"
   - 不确定性: "I'm not sure", "unclear"  
   - 访问限制: "don't have access", "cannot find"

核心类:
    RefusalDetector: 快速的基于规则的拒答检测器

使用示例:
    ```python
    from utils.refusal_detector import RefusalDetector
    
    # 创建检测器
    detector = RefusalDetector()
    
    # 检测拒答
    is_refusal, confidence = detector.detect_refusal("I don't know the answer.")
    print(f"Refusal: {is_refusal}, Confidence: {confidence:.3f}")
    ```

应用场景:
    - RLHF研究中的拒答行为分析
    - 对话系统中的intentional refusal检测
    - 大模型安全性和保守性评估
    - 交互式记忆任务中的拒答识别

性能特点:
    - 轻量级规则检测，速度极快 (~1ms)
    - 无外部依赖，纯Python实现
    - 专门针对intentional refusal优化
    - 支持英文拒答模式，可扩展其他语言
"""

import re
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class RefusalPattern:
    """拒答模式定义"""
    pattern: str
    confidence: float
    description: str
    case_sensitive: bool = False


class RefusalDetector:
    """
    基于规则的拒答检测器
    
    专门用于检测LLM回复中的intentional refusal，
    即模型有意拒绝回答用户问题的情况。
    
    特点:
        - 快速轻量级检测
        - 专注intentional refusal识别
        - 基于真实拒答模式设计
        - 无外部依赖
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        初始化拒答检测器
        
        Args:
            confidence_threshold: 拒答判断阈值，高于此值认为是拒答
        """
        self.confidence_threshold = confidence_threshold
        self.refusal_patterns = self._init_refusal_patterns()
    
    def _init_refusal_patterns(self) -> List[RefusalPattern]:
        """初始化拒答模式列表"""
        patterns = [
            # 明确否定型 - 高置信度
            RefusalPattern(r"\bi don'?t know\b", 0.95, "explicit_dont_know"),
            RefusalPattern(r"\bi cannot\b", 0.90, "explicit_cannot"),
            RefusalPattern(r"\bi can'?t\b", 0.90, "explicit_cant"),
            RefusalPattern(r"\bi don'?t have\b", 0.85, "explicit_dont_have"),
            RefusalPattern(r"\bi do not know\b", 0.95, "formal_negation"),
            RefusalPattern(r"\bi am not sure\b", 0.80, "explicit_not_sure"),
            RefusalPattern(r"\bi'm not sure\b", 0.80, "not_sure"),
            
            # 信息不足型 - 高置信度
            RefusalPattern(r"not enough information", 0.90, "insufficient_info"),
            RefusalPattern(r"insufficient information", 0.90, "insufficient_info_formal"),
            RefusalPattern(r"lack.*information", 0.85, "lack_information"),
            RefusalPattern(r"don'?t have.*information", 0.85, "no_information"),
            RefusalPattern(r"no information", 0.85, "no_info"),
            
            # 无法确定型 - 中等置信度
            RefusalPattern(r"cannot determine", 0.85, "cannot_determine"),
            RefusalPattern(r"can'?t determine", 0.85, "cant_determine"),
            RefusalPattern(r"unable to determine", 0.85, "unable_determine"),
            RefusalPattern(r"difficult to say", 0.75, "difficult_say"),
            RefusalPattern(r"hard to say", 0.75, "hard_say"),
            
            # 访问限制型 - 高置信度
            RefusalPattern(r"don'?t have access", 0.90, "no_access"),
            RefusalPattern(r"cannot access", 0.90, "cannot_access"),
            RefusalPattern(r"no access to", 0.85, "no_access_to"),
            RefusalPattern(r"cannot find", 0.80, "cannot_find"),
            RefusalPattern(r"can'?t find", 0.80, "cant_find"),
            
            # 记录缺失型 - 中等置信度
            RefusalPattern(r"not mentioned", 0.75, "not_mentioned"),
            RefusalPattern(r"wasn'?t mentioned", 0.75, "wasnt_mentioned"),
            RefusalPattern(r"not discussed", 0.75, "not_discussed"),
            RefusalPattern(r"not provided", 0.80, "not_provided"),
            RefusalPattern(r"no mention", 0.75, "no_mention"),
            
            # 委婉拒答型 - 新增高置信度模式 
            RefusalPattern(r"conversation history doesn'?t contain", 0.90, "history_missing"),
            RefusalPattern(r"conversation history does not contain", 0.90, "history_no_contain"),
            RefusalPattern(r"doesn'?t contain information", 0.85, "no_contain_info"),
            RefusalPattern(r"does not contain information", 0.85, "no_contain_info_formal"),
            RefusalPattern(r"there is no mention", 0.85, "explicit_no_mention"),
            RefusalPattern(r"no information about", 0.80, "no_info_about"),
            RefusalPattern(r"could you please provide", 0.85, "polite_request_info"),
            RefusalPattern(r"can you provide", 0.80, "request_more_info"),
            RefusalPattern(r"please provide more", 0.85, "need_more_info"),
            RefusalPattern(r"we need to know", 0.75, "need_to_know"),
            RefusalPattern(r"would need more", 0.75, "would_need_more"),
            RefusalPattern(r"to determine.*need", 0.75, "need_for_determination"),
            
            # 记忆相关拒答 - 高置信度
            RefusalPattern(r"don'?t recall", 0.85, "dont_recall"),
            RefusalPattern(r"cannot recall", 0.85, "cannot_recall"),
            RefusalPattern(r"no record", 0.80, "no_record"),
            RefusalPattern(r"not in.*conversation", 0.80, "not_in_conversation"),
            RefusalPattern(r"not in.*history", 0.80, "not_in_history"),
            
            # 模糊拒答型 - 较低置信度
            RefusalPattern(r"unclear", 0.70, "unclear"),
            RefusalPattern(r"uncertain", 0.70, "uncertain"),
            RefusalPattern(r"not clear", 0.70, "not_clear"),
            RefusalPattern(r"unsure", 0.75, "unsure"),
            
            # 直接拒答型 - 高置信度
            RefusalPattern(r"cannot answer", 0.90, "cannot_answer"),
            RefusalPattern(r"can'?t answer", 0.90, "cant_answer"),
            RefusalPattern(r"unable to answer", 0.90, "unable_answer"),
            RefusalPattern(r"don'?t answer", 0.85, "dont_answer"),
            
            # 单词拒答型 - 高置信度(针对极短回复)
            RefusalPattern(r"^unknown\.?$", 0.95, "single_word_unknown"),
            RefusalPattern(r"^unclear\.?$", 0.90, "single_word_unclear"),
            RefusalPattern(r"^unsure\.?$", 0.90, "single_word_unsure"),
        ]
        
        return patterns
    
    def detect_refusal(self, response: str, question: str = "") -> Tuple[bool, float]:
        """
        检测回复是否为intentional refusal
        
        Args:
            response: 模型回复文本
            question: 原始问题 (可选，用于增强检测)
            
        Returns:
            Tuple[bool, float]: (is_refusal, refusal_confidence)
                - is_refusal: True表示检测为拒答，False表示正常回答
                - confidence: 拒答置信度，范围[0,1]，越高越可能是拒答
        """
        # 基本检查
        if not response or len(response.strip()) == 0:
            return True, 1.0  # 空回复视为拒答
        
        # 清理和标准化文本
        cleaned_response = self._clean_text(response)
        
        # 如果回复过短且非实质内容，可能是拒答
        if len(cleaned_response.split()) <= 2:
            return self._detect_short_response(cleaned_response)
        
        # 模式匹配检测
        max_confidence = 0.0
        matched_patterns = []
        
        for pattern in self.refusal_patterns:
            if re.search(pattern.pattern, cleaned_response, re.IGNORECASE):
                max_confidence = max(max_confidence, pattern.confidence)
                matched_patterns.append(pattern.description)
        
        # 组合模式加权
        if len(matched_patterns) > 1:
            # 多个模式匹配，增加置信度
            max_confidence = min(1.0, max_confidence + 0.1 * (len(matched_patterns) - 1))
        
        # 特殊组合检测：礼貌性拒答
        if self._detect_polite_refusal_combination(cleaned_response):
            max_confidence = max(max_confidence, 0.85)
        
        # 上下文增强(如果提供了问题)
        if question:
            enhanced_confidence = self._enhance_with_question_context(
                cleaned_response, question, max_confidence
            )
            max_confidence = max(max_confidence, enhanced_confidence)
        
        is_refusal = max_confidence >= self.confidence_threshold
        
        return is_refusal, max_confidence
    
    def _detect_polite_refusal_combination(self, response: str) -> bool:
        """
        检测礼貌性拒答组合模式
        如："I'm sorry, but the conversation doesn't contain information..."
        """
        polite_indicators = [
            r"i'?m sorry,?\s*but",
            r"sorry,?\s*but", 
            r"unfortunately", 
            r"however,?\s*(?:there|the)",
        ]
        
        information_lack = [
            r"doesn'?t contain",
            r"does not contain", 
            r"no information",
            r"not mentioned",
            r"no mention",
            r"would need",
            r"could you provide",
            r"please provide"
        ]
        
        # 检查是否同时包含礼貌指示词和信息缺失指示词
        has_polite = any(re.search(pattern, response, re.IGNORECASE) for pattern in polite_indicators)
        has_lack = any(re.search(pattern, response, re.IGNORECASE) for pattern in information_lack)
        
        return has_polite and has_lack
    
    def _clean_text(self, text: str) -> str:
        """清理和标准化文本"""
        # 移除多余空白
        cleaned = re.sub(r'\s+', ' ', text.strip())
        # 移除常见标点
        cleaned = re.sub(r'[,!?.]', '', cleaned)
        return cleaned.lower()
    
    def _detect_short_response(self, response: str) -> Tuple[bool, float]:
        """检测短回复是否为拒答"""
        # 常见的短拒答词汇
        short_refusals = {
            'unknown': 0.95,
            'unclear': 0.90,
            'unsure': 0.90,
            "don't know": 0.95,
            "not sure": 0.85,
            "can't say": 0.85,
            'uncertain': 0.80,
            'maybe': 0.60,  # 较低置信度
            'possibly': 0.60,
        }
        
        cleaned = response.strip().lower()
        for phrase, conf in short_refusals.items():
            if phrase in cleaned:
                return True, conf
        
        # 如果是极短但不匹配的回复，保守处理
        if len(response.split()) == 1 and len(response) < 5:
            return True, 0.70  # 中等置信度拒答
        
        return False, 0.0
    
    def _enhance_with_question_context(self, response: str, question: str, base_confidence: float) -> float:
        """基于问题上下文增强检测"""
        # 如果问题是询问个人信息，而回答包含拒答关键词，提高置信度
        personal_question_patterns = [
            r'\bmy\b', r'\byour\b', r'\bi\b', r'\byou\b',
            r'name', r'age', r'address', r'phone', r'email'
        ]
        
        is_personal = any(re.search(pattern, question.lower()) for pattern in personal_question_patterns)
        
        if is_personal and base_confidence > 0.5:
            return min(1.0, base_confidence + 0.15)
        
        return base_confidence
    
    def get_detection_method(self) -> str:
        """返回当前使用的检测方法"""
        return "Rule-based"
    
    def get_detailed_analysis(self, response: str) -> Dict:
        """
        返回详细的检测分析信息
        
        Args:
            response: 待分析的回复文本
            
        Returns:
            dict: 包含详细分析信息的字典
        """
        is_refusal, confidence = self.detect_refusal(response)
        
        # 找出匹配的模式
        cleaned_response = self._clean_text(response)
        matched_patterns = []
        
        for pattern in self.refusal_patterns:
            if re.search(pattern.pattern, cleaned_response, re.IGNORECASE):
                matched_patterns.append({
                    'pattern': pattern.description,
                    'confidence': pattern.confidence
                })
        
        analysis = {
            "response": response[:100] + "..." if len(response) > 100 else response,
            "is_refusal": is_refusal,
            "confidence": confidence,
            "method": self.get_detection_method(),
            "threshold": self.confidence_threshold,
            "matched_patterns": matched_patterns,
            "response_length": len(response.split()),
            "cleaned_response": cleaned_response[:50] + "..." if len(cleaned_response) > 50 else cleaned_response
        }
        
        return analysis
    
    def batch_detect(self, responses: List[str]) -> List[Tuple[bool, float]]:
        """
        批量检测多个回复
        
        Args:
            responses: 回复文本列表
            
        Returns:
            list: 包含(is_refusal, confidence)元组的列表
        """
        results = []
        for response in responses:
            is_refusal, confidence = self.detect_refusal(response)
            results.append((is_refusal, confidence))
        return results


# 便捷函数
def detect_refusal_simple(response: str, confidence_threshold: float = 0.7) -> bool:
    """
    简化的拒答检测函数，只返回是否拒答的布尔值
    
    Args:
        response: 待检测的回复文本
        confidence_threshold: 置信度阈值
        
    Returns:
        bool: True表示拒答，False表示正常回答
    """
    detector = RefusalDetector(confidence_threshold=confidence_threshold)
    is_refusal, _ = detector.detect_refusal(response)
    return is_refusal


def batch_detect_refusal(responses: List[str], confidence_threshold: float = 0.7) -> List[Tuple[bool, float]]:
    """
    批量拒答检测函数
    
    Args:
        responses: 回复文本列表
        confidence_threshold: 置信度阈值
        
    Returns:
        list: 包含(is_refusal, confidence)元组的列表
    """
    detector = RefusalDetector(confidence_threshold=confidence_threshold)
    return detector.batch_detect(responses)


if __name__ == "__main__":
    # 测试代码
    print("🧪 规则拒答检测器测试")
    print("=" * 40)
    
    # 测试用例
    test_cases = [
        # 明确拒答
        "I don't know the answer to that question.",
        "I cannot find enough information to answer.",
        "I'm not sure about this.",
        "I don't have access to that information.",
        "Unable to determine based on the conversation.",
        "Unknown.",
        "Unclear.",
        
        # 正常回答
        "Based on the conversation, the user mentioned pizza.",
        "The answer is 42, as mentioned earlier.",
        "According to the chat history, the meeting is tomorrow.",
        "Your favorite color is blue.",
        "You graduated in 2020.",
        "You have 2 cats.",
        
        # 边缘案例
        "It might be true, but I'm not completely certain.",
        "Possibly, but I need more information.",
        "That's an interesting question.",
    ]
    
    try:
        # 创建检测器
        detector = RefusalDetector()
        print(f"检测方法: {detector.get_detection_method()}")
        print(f"置信度阈值: {detector.confidence_threshold}")
        print()
        
        # 测试每个案例
        for i, text in enumerate(test_cases, 1):
            is_refusal, confidence = detector.detect_refusal(text)
            status = "拒答" if is_refusal else "回答"
            print(f"案例 {i}: {status} (置信度: {confidence:.3f})")
            print(f"  文本: {text}")
            
            # 详细分析
            if is_refusal:
                analysis = detector.get_detailed_analysis(text)
                if analysis['matched_patterns']:
                    patterns = [p['pattern'] for p in analysis['matched_patterns']]
                    print(f"  匹配模式: {', '.join(patterns)}")
            print()
        
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")