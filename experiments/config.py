"""
实验配置文件
包含RQ2和RQ4实验的各种配置参数
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    path: str
    max_length: int = 4096
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    """文本生成配置"""
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class DataConfig:
    """数据配置"""
    longmemeval_path: str = "data/longmemeval_data"  # 相对路径
    dataset_variant: str = "oracle"  # oracle, small, medium
    max_sessions: int = 20
    max_context_length: int = 8192


@dataclass
class ExperimentConfig:
    """实验配置"""
    output_dir: str = "results"
    save_responses: bool = True
    save_prompts: bool = False
    batch_size: int = 1
    random_seed: int = 42


# 预定义的模型配置 - 只保留用户指定的模型
MODELS = {
    # Llama-3.2 3B模型配置 (128K上下文, 12GB GPU友好)
    "llama3.2-3b-base": ModelConfig(
        name="llama3.2-3b-base",
        path="meta-llama/Llama-3.2-3B",
    ),
    "llama3.2-3b-instruct": ModelConfig(
        name="llama3.2-3b-instruct", 
        path="meta-llama/Llama-3.2-3B-Instruct",
    ),
    
    # Qwen2.5 3B模型配置 (12GB GPU友好) - 使用本地路径
    "qwen2.5-3b-base": ModelConfig(
        name="qwen2.5-3b-base",
        path="/root/autodl-tmp/models/qwen/Qwen2.5-3B",  # Base模型
    ),
    "qwen2.5-3b-instruct": ModelConfig(
        name="qwen2.5-3b-instruct",
        path="/root/autodl-tmp/models/qwen/Qwen2.5-3B-Instruct",  # Instruct模型
    ),
    
    # Mistral-7B-v0.3 模型配置 (云上运行)
    "mistral-7b-base": ModelConfig(
        name="mistral-7b-base",
        path="mistralai/Mistral-7B-v0.3",
    ),
    "mistral-7b-instruct": ModelConfig(
        name="mistral-7b-instruct",
        path="mistralai/Mistral-7B-Instruct-v0.3",
    ),
}


# RQ2实验配置 - 只保留用户指定的模型配置
RQ2_EXPERIMENT_CONFIGS = {
    # 长上下文模型配置
    "long_context": {
        "description": "长上下文模型对比实验 - Llama3.2-3B,Qwen2.5-3B, 和Mistral-7B-v0.3",
        "model_pairs": [
            ("qwen2.5-3b-base", "qwen2.5-3b-instruct"),
            ("llama3.2-3b-base", "llama3.2-3b-instruct"),
            ("mistral-7b-base", "mistral-7b-instruct"),
        ],
        "data_config": DataConfig(
            dataset_variant="oracle",
            max_sessions=None,        # 不限制会话数
            max_context_length=None   # 不限制上下文长度，使用完整数据
        ),
        "generation_config": GenerationConfig(
            temperature=0.1,
            max_new_tokens=200,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        "experiment_config": ExperimentConfig(
            output_dir="results/rq2_long_context",
            save_responses=True,
            save_prompts=True,
            batch_size=1
        )
    },
    
    # 单模型对配置 - 方便选择特定模型
    "qwen2.5-3b": {
        "description": "Qwen2.5-3B Base vs Instruct 对比实验",
        "model_pairs": [
            ("qwen2.5-3b-base", "qwen2.5-3b-instruct"),
        ],
        "data_config": DataConfig(
            dataset_variant="oracle",
            max_sessions=None,
            max_context_length=None
        ),
        "generation_config": GenerationConfig(
            temperature=0.1,
            max_new_tokens=200,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        "experiment_config": ExperimentConfig(
            output_dir="results/rq2_qwen2.5_3b",
            save_responses=True,
            save_prompts=True,
            batch_size=1
        )
    },
    
    "llama3.2-3b": {
        "description": "Llama-3.2-3B Base vs Instruct 对比实验",
        "model_pairs": [
            ("llama3.2-3b-base", "llama3.2-3b-instruct"),
        ],
        "data_config": DataConfig(
            dataset_variant="oracle",
            max_sessions=None,
            max_context_length=None
        ),
        "generation_config": GenerationConfig(
            temperature=0.1,
            max_new_tokens=200,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        "experiment_config": ExperimentConfig(
            output_dir="results/rq2_llama3.2_3b",
            save_responses=True,
            save_prompts=True,
            batch_size=1
        )
    },
    
    "mistral-7b": {
        "description": "Mistral-7B Base vs Instruct 对比实验 (云上运行)",
        "model_pairs": [
            ("mistral-7b-base", "mistral-7b-instruct"),
        ],
        "data_config": DataConfig(
            dataset_variant="oracle",
            max_sessions=None,
            max_context_length=None
        ),
        "generation_config": GenerationConfig(
            temperature=0.1,
            max_new_tokens=200,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        "experiment_config": ExperimentConfig(
            output_dir="results/rq2_mistral_7b",
            save_responses=True,
            save_prompts=True,
            batch_size=1
        )
    }
}


# RQ4实验配置
RQ4_EXPERIMENT_CONFIGS = {
    "default": {
        "base_model": "llama2-7b-base",
        "rlhf_model": "llama2-7b-chat", 
        "data_config": DataConfig(
            dataset_variant="oracle",
            max_sessions=25  # 知识更新可能需要更多上下文
        ),
        "generation_config": GenerationConfig(
            temperature=0.1,
            max_new_tokens=512
        ),
        "experiment_config": ExperimentConfig(
            output_dir="results/rq4_default"
        )
    },
    
    # 云端全上下文配置 - 使用28K tokens覆盖100%数据
    "cloud_full_context": {
        "description": "云端RTX 4090完整上下文实验 - 覆盖100%数据(28K tokens)",
        "model_pairs": [
            ("qwen2.5-3b-base", "qwen2.5-3b-instruct"),
            ("llama3.2-3b-base", "llama3.2-3b-instruct"), 
            ("mistral-7b-base", "mistral-7b-instruct"),
        ],
        "data_config": DataConfig(
            dataset_variant="oracle",
            max_sessions=None,
            max_context_length=28000  # 覆盖100%数据
        ),
        "generation_config": GenerationConfig(
            temperature=0.1,
            max_new_tokens=100,  # 优化输出长度
            top_p=0.9,
            repetition_penalty=1.1
        ),
        "experiment_config": ExperimentConfig(
            output_dir="results/rq2_cloud_full_context",
            save_responses=True,
            save_prompts=True,
            batch_size=1
        )
    },
}


def get_model_config(model_name: str) -> ModelConfig:
    """获取模型配置"""
    if model_name not in MODELS:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(MODELS.keys())}")
    return MODELS[model_name]


def get_rq2_config(config_name: str = "default") -> Dict:
    """获取RQ2实验配置"""
    if config_name not in RQ2_EXPERIMENT_CONFIGS:
        raise ValueError(f"未知RQ2配置: {config_name}. 可用配置: {list(RQ2_EXPERIMENT_CONFIGS.keys())}")
    return RQ2_EXPERIMENT_CONFIGS[config_name]


def get_rq4_config(config_name: str = "default") -> Dict:
    """获取RQ4实验配置"""
    if config_name not in RQ4_EXPERIMENT_CONFIGS:
        raise ValueError(f"未知RQ4配置: {config_name}. 可用配置: {list(RQ4_EXPERIMENT_CONFIGS.keys())}")
    return RQ4_EXPERIMENT_CONFIGS[config_name]


def list_available_configs():
    """列出所有可用配置"""
    print("🔧 可用配置:")
    print("\n📊 模型配置:")
    for name, config in MODELS.items():
        print(f"  {name}: {config.path}")
    
    print("\n🔍 RQ2实验配置:")
    for name in RQ2_EXPERIMENT_CONFIGS.keys():
        print(f"  {name}")
    
    print("\n🔄 RQ4实验配置:")
    for name in RQ4_EXPERIMENT_CONFIGS.keys():
        print(f"  {name}")


if __name__ == "__main__":
    list_available_configs()
