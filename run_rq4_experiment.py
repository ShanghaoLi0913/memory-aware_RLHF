#!/usr/bin/env python3
"""
RQ4实验启动器: 知识更新（Knowledge Update）一致性与拒答分析
=============================================================

功能概述:
    本脚本是 RQ4 实验的用户友好前端，评估 Base vs RLHF 模型在“知识更新(KU)”场景中
    是否能正确使用最新信息，还是更倾向使用旧信息或产生拒答。

🎯 核心研究问题 (RQ4)
    在包含事实更新的上下文中，RLHF 模型是否更难使用最新信息，出现“答旧/拒答”的偏差？

🔬 实验方法
    - 数据: LongMemEval 的 KU 子集（72 个应答样本 + 6 个 _abs 样本）
    - 指标:
        1) UC (Update Consistency): 使用最新事实回答的比例
        2) SAR (Stale Answer Rate): 使用旧事实回答的比例
        3) ORR (Overall Refusal Rate in KU): 在应答型 KU 样本上的拒答比例
    - 统计:
        - UC/SAR: McNemar（配对，correction=True）
        - ORR: Wilson 95% CI（可补充 bootstrap 做稳健性）

📊 支持的模型配置（与RQ2保持一致的预设名）
    1. qwen2.5-3b    2. llama3.2-3b    3. mistral-7b

🚀 快速开始命令
    # 预设（--config / --model-pair 与 RQ2 一致）
    # --config 仅用于选择预设模型对（qwen2.5-3b / llama3.2-3b / mistral-7b），
    # KU 专属参数（如 --max-sessions / --temperature / --data 等）通过命令行单独控制。
    python run_rq4_experiment.py --config qwen2.5-3b --quick-test
    python run_rq4_experiment.py --model-pair qwen2.5-3b --quick-test
    # 自定义模型对
    python run_rq4_experiment.py --model-pair \
        "/root/autodl-tmp/models/qwen/Qwen2.5-3B,/root/autodl-tmp/models/qwen/Qwen2.5-3B-Instruct" --quick-test

🧰 常用参数（与RQ2对齐）
    --config/-c, --model-pair/-m, --quick-test, --list-configs/-l, --test-only/-t
    以及 KU 专属：--max-sessions, --temperature, --top-p, --max-new-tokens, --data, --out

📁 输出结构（与RQ2风格对齐）
    results/rq4_knowledge_update/
      └── rq4_<pair_name>[_quick_test]/
          ├── rq4_base_ku_responses_<TIMESTAMP>.json
          ├── rq4_rlhf_ku_responses_<TIMESTAMP>.json
          ├── rq4_analysis_<TIMESTAMP>.json                 # 含 UC/SAR/ORR + 统计
          └── rq4_analysis_<TIMESTAMP>_annotated.json       # 明细：每条分类与响应

🛠 故障排除
    - CUDA OOM: 降低并发(=1)、减少 max_sessions、缩短 max_new_tokens
    - 路径问题: 使用 experiments/config.py 中的本地路径别名
    - 结果为空: 检查数据文件与 KU 抽取逻辑（haystack_sessions 结构）

📎 依赖文件
    - experiments/rq4_knowledge_update.py  (核心实验引擎)
    - experiments/config.py                (模型路径/配置)
    - data/longmemeval_data/longmemeval_oracle.json
    - utils/refusal_detector.py            (拒答检测)
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiments.config import get_model_config, get_rq2_config, list_available_configs
from experiments.rq4_knowledge_update import RQ4Config, run_rq4_experiment


def main():
    parser = argparse.ArgumentParser(description="RQ4: Knowledge Update")
    # 与 RQ2 对齐的参数
    parser.add_argument("--config", "-c", default="qwen2.5-3b", help="实验配置名称 (qwen2.5-3b, llama3.2-3b, mistral-7b, long_context)")
    parser.add_argument("--test-only", "-t", action="store_true", help="只运行环境测试，不执行实际实验")
    parser.add_argument("--list-configs", "-l", action="store_true", help="列出所有可用配置")
    parser.add_argument("--model-pair", "-m", help="指定模型对 (格式1: 预设配置名如 'qwen2.5-3b'; 格式2: 自定义 'base_model,instruct_model')")
    parser.add_argument("--quick-test", action="store_true", help="快速测试模式 (小样本抽样)")

    # RQ4 专属细节
    parser.add_argument("--max-sessions", type=int, default=2, help="每条样本最多拼接的会话段数")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--data", default="data/longmemeval_data/longmemeval_oracle.json")
    parser.add_argument("--out", default="results/rq4_knowledge_update")

    args = parser.parse_args()

    if args.list_configs:
        list_available_configs()
        return

    if args.test_only:
        # 轻量环境测试：检查数据存在与最小字段
        from pathlib import Path
        ok = Path(args.data).exists()
        print(f"数据文件存在: {ok} -> {args.data}")
        if not ok:
            raise SystemExit(1)
        print("✅ 环境基本检查通过。")
        return

    def cfg_from_models(base_model: str, rlhf_model: str) -> RQ4Config:
        try:
            base_cfg = get_model_config(base_model)
            base_name = base_cfg.path
        except Exception:
            base_name = base_model
        try:
            rlhf_cfg = get_model_config(rlhf_model)
            rlhf_name = rlhf_cfg.path
        except Exception:
            rlhf_name = rlhf_model
        return RQ4Config(
            base_model_name=base_name,
            rlhf_model_name=rlhf_name,
            longmemeval_path=args.data,
            output_dir=args.out,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_sessions=args.max_sessions,
            quick_test=args.quick_test,
            save_responses=True,
        )

    def cfg_from_preset(preset_name: str) -> RQ4Config:
        preset = get_rq2_config(preset_name)
        if "model_pairs" in preset:
            base_model_name, rlhf_model_name = preset["model_pairs"][0]
        else:
            base_model_name = preset["base_model"]
            rlhf_model_name = preset["rlhf_model"]
        return cfg_from_models(base_model_name, rlhf_model_name)

    # 处理模型对参数（优先）
    if args.model_pair:
        if "," in args.model_pair:
            base_model, instruct_model = args.model_pair.split(",")
            print(f"🎯 运行自定义模型对 RQ4: {base_model.strip()} vs {instruct_model.strip()}")
            cfg = cfg_from_models(base_model.strip(), instruct_model.strip())
        else:
            preset_name = args.model_pair
            print(f"🎯 运行预设 RQ4: {preset_name}")
            cfg = cfg_from_preset(preset_name)
            if args.quick_test:
                cfg.quick_test = True
        run_rq4_experiment(cfg)
        return

    # 若未提供 --model-pair，则使用 --config
    print(f"🚀 运行RQ4实验，配置: {args.config}")
    cfg = cfg_from_preset(args.config)
    if args.quick_test:
        cfg.quick_test = True
    run_rq4_experiment(cfg)


if __name__ == "__main__":
    main()


