#!/usr/bin/env python3
"""
RQ4 实验：Knowledge Update（KU）下的更新一致性与拒答行为
-----------------------------------------------------------------

研究问题：在包含事实更新的上下文中，RLHF 模型是否不能准确使用最新信息，
而更可能回答旧信息或拒答？

指标：
- Update Consistency (UC): 使用最新事实回答的比例
- Stale Answer Rate (SAR): 使用旧事实回答的比例
- Overall Refusal Rate (ORR): 在应答型 KU（不含 _abs）中选择拒答的比例

实现说明：
- 数据源：LongMemEval（JSON），自动提取 KU 候选（满足有两个 haystack_sessions，
  第一段含旧答案、第二段含新答案），并以 question_id 是否包含 "_abs" 标记应拒答样本。
- 推理：与 RQ2 一致的本地 HF 模型加载逻辑（fp16, device_map="auto"）。
- 拒答检测：复用 utils/refusal_detector.RefusalDetector。
- 统计：UC/SAR 使用 McNemar（连续性修正）；ORR 给出 Wilson 95% CI。
"""

from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.refusal_detector import RefusalDetector
from experiments.config import get_model_config


@dataclass
class RQ4Config:
    base_model_name: str
    rlhf_model_name: str
    longmemeval_path: str = "data/longmemeval_data/longmemeval_oracle.json"
    output_dir: str = "results/rq4_knowledge_update"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    max_sessions: Optional[int] = None  # 限制每条样本拼接的会话轮数
    save_responses: bool = True
    quick_test: bool = False  # 仅取前若干 KU 样本


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _current_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    return "".join(ch.lower() for ch in text if ch.isalnum())


def construct_prompt(sessions: List[Dict[str, Any]], question: str, max_sessions: Optional[int]) -> str:
    if max_sessions is not None and max_sessions > 0:
        sessions = sessions[:max_sessions]
    history_lines: List[str] = []

    def add_entry(entry: Any) -> None:
        if isinstance(entry, dict):
            content = entry.get("content", "")
            role = entry.get("role", "assistant")
            if content:
                history_lines.append(f"{role}: {content}")
        elif isinstance(entry, list):
            for e in entry:
                add_entry(e)
        else:
            # Fallback to string
            text = str(entry)
            if text:
                history_lines.append(f"assistant: {text}")

    for s in sessions:
        add_entry(s)
    history = "\n\n".join(history_lines)
    return (
        "You are given a user–assistant chat history.\n"
        "Answer the user’s last question only using the information in the history.\n"
        "If the history does not contain enough information, say you cannot answer.\n\n"
        f"History:\n{history}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_model(model_name_or_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        cfg = get_model_config(model_name_or_path)
        model_path = cfg.path
        print(f"📁 使用配置路径: {model_path}")
    except Exception:
        model_path = model_name_or_path
        print(f"⚠️ 配置中未找到 {model_name_or_path}，使用原始路径: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tok


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **inputs,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # Return only the suffix after prompt
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


def mcnemar_with_correction(b: int, c: int) -> Dict[str, Any]:
    # 统计量 with continuity correction
    total = b + c
    if total == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / total
    try:
        from scipy.stats import chi2 as _chi2
        p_value = 1 - _chi2.cdf(chi2, df=1)
    except Exception:
        # df=1: survival = erfc(sqrt(x/2))
        p_value = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return {"statistic": float(chi2), "p_value": float(p_value)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    return phat, lo, hi


def _entry_has_answer(entry: Any) -> bool:
    # entry can be dict, list, or other
    if isinstance(entry, dict):
        return bool(entry.get("has_answer", False))
    if isinstance(entry, list):
        return any(_entry_has_answer(e) for e in entry)
    return False


def extract_ku_samples(dataset: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ku_items: List[Dict[str, Any]] = []
    for item in dataset:
        sessions = item.get("haystack_sessions", [])
        if not isinstance(sessions, list) or len(sessions) < 2:
            continue
        # 要求第一段含旧答案候选，第二段含新答案候选（通过 has_answer 标记）
        old_has = _entry_has_answer(sessions[0])
        new_has = _entry_has_answer(sessions[1])
        if not (old_has or new_has):
            continue
        # 必须存在标准答案字段供新答案匹配
        if "answer" not in item:
            continue
        ku_items.append(item)
        if limit is not None and len(ku_items) >= limit:
            break
    return ku_items


def classify_response(resp: str, new_answer: str, old_candidates: List[str], detector: RefusalDetector) -> str:
    # 先拒答检测
    is_refusal, _ = detector.detect_refusal(resp or "")
    if is_refusal:
        return "refusal"

    r_norm = _normalize_text(resp)
    new_norm = _normalize_text(new_answer)

    new_hit = bool(new_norm and new_norm in r_norm)
    old_hit = False
    for oc in old_candidates:
        if not oc:
            continue
        ocn = _normalize_text(oc)
        if ocn and ocn in r_norm:
            old_hit = True
            break

    # 若新旧同时命中，归为 other，由人工复核
    if new_hit and old_hit:
        return "other"
    if new_hit:
        return "new"
    if old_hit:
        return "old"
    # 都未命中
    return "other"


def run_rq4_experiment(config: RQ4Config) -> Dict[str, Any]:
    _safe_mkdir(config.output_dir)

    # 加载数据
    with open(config.longmemeval_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 抽取 KU 子集（可回答+_abs混入，后续拆分）
    limit = 15 if config.quick_test else None
    ku_items = extract_ku_samples(dataset, limit=limit)

    # 标记 ABS（应拒答）
    for it in ku_items:
        qid = str(it.get("question_id", ""))
        it["is_abs"] = ("_abs" in qid)

    # 准备模型
    base_model, base_tok = load_model(config.base_model_name)
    rlhf_model, rlhf_tok = load_model(config.rlhf_model_name)
    detector = RefusalDetector()

    base_results: List[Dict[str, Any]] = []
    rlhf_results: List[Dict[str, Any]] = []

    def infer_one(model, tok, item) -> Dict[str, Any]:
        prompt = construct_prompt(item["haystack_sessions"], item.get("question", ""), config.max_sessions)
        out = generate(model, tok, prompt, config.temperature, config.top_p, config.max_new_tokens)
        return {
            "question_id": item.get("question_id"),
            "is_abs": item.get("is_abs", False),
            "response": out
        }

    # 逐条推理（并发=1，避免显存溢出）
    for item in ku_items:
        base_results.append(infer_one(base_model, base_tok, item))
        rlhf_results.append(infer_one(rlhf_model, rlhf_tok, item))

    # 统计：需准备新旧答案候选
    uc_base = 0
    uc_rlhf = 0
    sar_base = 0
    sar_rlhf = 0
    orr_base = 0
    orr_rlhf = 0

    # old candidates: 从 sessions[0] 的 content 中简单抽取——此处采用简化策略：
    # 使用 item["answer"] 作为新答案，旧答案候选从第一段中寻找高置信 has_answer 的消息内容集合。
    # 若无结构化旧答案，统计时仅以是否命中新答案/拒答作为主口径，old 为保守匹配。
    annotated: List[Dict[str, Any]] = []

    for item, b, r in zip(ku_items, base_results, rlhf_results):
        sessions = item.get("haystack_sessions", [])
        question = item.get("question", "")
        new_answer = item.get("answer", "")
        old_candidates: List[str] = []
        if sessions and len(sessions) >= 1:
            # 收集所有 has_answer=true 的内容作为旧候选（除了最后一个）
            all_has_answer_contents = []
            def collect_all_has_answer(entry: Any, acc: List[str]):
                if isinstance(entry, dict):
                    if entry.get("has_answer", False):
                        acc.append(entry.get("content", ""))
                elif isinstance(entry, list):
                    for e in entry:
                        collect_all_has_answer(e, acc)
            
            # 收集所有会话中的 has_answer 内容
            for session in sessions:
                collect_all_has_answer(session, all_has_answer_contents)
            
            # 排除最后一个（应该是新答案）
            if len(all_has_answer_contents) > 1:
                old_candidates.extend(all_has_answer_contents[:-1])
            elif len(all_has_answer_contents) == 1:
                # 如果只有一个，说明没有旧答案，保持空列表
                pass
        # 分类
        base_cls = classify_response(b["response"], new_answer, old_candidates, detector)
        rlhf_cls = classify_response(r["response"], new_answer, old_candidates, detector)

        # 累加 UC/SAR（仅在应答型样本，非 _abs）
        if not item.get("is_abs", False):
            if base_cls == "new":
                uc_base += 1
            if rlhf_cls == "new":
                uc_rlhf += 1
            if base_cls == "old":
                sar_base += 1
            if rlhf_cls == "old":
                sar_rlhf += 1
            if base_cls == "refusal":
                orr_base += 1
            if rlhf_cls == "refusal":
                orr_rlhf += 1

        annotated.append({
            "question_id": item.get("question_id"),
            "is_abs": item.get("is_abs", False),
            "question": question,
            "new_answer": new_answer,
            "old_candidates": old_candidates,
            "base": {"class": base_cls, "response": b["response"]},
            "rlhf": {"class": rlhf_cls, "response": r["response"]},
        })

    # 分母（应答型 KU）
    denom_ku = sum(1 for it in ku_items if not it.get("is_abs", False))

    # 比例
    uc_base_rate = (uc_base / denom_ku) if denom_ku else 0.0
    uc_rlhf_rate = (uc_rlhf / denom_ku) if denom_ku else 0.0
    sar_base_rate = (sar_base / denom_ku) if denom_ku else 0.0
    sar_rlhf_rate = (sar_rlhf / denom_ku) if denom_ku else 0.0
    orr_base_rate = (orr_base / denom_ku) if denom_ku else 0.0
    orr_rlhf_rate = (orr_rlhf / denom_ku) if denom_ku else 0.0

    # McNemar：UC 与 SAR 的配对二元结果
    # b: Base=1, RLHF=0；c: Base=0, RLHF=1（以“使用最新事实”为例）
    b_uc = 0
    c_uc = 0
    b_sar = 0
    c_sar = 0
    for a in annotated:
        if a["is_abs"]:
            continue
        base_is_new = (a["base"]["class"] == "new")
        rlhf_is_new = (a["rlhf"]["class"] == "new")
        if base_is_new and not rlhf_is_new:
            b_uc += 1
        if (not base_is_new) and rlhf_is_new:
            c_uc += 1
        base_is_old = (a["base"]["class"] == "old")
        rlhf_is_old = (a["rlhf"]["class"] == "old")
        if base_is_old and not rlhf_is_old:
            b_sar += 1
        if (not base_is_old) and rlhf_is_old:
            c_sar += 1

    m_uc = mcnemar_with_correction(b_uc, c_uc)
    m_sar = mcnemar_with_correction(b_sar, c_sar)

    # Wilson for ORR
    orr_base_w = wilson_ci(orr_base, denom_ku)
    orr_rlhf_w = wilson_ci(orr_rlhf, denom_ku)

    ts = _current_ts()

    # 构造与 RQ2 一致的结果目录命名：results/rq4_<pair_name>
    def _infer_pair_name(base_path: str, rlhf_path: str) -> str:
        lp = f"{base_path}".lower()
        if "qwen2.5-3b" in lp or "qwen2___5-3b" in lp or "qwen2.5_3b" in lp:
            return "qwen2.5_3b"
        if "llama" in lp and "3" in lp:
            return "llama3"
        if "mistral" in lp and "7b" in lp:
            return "mistral-7b"
        # fallback: 使用base目录名
        try:
            return Path(base_path).name.replace(" ", "_")
        except Exception:
            return "custom"

    pair_name = _infer_pair_name(config.base_model_name, config.rlhf_model_name)
    out_dir_name = f"rq4_{pair_name}"
    if config.quick_test:
        out_dir_name = f"{out_dir_name}_quick_test"
    # 保存到指定根目录（与RQ2一致的分组目录），例如: results/rq4_knowledge_update/rq4_qwen2.5_3b[_quick_test]
    out_dir = os.path.join(config.output_dir, out_dir_name)
    _safe_mkdir(out_dir)

    analysis = {
        "_readme": {
            "description": "RQ4实验结果分析 - 知识更新任务下的更新一致性与拒答",
            "experiment_type": "Base vs RLHF 在KU任务",
        },
        "experiment_info": {
            "ku_total": len(ku_items),
            "ku_answerable": denom_ku,
            "base_model": config.base_model_name,
            "rlhf_model": config.rlhf_model_name,
        },
        "ku_analysis": {
            "uc": {
                "base": uc_base_rate,
                "rlhf": uc_rlhf_rate,
                "diff": uc_rlhf_rate - uc_base_rate,
                "mcnemar": {"b": b_uc, "c": c_uc, **m_uc},
            },
            "sar": {
                "base": sar_base_rate,
                "rlhf": sar_rlhf_rate,
                "diff": sar_rlhf_rate - sar_base_rate,
                "mcnemar": {"b": b_sar, "c": c_sar, **m_sar},
            },
            "orr": {
                "base": {"rate": orr_base_rate, "wilson": {"p": orr_base_w[0], "lo": orr_base_w[1], "hi": orr_base_w[2]}},
                "rlhf": {"rate": orr_rlhf_rate, "wilson": {"p": orr_rlhf_w[0], "lo": orr_rlhf_w[1], "hi": orr_rlhf_w[2]}},
                "diff": orr_rlhf_rate - orr_base_rate,
            },
        },
    }

    # 保存分析（精简版）
    analysis_path = os.path.join(out_dir, f"rq4_analysis_{ts}.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # 生成带注释的分析（与RQ2风格一致：在主要字段下加入 _notes 解释）
    annotated_analysis = json.loads(json.dumps(analysis))  # 深拷贝
    annotated_analysis.setdefault("_readme", {}).setdefault("field_explanations", {
        "experiment_info": "实验基本信息（样本规模、模型路径）",
        "ku_analysis": "KU 指标分析（UC/SAR/ORR 与统计）",
    })
    annotated_analysis.setdefault("experiment_info", {}).setdefault("_notes", {
        "ku_total": "KU 子集样本总数（含 _abs）",
        "ku_answerable": "应答型 KU 分母（排除 _abs）",
        "base_model": "基础模型路径",
        "rlhf_model": "RLHF 模型路径",
    })
    ka = annotated_analysis.setdefault("ku_analysis", {})
    ka.setdefault("uc", {}).setdefault("_notes", {
        "base": "Base 使用最新事实的比例",
        "rlhf": "RLHF 使用最新事实的比例",
        "diff": "差值（正值表示 RLHF 更一致）",
        "mcnemar": "配对检验（b/c 与 p 值）",
    })
    ka.setdefault("sar", {}).setdefault("_notes", {
        "base": "Base 使用旧事实的比例",
        "rlhf": "RLHF 使用旧事实的比例",
        "diff": "差值（正值表示 RLHF 更容易答旧）",
        "mcnemar": "配对检验（b/c 与 p 值）",
    })
    ka.setdefault("orr", {}).setdefault("_notes", {
        "base": "Base 拒答比例（KU 应答分母）",
        "rlhf": "RLHF 拒答比例（KU 应答分母）",
        "diff": "差值（正值表示 RLHF 拒答更多）",
    })
    with open(os.path.join(out_dir, f"rq4_analysis_{ts}_annotated.json"), "w", encoding="utf-8") as f:
        json.dump(annotated_analysis, f, ensure_ascii=False, indent=2)

    # 拆分并保存 base/rlhf 响应文件（与 RQ2 一致的风格）
    if config.save_responses:
        base_only = []
        rlhf_only = []
        for a in annotated:
            base_only.append({
                "question_id": a["question_id"],
                "is_abs": a["is_abs"],
                "question": a["question"],
                "new_answer": a["new_answer"],
                "old_candidates": a["old_candidates"],
                "response": a["base"]["response"],
                "class": a["base"]["class"],
            })
            rlhf_only.append({
                "question_id": a["question_id"],
                "is_abs": a["is_abs"],
                "question": a["question"],
                "new_answer": a["new_answer"],
                "old_candidates": a["old_candidates"],
                "response": a["rlhf"]["response"],
                "class": a["rlhf"]["class"],
            })

        with open(os.path.join(out_dir, f"rq4_base_ku_responses_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(base_only, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"rq4_rlhf_ku_responses_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(rlhf_only, f, ensure_ascii=False, indent=2)
        # 保存逐条实例明细（供人工复核，不与 analysis_annotated 混淆）
        with open(os.path.join(out_dir, f"rq4_ku_instances_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)

    print(f"✅ RQ4 实验完成，输出目录: {out_dir}")
    return analysis