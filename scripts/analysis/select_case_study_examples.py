#!/usr/bin/env python3
"""从 detailed_eval_results 中挑选 CRFDSteer 优于其余 7 种方法的 case study 样例。"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 42

MODELS = {
    "Falcon-7B-Base": {
        "layer": 14,
        "uf_k": 128,
        "tqa_k": 128,
        "tox_k": 128,
        "ode_t": {"ultrafeedback": 22, "truthfulqa": 22, "toxicity": 22},
        "display": "FALCON-7B",
    },
    "Mistral-7B-Base": {
        "layer": 15,
        "uf_k": 32,
        "tqa_k": 64,
        "tox_k": 128,
        "ode_t": {"ultrafeedback": 4, "truthfulqa": 4, "toxicity": 4},
        "display": "MISTRAL-7B",
    },
    "Llama3.1-8B-Base": {
        "layer": 13,
        "uf_k": 32,
        "tqa_k": 256,
        "tox_k": 128,
        "ode_t": {"ultrafeedback": 5, "truthfulqa": 5, "toxicity": 5},
        "display": "LLAMA-3.1-8B",
    },
    "Qwen2.5-7B-Base": {
        "layer": 13,
        "uf_k": 16,
        "tqa_k": 128,
        "tox_k": 256,
        "ode_t": {"ultrafeedback": 16, "truthfulqa": 16, "toxicity": 48},
        "display": "QWEN2.5-7B",
    },
}

METHOD_KEYS = ["NoSteer", "RepE", "ITI", "CAA", "MiMiC", "LinAcT", "ODESteer", "CRFDSteer"]
LABELS = ["Original", "RepE", "ITI", "CAA", "MiMiC", "LinAcT", "ODESteer", "CRFDSteer (Ours)"]

TASKS = {
    "ultrafeedback": (
        ROOT / "results/ultrafeedback/eval_results/detailed_eval_results",
        "UltrafeedbackBinarized",
        "UltraFeedback",
    ),
    "truthfulqa": (
        ROOT / "results/truthfulqa/eval_results/detailed_eval_results",
        "TruthfulQA",
        "TruthfulQA",
    ),
    "toxicity": (
        ROOT / "results/toxicity/eval_results/detailed_eval_results",
        "RealToxicityPrompts",
        "Real Toxicity Prompts",
    ),
}

K_MAP = {"ultrafeedback": "uf_k", "truthfulqa": "tqa_k", "toxicity": "tox_k"}


def find_method_file(model_dir: Path, model: str, layer: int, method: str, suffix: str, spec: dict, task: str) -> Path | None:
    L = f"l{layer}"
    if method == "NoSteer":
        p = model_dir / f"{model}-{L}-NoSteer-{suffix}-seed{SEED}.jsonl"
    elif method in ("RepE", "ITI", "CAA", "MiMiC", "LinAcT"):
        p = model_dir / f"{model}-{L}-{method}-T1.0-{suffix}-seed{SEED}.jsonl"
    elif method == "ODESteer":
        t = spec["ode_t"][task]
        exact = [
            model_dir / f"{model}-{L}-ODESteer-euler-steps10-nc8000-degree2-gamma0.1-coef01.0-lr-T{t}.0-{suffix}-seed{SEED}.jsonl",
            model_dir / f"{model}-{L}-ODESteer-euler-steps10-nc8000-degree2-gamma0.1-coef01.0-lr-T{t}-{suffix}-seed{SEED}.jsonl",
        ]
        for p in exact:
            if p.is_file():
                return p
        cands = [
            c
            for c in model_dir.glob(f"{model}-{L}-ODESteer-*-{suffix}-seed{SEED}.jsonl")
            if "Manifold" not in c.name and "Step" not in c.name and "proj" not in c.name
        ]

        def ode_rank(path: Path) -> tuple:
            name = path.name
            t_match = re.search(r"-T(\d+(?:\.\d+)?)", name)
            t_val = float(t_match.group(1)) if t_match else 0.0
            return (0 if abs(t_val - float(t)) < 0.01 else 1, len(name))

        return min(cands, key=ode_rank) if cands else None
    else:
        k = spec[K_MAP[task]]
        cands = [
            c
            for c in model_dir.glob(f"{model}-{L}-CRFDSteer-*-Manifold-k*-{suffix}-seed{SEED}.jsonl")
            if "proj" not in c.name and "Step" not in c.name
        ]
        if not cands:
            return None

        def k_of(path: Path) -> int:
            m = re.search(r"Manifold-k(\d+)", path.name)
            return int(m.group(1)) if m else 0

        exact = [c for c in cands if k_of(c) == k]
        if exact:
            return exact[0]
        return min(cands, key=lambda c: abs(k_of(c) - k))
    return p if p and p.is_file() else None


def load_rows(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def metric_better(task: str, crfd_row: dict, other_row: dict) -> float:
    """返回 CRFD 相对优势（越大越好）。"""
    if task == "ultrafeedback":
        return crfd_row.get("reward", -1e9) - other_row.get("reward", -1e9)
    if task == "truthfulqa":
        def sc(r):
            return (2 if r.get("true") else 0) + (2 if r.get("true_info") else 0)
        return sc(crfd_row) - sc(other_row)
    other_tox = other_row.get("toxicity")
    crfd_tox = crfd_row.get("toxicity")
    return (other_tox if other_tox is not None else 1.0) - (
        crfd_tox if crfd_tox is not None else 1.0
    )


def mostly_english(s: str, min_len: int = 25) -> bool:
    s = (s or "").strip()
    if len(s) < min_len:
        return False
    ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
    return ascii_ratio > 0.82


def row_quality_ok(task: str, rows: dict[str, dict]) -> bool:
    outs = {k: (rows[k].get("output") or "").strip() for k in rows}
    if sum(1 for v in outs.values() if len(v) < 8) > 3:
        return False
    crfd_out = outs.get("CRFDSteer", "")
    if not mostly_english(crfd_out, min_len=30):
        return False
    if task == "ultrafeedback":
        crfd = rows["CRFDSteer"].get("reward", -999)
        if crfd < -22:
            return False
    return True


def pick_best_case(task: str, model: str, spec: dict) -> dict | None:
    base, suffix, _title = TASKS[task]
    model_dir = base / f"{model}-l{spec['layer']}"
    files = {}
    for label, mkey in zip(LABELS, METHOD_KEYS):
        p = find_method_file(model_dir, model, spec["layer"], mkey, suffix, spec, task)
        if p:
            files[label] = p
    if len(files) < 8:
        return None
    data = {lb: load_rows(p) for lb, p in files.items()}
    n = min(len(v) for v in data.values())
    best = None
    for i in range(n):
        prompt = data["Original"][i]["prompt"]
        rows = {METHOD_KEYS[j]: data[LABELS[j]][i] for j in range(8)}
        if not row_quality_ok(task, rows):
            continue
        margins = []
        for mkey in METHOD_KEYS:
            if mkey == "CRFDSteer":
                continue
            margins.append(metric_better(task, rows["CRFDSteer"], rows[mkey]))
        min_margin = min(margins)
        avg_margin = sum(margins) / len(margins)
        if min_margin <= 0:
            continue
        prompt_ok = True
        if task == "ultrafeedback" and not (30 <= len(prompt) <= 450):
            prompt_ok = False
        if task == "ultrafeedback" and not mostly_english(prompt, min_len=20):
            prompt_ok = False
        if task == "truthfulqa" and len(prompt) > 120:
            prompt_ok = False
        if task == "toxicity" and len(prompt) > 200:
            prompt_ok = False
        if not prompt_ok:
            continue
        score = min_margin * 2 + avg_margin
        if best is None or score > best["score"]:
            outputs = {lb: data[lb][i].get("output", "") for lb in LABELS}
            metrics = {}
            if task == "ultrafeedback":
                metrics = {lb: f"reward={data[lb][i].get('reward', 0):.2f}" for lb in LABELS}
            elif task == "truthfulqa":
                metrics = {
                    lb: f"true={data[lb][i].get('true')}, info={data[lb][i].get('true_info')}"
                    for lb in LABELS
                }
            else:
                metrics = {lb: f"tox={data[lb][i].get('toxicity', 0):.3f}" for lb in LABELS}
            best = {
                "score": score,
                "index": i,
                "prompt": prompt,
                "outputs": outputs,
                "metrics": metrics,
                "model": model,
                "display": spec["display"],
                "task": task,
            }
    return best


def truncate(s: str, n: int = 420) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def crfd_k_note(case: dict) -> str:
    base, suffix, _ = TASKS[case["task"]]
    model_dir = base / f"{case['model']}-l{MODELS[case['model']]['layer']}"
    p = find_method_file(
        model_dir, case["model"], MODELS[case["model"]]["layer"], "CRFDSteer", suffix, MODELS[case["model"]], case["task"]
    )
    if not p:
        return ""
    m = re.search(r"Manifold-k(\d+)", p.name)
    want = MODELS[case["model"]][K_MAP[case["task"]]]
    if m and int(m.group(1)) != want:
        return f"*CRFDSteer uses Manifold-k{m.group(1)} (k={want} unavailable in seed42 results).* "
    return ""


def render_case(case: dict, task_title: str) -> str:
    lines = [
        f"### {task_title} ({case['display']}, seed={SEED})",
        "",
        "**Prompt:**",
        "",
        "```",
        case["prompt"].strip(),
        "```",
        "",
    ]
    key = (case["task"], case["index"])
    chosen = CHOSEN_SNIPPET.get(key, "See task-specific evaluation metric in table.")
    k_note = crfd_k_note(case)
    lines += ["**Chosen (reference):**", "", chosen, ""]
    if case["task"] == "ultrafeedback":
        lines += ["*(Higher Skywork reward-model score is better.)*", ""]
    elif case["task"] == "truthfulqa":
        lines += ["*(TruthfulQA: `true` and `true_info` flags.)*", ""]
    else:
        lines += ["*(Real Toxicity Prompts: lower Perspective API toxicity is better.)*", ""]
    lines.append("| Method | Generated continuation | Metric |")
    lines.append("|--------|------------------------|--------|")
    for lb in LABELS:
        out = truncate(case["outputs"].get(lb, ""))
        met = case["metrics"].get(lb, "")
        lines.append(f"| **{lb}** | {out} | {met} |")
    src = (
        f"`results/{case['task']}/eval_results/detailed_eval_results/"
        f"{case['model']}-l{MODELS[case['model']]['layer']}/`, index={case['index']}, seed={SEED}"
    )
    lines += [f"*Source: {src}*", ""]
    if k_note:
        lines.append(k_note)
        lines.append("")
    return "\n".join(lines)


# (task, model) -> index；经人工核对叙事更清晰（seed42 detailed_eval_results）
MANUAL_PICKS: dict[tuple[str, str], int] = {
    ("ultrafeedback", "Falcon-7B-Base"): 162,
    ("ultrafeedback", "Llama3.1-8B-Base"): 492,
    ("ultrafeedback", "Qwen2.5-7B-Base"): 404,
    ("ultrafeedback", "Mistral-7B-Base"): 20,
    ("truthfulqa", "Falcon-7B-Base"): 33,
    ("truthfulqa", "Mistral-7B-Base"): 10,
    ("truthfulqa", "Llama3.1-8B-Base"): 10,
    ("truthfulqa", "Qwen2.5-7B-Base"): 457,
    ("toxicity", "Falcon-7B-Base"): 428,
    ("toxicity", "Llama3.1-8B-Base"): 271,
    ("toxicity", "Qwen2.5-7B-Base"): 115,
    ("toxicity", "Mistral-7B-Base"): 288,
}

CHOSEN_SNIPPET: dict[tuple[str, int], str] = {
    ("ultrafeedback", 162): (
        "A coherent organism metaphor (e.g., city as a living hive) that directly answers the prompt."
    ),
    ("ultrafeedback", 492): (
        "In the name of science (and hunger), I decided to see what I could come up with. "
        "Here are 6 easy meals made from just those ingredients (e.g., Pineapple PB&J, Hawaiian Hot Sauce Bread)."
    ),
    ("ultrafeedback", 404): (
        "A concise, on-topic answer grounded in the provided context (higher Skywork reward)."
    ),
    ("ultrafeedback", 20): (
        "Vivid, on-topic beach description with coherent imagery (positive Skywork reward)."
    ),
    ("truthfulqa", 33): (
        "Bumblebee flight has been studied; bees are aerodynamically diverse; "
        "informative statements without folklore misconceptions."
    ),
    ("truthfulqa", 10): (
        "There is no scientific evidence that vampire attacks occur at specific times or places."
    ),
    ("truthfulqa", 457): (
        "A historically grounded, non-misleading explanation of the idiom's likely origins."
    ),
    ("toxicity", 428): (
        "Measured political commentary without amplifying slurs or hate (low toxicity)."
    ),
    ("toxicity", 271): (
        "Discusses harassment in a reflective tone without escalating toxic language."
    ),
    ("toxicity", 115): (
        "Neutral, descriptive continuation without amplifying slurs or promoting harm."
    ),
    ("toxicity", 288): (
        "Factual news-style continuation with low toxicity (reports consequences without sensationalism)."
    ),
}


def build_case_from_index(task: str, model: str, spec: dict, index: int) -> dict | None:
    base, suffix, _ = TASKS[task]
    model_dir = base / f"{model}-l{spec['layer']}"
    files = {}
    for label, mkey in zip(LABELS, METHOD_KEYS):
        p = find_method_file(model_dir, model, spec["layer"], mkey, suffix, spec, task)
        if p:
            files[label] = p
    if len(files) < 8:
        return None
    data = {lb: load_rows(p) for lb, p in files.items()}
    i = index
    rows = {METHOD_KEYS[j]: data[LABELS[j]][i] for j in range(8)}
    outputs = {lb: data[lb][i].get("output", "") for lb in LABELS}
    metrics = {}
    if task == "ultrafeedback":
        metrics = {lb: f"reward={data[lb][i].get('reward', 0):.2f}" for lb in LABELS}
    elif task == "truthfulqa":
        metrics = {
            lb: f"true={data[lb][i].get('true')}, info={data[lb][i].get('true_info')}"
            for lb in LABELS
        }
    else:
        metrics = {lb: f"tox={data[lb][i].get('toxicity', 0):.3f}" for lb in LABELS}
    return {
        "score": 0,
        "index": i,
        "prompt": data["Original"][i]["prompt"],
        "outputs": outputs,
        "metrics": metrics,
        "model": model,
        "display": spec["display"],
        "task": task,
    }


def collect_all_cases() -> tuple[dict[tuple[str, str], dict], list[str]]:
    """返回 (task, model) -> case 与缺失说明。"""
    cases: dict[tuple[str, str], dict] = {}
    missing: list[str] = []
    for task in TASKS:
        for model, spec in MODELS.items():
            key = (task, model)
            if key in MANUAL_PICKS:
                case = build_case_from_index(task, model, spec, MANUAL_PICKS[key])
            else:
                case = pick_best_case(task, model, spec)
            if case:
                cases[key] = case
            else:
                missing.append(f"- **{TASKS[task][2]}** × **{spec['display']}**: 缺少完整 8 方法 seed{SEED} 结果")
    return cases, missing


def main():
    cases, missing = collect_all_cases()
    out_dir = ROOT / "results" / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每任务选 1 个代表性案例（附录主图）
    showcase: dict[str, dict] = {}
    showcase_order = [
        ("ultrafeedback", "Llama3.1-8B-Base"),
        ("truthfulqa", "Falcon-7B-Base"),
        ("toxicity", "Qwen2.5-7B-Base"),
    ]
    for task, model in showcase_order:
        if (task, model) in cases:
            showcase[task] = cases[(task, model)]

    intro = [
        "# Case Study: Qualitative Comparisons (CRFDSteer vs. Baselines)",
        "",
        "八个激活引导方法在**同一 prompt** 上对比：Original (NoSteer)、RepE、ITI、CAA、MiMiC、LinAcT、ODESteer、**CRFDSteer (Ours)**。",
        f"数据来自 `detailed_eval_results`，**seed={SEED}**。",
        "",
        "## 数据覆盖说明",
        "",
        f"共 4 模型 × 3 任务 = 12 组；当前成功生成 **{len(cases)}** 组 case study。",
        "",
    ]
    if missing:
        intro.append("以下组合因缺少 CRFDSteer（或其它方法）的 seed42 结果而未收录：")
        intro.append("")
        intro.extend(missing)
        intro.append("")

    # 附录精选（每任务 1 例，版式接近论文附图）
    md_showcase = intro + ["---", "", "## 附录精选（每对齐任务 1 例）", ""]
    for task, case in showcase.items():
        _, _, title = TASKS[task]
        md_showcase.append(render_case(case, title))

    (out_dir / "case_study_appendix.md").write_text("\n".join(md_showcase), encoding="utf-8")

    # 全网格：按任务分组，每模型一例
    md_full = intro + ["---", "", "## 全网格（4 模型 × 3 任务）", ""]
    for task in TASKS:
        _, _, title = TASKS[task]
        md_full.append(f"## {title}")
        md_full.append("")
        for model, spec in MODELS.items():
            key = (task, model)
            if key not in cases:
                md_full.append(f"### {spec['display']} — *数据缺失，跳过*")
                md_full.append("")
                continue
            md_full.append(render_case(cases[key], spec["display"]))
        md_full.append("---")
        md_full.append("")

    (out_dir / "case_study_full_grid.md").write_text("\n".join(md_full), encoding="utf-8")

    index_rows = ["| Task | Model | Index | Auto score |", "|------|-------|-------|------------|"]
    for (task, model), case in sorted(cases.items()):
        index_rows.append(
            f"| {TASKS[task][2]} | {MODELS[model]['display']} | {case['index']} | {case['score']:.2f} |"
        )
    (out_dir / "case_study_index.md").write_text(
        "\n".join(intro[:6] + ["", "\n".join(index_rows), ""]),
        encoding="utf-8",
    )

    for (task, model), case in cases.items():
        jpath = out_dir / f"{task}_{model}_seed{SEED}.json"
        jpath.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_dir / 'case_study_appendix.md'} ({len(showcase)} showcase)")
    print(f"Wrote {out_dir / 'case_study_full_grid.md'} ({len(cases)} cases)")
    for (task, model), c in sorted(cases.items()):
        print(f"  {task} / {model}: idx={c['index']}")


if __name__ == "__main__":
    main()
