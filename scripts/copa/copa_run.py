#!/usr/bin/env python3
"""
COPA causal commonsense MC evaluation for Llama3.1-8B-Base.

Runs 8 methods (Original / RepE / ITI / CAA / MiMiC / Linear-AcT / ODESteer / GSCSteer)
with zero-tuning transfer of LLaMA TruthfulQA parameters. Test set never used for fitting.

Usage (repo root):
  uv run python -u scripts/copa/copa_smoke_test.py
  uv run python -u scripts/copa/copa_run.py mode=smoke
  uv run python -u scripts/copa/copa_run.py mode=full
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from tqdm import trange

from odesteer.lm import HuggingFaceLM
from odesteer.lm._config import _FULL_LLM_NAMES
from odesteer.steer import get_steer_model
from odesteer.utils import get_project_dir
from odesteer.utils.copa import (
    COPASample,
    DATASET_SOURCE,
    DATASET_VERSION,
    SEED,
    build_contrastive_metadata,
    compute_split_statistics,
    format_copa_prompt,
    load_copa_official,
    save_json,
    samples_to_records,
    split_dev_for_validation,
)


# ---------------------------------------------------------------------------
# Resolved LLaMA3.1-8B parameters (TruthfulQA zero-tuning transfer)
# Source of truth: scripts/repro/run_multi_seed_grid.py + confs/truthfulqa.yaml
# ---------------------------------------------------------------------------
MODEL_NAME = "Llama3.1-8B-Base"
LAYER_IDX = 13  # 0-indexed; paper Table often reports 14 (1-indexed)
BASELINE_T = 1.0
ODE_T = 5.0
MANIFOLD_K_TRUTHFULQA = 256
ODE_STEPS = 10
ODE_SOLVER = "euler"
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
DTYPE = torch.float32

# Display order required by the COPA brief
METHOD_DISPLAY_ORDER = [
    "Original",
    "RepE",
    "ITI",
    "CAA",
    "MiMiC",
    "Linear-AcT",
    "ODESteer",
    "GSCSteer",
]


@dataclass(frozen=True)
class MethodSpec:
    display_name: str
    steer_type: str  # NoSteer | RepE | ... | CRFDSteer
    T: float
    use_manifold: bool = False
    use_manifold_retraction: bool = False
    manifold_dim: int = MANIFOLD_K_TRUTHFULQA


def default_method_specs(vk_path: Path | None) -> list[MethodSpec]:
    """Eight configs; GSCSteer = CRFDSteer + OTSP projection + RRC retraction."""
    return [
        MethodSpec("Original", "NoSteer", T=BASELINE_T),
        MethodSpec("RepE", "RepE", T=BASELINE_T),
        MethodSpec("ITI", "ITI", T=BASELINE_T),
        MethodSpec("CAA", "CAA", T=BASELINE_T),
        MethodSpec("MiMiC", "MiMiC", T=BASELINE_T),
        MethodSpec("Linear-AcT", "LinAcT", T=BASELINE_T),
        MethodSpec("ODESteer", "ODESteer", T=ODE_T),
        MethodSpec(
            "GSCSteer",
            "CRFDSteer",
            T=ODE_T,
            use_manifold=True,
            use_manifold_retraction=True,
            manifold_dim=MANIFOLD_K_TRUTHFULQA,
        ),
    ]


def resolve_vk_path(project_dir: Path, layer_idx: int, k: int) -> Path:
    """Reuse TruthfulQA V_k (zero-tuning); do not fit subspace on COPA test."""
    candidates = [
        project_dir
        / "data"
        / "truthfulqa"
        / "activations"
        / MODEL_NAME
        / f"manifold_Vk_layer{layer_idx}_k{k}.pt",
        project_dir
        / "data"
        / "truthfulqa"
        / "activations"
        / MODEL_NAME
        / f"manifold_Vk_layer{layer_idx}.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "GSCSteer requires a precomputed V_k. Looked for:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nRun data/extract_manifold.py for TruthfulQA Llama layer "
        f"{layer_idx} with -k {k}."
    )


def load_steer_kwargs_from_yaml(steer_type: str) -> dict[str, Any]:
    cfg_path = get_project_dir() / "confs" / "steer" / f"{steer_type}.yaml"
    if not cfg_path.is_file():
        # NoSteer / fallbacks
        if steer_type == "NoSteer":
            return {}
        raise FileNotFoundError(cfg_path)
    raw = OmegaConf.load(cfg_path)
    kw = OmegaConf.to_container(raw.get("kwargs", {}), resolve=True) or {}
    return dict(kw)


def build_steer_model_kwargs(spec: MethodSpec, vk_path: Path | None) -> dict[str, Any]:
    kw = load_steer_kwargs_from_yaml(spec.steer_type)
    # Force COPA zero-tuning flags without inventing new hyperparameters
    if spec.steer_type in ("ODESteer", "CRFDSteer"):
        kw["solver"] = ODE_SOLVER
        kw["steps"] = ODE_STEPS
    kw["use_manifold"] = bool(spec.use_manifold)
    kw["use_manifold_retraction"] = bool(spec.use_manifold_retraction)
    kw["manifold_dim"] = int(spec.manifold_dim)
    if spec.use_manifold:
        if vk_path is None:
            raise ValueError("vk_path required for GSCSteer / manifold methods")
        kw["vk_path"] = str(vk_path)
    else:
        kw["vk_path"] = None
    return kw


@torch.no_grad()
def score_answer_loglikelihood(
    lm: HuggingFaceLM,
    prompt: str,
    answer: str,
    *,
    steer: bool,
    T: float,
) -> tuple[float, float]:
    """
    Conditional log-likelihood of ``answer`` tokens given ``prompt``.

    Prompt and answer are tokenized separately then concatenated so that Llama
    retokenization cannot merge the answer letter into the ``Answer:`` boundary.
    Uses existing steer_prob_hook on the answer span. Returns (sum_logprob, mean_logprob).
    """
    prompt_ids = lm.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    answer_ids = lm.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    if answer_ids.numel() == 0:
        raise ValueError(f"Answer {answer!r} produced zero tokens")
    question_len = prompt_ids.shape[1]
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(lm.model.device)
    attention_mask = torch.ones_like(input_ids)
    steer_kwargs = dict(T=T)
    if steer and lm.steer_model is not None:
        lm.register_steer_prob_hook(question_len - 1, steer_kwargs)
        try:
            outputs = lm.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            lm.remove_steer_prob_hook()
    else:
        outputs = lm.model(input_ids=input_ids, attention_mask=attention_mask)
    ans = input_ids[0, question_len:]
    answer_logits = outputs.logits[0, question_len - 1 : -1]
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_lp = log_probs.gather(-1, ans.unsqueeze(-1)).squeeze(-1)
    s = float(token_lp.sum().item())
    m = float(token_lp.mean().item())
    return s, m


def predict_ab(
    lm: HuggingFaceLM,
    prompt: str,
    *,
    steer: bool,
    T: float,
) -> dict[str, Any]:
    score_a, mean_a = score_answer_loglikelihood(lm, prompt, "A", steer=steer, T=T)
    score_b, mean_b = score_answer_loglikelihood(lm, prompt, "B", steer=steer, T=T)
    is_tie = score_a == score_b
    if score_a > score_b:
        pred = "A"
    elif score_b > score_a:
        pred = "B"
    else:
        pred = "A"  # fixed tie-break
    return {
        "score_A": score_a,
        "score_B": score_b,
        "mean_logprob_A": mean_a,
        "mean_logprob_B": mean_b,
        "predicted_letter": pred,
        "is_tie": bool(is_tie),
        "invalid_output": False,
    }


@torch.no_grad()
def extract_contrastive_activations(
    lm: HuggingFaceLM,
    samples: list[COPASample],
    layer_idx: int,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Last-token activations for prompt⊕correct vs prompt⊕incorrect.

    Prompt / answer are tokenized separately then concatenated (same rule as
    scoring) so Llama BPE cannot merge the answer letter into ``Answer:``.
    Batches use left-padding so the final token is always the answer letter.
    """
    from types import SimpleNamespace

    from odesteer.lm._huggingface_lm import _extract_and_set_hidden

    pos_pairs: list[tuple[str, str]] = []
    neg_pairs: list[tuple[str, str]] = []
    for s in samples:
        prompt = format_copa_prompt(s)
        pos_pairs.append((prompt, s.gold_letter))
        neg_pairs.append((prompt, s.incorrect_letter))

    pad_id = lm.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = lm.tokenizer.eos_token_id

    def _encode_pair(prompt: str, answer: str) -> torch.Tensor:
        prompt_ids = lm.tokenizer(prompt, add_special_tokens=True).input_ids
        answer_ids = lm.tokenizer(answer, add_special_tokens=False).input_ids
        if not answer_ids:
            raise ValueError(f"Answer {answer!r} produced zero tokens")
        return torch.tensor(prompt_ids + answer_ids, dtype=torch.long)

    def _batched(pairs: list[tuple[str, str]], desc: str) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        n_batches = (len(pairs) + batch_size - 1) // batch_size
        for bi in trange(n_batches, desc=desc):
            batch = pairs[bi * batch_size : (bi + 1) * batch_size]
            seqs = [_encode_pair(p, a) for p, a in batch]
            max_len = max(int(s.numel()) for s in seqs)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, seq in enumerate(seqs):
                L = int(seq.numel())
                input_ids[i, -L:] = seq
                attention_mask[i, -L:] = 1
            input_ids = input_ids.to(lm.model.device)
            attention_mask = attention_mask.to(lm.model.device)
            captured: list[torch.Tensor] = []

            def capture_hook(_module, _inp, out):
                hidden, _ = _extract_and_set_hidden(out)
                captured.append(hidden.detach())

            handle = lm._get_layer_module(layer_idx).register_forward_hook(capture_hook)
            try:
                lm._forward_for_activation_capture(
                    SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
                )
            finally:
                handle.remove()
            if not captured:
                raise RuntimeError("activation hook did not run")
            chunks.append(captured[0][:, -1, :].cpu())
        return torch.cat(chunks, dim=0)

    return _batched(pos_pairs, "extract_pos"), _batched(neg_pairs, "extract_neg")


def accuracy_and_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict[str, float]:
    n = len(correct)
    acc = float(correct.mean() * 100.0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(correct[idx].mean() * 100.0))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "accuracy": acc,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "correct": int(correct.sum()),
        "total": int(n),
    }


def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """Paired McNemar (exact binomial on discordant pairs)."""
    a = correct_a.astype(bool)
    b = correct_b.astype(bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n_disc = n01 + n10
    if n_disc == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(n01, n_disc, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "n_discordant": n_disc, "pvalue": p}


def subset_acc(correct: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(correct[mask].mean() * 100.0)


def run_one_method(
    spec: MethodSpec,
    cal_samples: list[COPASample],
    test_samples: list[COPASample],
    layer_idx: int,
    batch_size: int,
    vk_path: Path | None,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("=" * 80)
    print(f"→ Method {spec.display_name} (steer={spec.steer_type}, T={spec.T})")
    t0 = time.time()

    steer_kwargs = build_steer_model_kwargs(spec, vk_path)
    resolved = {
        "display_name": spec.display_name,
        "steer_type": spec.steer_type,
        "T": spec.T,
        "layer_idx": layer_idx,
        "steer_model_kwargs": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in steer_kwargs.items()
            if k != "V_k"
        },
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "dtype": str(DTYPE),
        "evaluation_mode": "forced_choice_loglikelihood",
        "parameter_source": "truthfulqa_llama3.1-8b_zero_tuning_transfer",
    }

    lm = HuggingFaceLM(
        MODEL_NAME,
        steer_name=None if spec.steer_type == "NoSteer" else spec.steer_type,
        steer_model_kwargs=steer_kwargs if spec.steer_type != "NoSteer" else {},
        steer_layer_idx=layer_idx,
        device="auto",
        dtype=DTYPE,
    )

    if lm.steer_model is not None:
        print(f"→ Fitting steer on {len(cal_samples)} calibration samples ...")
        pos_X, neg_X = extract_contrastive_activations(
            lm, cal_samples, layer_idx, batch_size=batch_size
        )
        lm.fit_steer_model(pos_X, neg_X)
        del pos_X, neg_X
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    steer_flag = lm.steer_model is not None
    for s in test_samples:
        prompt = format_copa_prompt(s)
        pred = predict_ab(lm, prompt, steer=steer_flag, T=spec.T)
        is_correct = pred["predicted_letter"] == s.gold_letter
        rows.append(
            {
                "idx": s.idx,
                "premise": s.premise,
                "question_type": s.question,
                "choice1": s.choice1,
                "choice2": s.choice2,
                "gold_label": s.label,
                "gold_letter": s.gold_letter,
                "method": spec.display_name,
                "score_A": pred["score_A"],
                "score_B": pred["score_B"],
                "mean_logprob_A": pred["mean_logprob_A"],
                "mean_logprob_B": pred["mean_logprob_B"],
                "predicted_letter": pred["predicted_letter"],
                "is_correct": bool(is_correct),
                "is_tie": pred["is_tie"],
                "invalid_output": False,
            }
        )

    del lm
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    correct = df["is_correct"].to_numpy(dtype=np.float64)
    cause_mask = (df["question_type"] == "cause").to_numpy()
    effect_mask = (df["question_type"] == "effect").to_numpy()
    c1_mask = (df["gold_label"] == 0).to_numpy()
    c2_mask = (df["gold_label"] == 1).to_numpy()
    acc_info = accuracy_and_ci(correct)
    metrics = {
        **acc_info,
        "cause_accuracy": subset_acc(correct, cause_mask),
        "effect_accuracy": subset_acc(correct, effect_mask),
        "choice1_accuracy": subset_acc(correct, c1_mask),
        "choice2_accuracy": subset_acc(correct, c2_mask),
        "invalid_output_rate": 0.0,
        "runtime_sec": time.time() - t0,
        "resolved_parameters": resolved,
    }
    pred_path = out_dir / f"predictions_{spec.display_name.replace('/', '_')}.csv"
    df.to_csv(pred_path, index=False)
    print(
        f"✓ {spec.display_name}: Acc={metrics['accuracy']:.2f}% "
        f"({metrics['correct']}/{metrics['total']}) "
        f"[{metrics['runtime_sec']:.1f}s]"
    )
    return df, metrics


def aggregate_results(
    all_preds: dict[str, pd.DataFrame],
    all_metrics: dict[str, dict[str, Any]],
    out_dir: Path,
    model_path: str,
    split_stats: dict[str, Any],
    resolved_config: dict[str, Any],
) -> None:
    original_correct = all_preds["Original"]["is_correct"].to_numpy(dtype=bool)
    orig_acc = all_metrics["Original"]["accuracy"]

    # Combined predictions
    pred_all = pd.concat([all_preds[m] for m in METHOD_DISPLAY_ORDER], ignore_index=True)
    pred_all.to_csv(out_dir / "copa_predictions.csv", index=False)

    table_rows = []
    methods_blob: dict[str, Any] = {}
    for name in METHOD_DISPLAY_ORDER:
        m = all_metrics[name]
        delta = m["accuracy"] - orig_acc
        table_rows.append(
            {
                "Method": name,
                "Correct/Total": f"{m['correct']}/{m['total']}",
                "Accuracy(%)": round(m["accuracy"], 2),
                "ΔAcc(pp)": round(delta, 2),
                "Cause Acc(%)": round(m["cause_accuracy"], 2),
                "Effect Acc(%)": round(m["effect_accuracy"], 2),
                "Choice-1 Acc(%)": round(m["choice1_accuracy"], 2),
                "Choice-2 Acc(%)": round(m["choice2_accuracy"], 2),
                "95% CI": f"[{m['ci95_low']:.2f}, {m['ci95_high']:.2f}]",
            }
        )
        methods_blob[name] = {
            "overall_accuracy": m["accuracy"],
            "accuracy_gain_vs_original": delta,
            "cause_accuracy": m["cause_accuracy"],
            "effect_accuracy": m["effect_accuracy"],
            "choice1_accuracy": m["choice1_accuracy"],
            "choice2_accuracy": m["choice2_accuracy"],
            "confidence_interval_95": [m["ci95_low"], m["ci95_high"]],
            "correct_total": f"{m['correct']}/{m['total']}",
            "invalid_output_rate": m["invalid_output_rate"],
            "runtime_sec": m["runtime_sec"],
            "resolved_parameters": m["resolved_parameters"],
        }

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(out_dir / "copa_results_table.csv", index=False)

    # McNemar: GSCSteer vs Original and vs best baseline (excl. GSCSteer)
    gsc = all_preds["GSCSteer"]["is_correct"].to_numpy(dtype=bool)
    baselines = [n for n in METHOD_DISPLAY_ORDER if n not in ("GSCSteer",)]
    best_base = max(baselines, key=lambda n: all_metrics[n]["accuracy"])
    mcnemar = {
        "gscsteer_vs_original": mcnemar_pvalue(gsc, original_correct),
        "gscsteer_vs_best_baseline": {
            "baseline": best_base,
            **mcnemar_pvalue(gsc, all_preds[best_base]["is_correct"].to_numpy(dtype=bool)),
        },
    }

    metrics_json = {
        "model": MODEL_NAME,
        "model_path": model_path,
        "dataset_source": DATASET_SOURCE,
        "dataset_version": DATASET_VERSION,
        "calibration_size": int(split_stats["splits"]["calibration_full_500"]["n"]),
        "test_size": int(split_stats["splits"]["test_500"]["n"]),
        "seed": SEED,
        "evaluation_mode": "forced_choice_loglikelihood",
        "parameter_source": "truthfulqa_llama3.1-8b_zero_tuning_transfer",
        "methods": methods_blob,
        "overall_accuracy": {n: all_metrics[n]["accuracy"] for n in METHOD_DISPLAY_ORDER},
        "accuracy_gain_vs_original": {
            n: all_metrics[n]["accuracy"] - orig_acc for n in METHOD_DISPLAY_ORDER
        },
        "cause_accuracy": {n: all_metrics[n]["cause_accuracy"] for n in METHOD_DISPLAY_ORDER},
        "effect_accuracy": {n: all_metrics[n]["effect_accuracy"] for n in METHOD_DISPLAY_ORDER},
        "choice1_accuracy": {n: all_metrics[n]["choice1_accuracy"] for n in METHOD_DISPLAY_ORDER},
        "choice2_accuracy": {n: all_metrics[n]["choice2_accuracy"] for n in METHOD_DISPLAY_ORDER},
        "confidence_interval_95": {
            n: [all_metrics[n]["ci95_low"], all_metrics[n]["ci95_high"]]
            for n in METHOD_DISPLAY_ORDER
        },
        "mcnemar_test": mcnemar,
        "invalid_output_rate": {n: 0.0 for n in METHOD_DISPLAY_ORDER},
        "runtime": {n: all_metrics[n]["runtime_sec"] for n in METHOD_DISPLAY_ORDER},
        "resolved_parameters": resolved_config,
    }
    save_json(metrics_json, out_dir / "copa_metrics.json")
    print(table_df.to_string(index=False))
    print(f"✓ Wrote results under {out_dir}")


@hydra.main(
    config_path=str(get_project_dir() / "confs"),
    config_name="copa",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    project_dir = get_project_dir()
    mode = str(cfg.mode)
    # Keep smoke outputs out of the formal results directory
    out_dir = project_dir / "results" / "copa" / "llama3.1-8b"
    if mode.startswith("smoke"):
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_idx = int(cfg.layer_idx)
    batch_size = int(cfg.batch_size)
    assert cfg.model == MODEL_NAME, "COPA script is locked to Llama3.1-8B-Base"

    # --- data ---
    dev, test = load_copa_official()
    build, validation = split_dev_for_validation(dev, seed=int(cfg.seed))
    split_stats = compute_split_statistics(build, validation, test, calibration_full=dev)
    save_json(split_stats, out_dir / "copa_split_statistics.json")

    # leakage hard-fail for build/test (pair key order follows leakage_check nesting)
    idx_ov = split_stats["leakage"]["pairwise_idx_overlap"]
    text_ov = split_stats["leakage"]["pairwise_text_overlap"]
    for key in ("build_400_vs_test_500", "validation_100_vs_test_500", "test_500_vs_calibration_full_500"):
        if idx_ov.get(key):
            raise RuntimeError(f"Leakage: idx overlap at {key}: {idx_ov[key][:10]}...")
        if text_ov.get(key, 0) != 0:
            raise RuntimeError(f"Leakage: text overlap at {key}")

    # Contrastive metadata for full calibration (and optional build-only)
    save_json(
        build_contrastive_metadata(dev),
        out_dir / "copa_contrastive_metadata_calibration500.json",
    )
    save_json(
        build_contrastive_metadata(build),
        out_dir / "copa_contrastive_metadata_build400.json",
    )
    save_json(samples_to_records(test), out_dir / "copa_test_samples.json")

    # Formal experiment: fit on full 500 development; evaluate on 500 test
    vk_path = resolve_vk_path(project_dir, layer_idx, MANIFOLD_K_TRUTHFULQA)
    methods = default_method_specs(vk_path)

    if mode == "smoke":
        cal_samples = build[:8]
        test_samples = test[:8]
    elif mode == "smoke_full_cal":
        cal_samples = build
        test_samples = test[:8]
    else:
        cal_samples = dev  # full 500 development
        test_samples = test

    model_path = _FULL_LLM_NAMES[MODEL_NAME]
    resolved_config = {
        "model": MODEL_NAME,
        "model_path": model_path,
        "layer_idx": layer_idx,
        "layer_note": "0-indexed code layer_idx=13 corresponds to paper's 14th layer",
        "seed": int(cfg.seed),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "dtype": "float32",
        "odesteer_T": ODE_T,
        "baseline_T": BASELINE_T,
        "ode_solver": ODE_SOLVER,
        "ode_steps": ODE_STEPS,
        "manifold_k": MANIFOLD_K_TRUTHFULQA,
        "vk_path": str(vk_path),
        "parameter_source": "truthfulqa_llama3.1-8b_zero_tuning_transfer",
        "evaluation_mode": "forced_choice_loglikelihood",
        "mode": mode,
        "calibration_n": len(cal_samples),
        "test_n": len(test_samples),
        "methods": [asdict(m) for m in methods],
        "sampling_note": (
            "temperature/top_p/repetition_penalty retained in snapshot for parity with "
            "generation experiments; forced-choice scoring is deterministic and does not sample."
        ),
    }
    with open(out_dir / "copa_resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)

    print("→ Resolved parameters:")
    print(yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True))

    all_preds: dict[str, pd.DataFrame] = {}
    all_metrics: dict[str, dict[str, Any]] = {}
    for spec in methods:
        df, metrics = run_one_method(
            spec,
            cal_samples,
            test_samples,
            layer_idx=layer_idx,
            batch_size=batch_size,
            vk_path=vk_path,
            out_dir=out_dir,
        )
        all_preds[spec.display_name] = df
        all_metrics[spec.display_name] = metrics

    # Consistency checks
    test_ids = [s.idx for s in test_samples]
    for name, df in all_preds.items():
        if df["idx"].tolist() != test_ids:
            raise RuntimeError(f"{name}: test idx order mismatch")

    aggregate_results(
        all_preds,
        all_metrics,
        out_dir=out_dir,
        model_path=model_path,
        split_stats=split_stats,
        resolved_config=resolved_config,
    )


if __name__ == "__main__":
    main()
