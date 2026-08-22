"""
COPA (Choice of Plausible Alternatives) data adapter for zero-tuning transfer
from existing Llama3.1-8B TruthfulQA steering setups.

Uses the official COPA XML under ``data/COPA/datasets/`` (500 dev + 500 test).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from . import get_project_dir

COPA_PROMPT_TEMPLATE = (
    "Question:\n"
    "{premise} Based on the previous passage, choose the most reasonable {question}.\n"
    "A: {choice1}\n"
    "B: {choice2}\n"
    "\n"
    "Answer:"
)

DATASET_SOURCE = "COPA-resources (Roemmele et al., 2011)"
DATASET_VERSION = "1.0"
SEED = 42


@dataclass(frozen=True)
class COPASample:
    idx: int
    premise: str
    question: Literal["cause", "effect"]
    choice1: str
    choice2: str
    label: int  # 0 -> choice1, 1 -> choice2

    @property
    def gold_letter(self) -> str:
        return "A" if self.label == 0 else "B"

    @property
    def incorrect_letter(self) -> str:
        return "B" if self.label == 0 else "A"


def default_copa_root() -> Path:
    return get_project_dir() / "data" / "COPA"


def format_copa_prompt(sample: COPASample) -> str:
    """Exact COPA prompt shared by all methods (activation fit + MC scoring)."""
    return COPA_PROMPT_TEMPLATE.format(
        premise=sample.premise,
        question=sample.question,
        choice1=sample.choice1,
        choice2=sample.choice2,
    )


def _parse_copa_xml(xml_path: Path) -> list[COPASample]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples: list[COPASample] = []
    for item in root.findall("item"):
        idx = int(item.attrib["id"])
        question = item.attrib["asks-for"]
        # most-plausible-alternative is 1-based in XML
        alt = int(item.attrib["most-plausible-alternative"])
        label = alt - 1
        premise = (item.findtext("p") or "").strip()
        choice1 = (item.findtext("a1") or "").strip()
        choice2 = (item.findtext("a2") or "").strip()
        if question not in ("cause", "effect"):
            raise ValueError(f"Unexpected question type {question!r} for idx={idx}")
        if label not in (0, 1):
            raise ValueError(f"Unexpected label {label} for idx={idx}")
        samples.append(
            COPASample(
                idx=idx,
                premise=premise,
                question=question,  # type: ignore[arg-type]
                choice1=choice1,
                choice2=choice2,
                label=label,
            )
        )
    return samples


def load_copa_official(
    copa_root: Path | None = None,
) -> tuple[list[COPASample], list[COPASample]]:
    """Load official 500-dev and 500-test samples with real labels."""
    root = Path(copa_root) if copa_root is not None else default_copa_root()
    dev_path = root / "datasets" / "copa-dev.xml"
    test_path = root / "datasets" / "copa-test.xml"
    if not dev_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"COPA XML not found under {root}/datasets/. "
            "Expected copa-dev.xml and copa-test.xml."
        )
    dev = _parse_copa_xml(dev_path)
    test = _parse_copa_xml(test_path)
    if len(dev) != 500 or len(test) != 500:
        raise ValueError(f"Expected 500/500, got {len(dev)}/{len(test)}")
    return dev, test


def _stratify_keys(samples: list[COPASample]) -> list[str]:
    return [f"{s.question}_{s.label}" for s in samples]


def split_dev_for_validation(
    dev: list[COPASample],
    seed: int = SEED,
    val_size: int = 100,
) -> tuple[list[COPASample], list[COPASample]]:
    """
    Stratified split of 500 development samples into 400 build + 100 validation
    by (question_type × label). Test remains the official 500-test set.
    """
    if len(dev) != 500:
        raise ValueError(f"Expected 500 development samples, got {len(dev)}")
    indices = np.arange(len(dev))
    build_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=_stratify_keys(dev),
    )
    build = [dev[i] for i in sorted(build_idx.tolist())]
    val = [dev[i] for i in sorted(val_idx.tolist())]
    return build, val


def count_by_question(samples: list[COPASample]) -> dict[str, int]:
    out = {"cause": 0, "effect": 0}
    for s in samples:
        out[s.question] += 1
    return out


def count_by_label(samples: list[COPASample]) -> dict[str, int]:
    out = {"label_0": 0, "label_1": 0}
    for s in samples:
        out[f"label_{s.label}"] += 1
    return out


def leakage_check(
    *splits: tuple[str, list[COPASample]],
) -> dict[str, Any]:
    """Check duplicate idx / (premise, choice1, choice2) across splits."""
    report: dict[str, Any] = {"pairwise_idx_overlap": {}, "pairwise_text_overlap": {}}
    named = list(splits)
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            n_i, a = named[i]
            n_j, b = named[j]
            idx_a = {s.idx for s in a}
            idx_b = {s.idx for s in b}
            text_a = {(s.premise, s.choice1, s.choice2) for s in a}
            text_b = {(s.premise, s.choice1, s.choice2) for s in b}
            key = f"{n_i}_vs_{n_j}"
            report["pairwise_idx_overlap"][key] = sorted(idx_a & idx_b)
            report["pairwise_text_overlap"][key] = len(text_a & text_b)
    return report


def build_contrastive_metadata(samples: list[COPASample]) -> list[dict[str, Any]]:
    """Lightweight metadata for pos/neg prompts (no activations)."""
    rows: list[dict[str, Any]] = []
    for s in samples:
        prompt = format_copa_prompt(s)
        rows.append(
            {
                "idx": s.idx,
                "prompt": prompt,
                "correct_letter": s.gold_letter,
                "incorrect_letter": s.incorrect_letter,
                "question_type": s.question,
                "label": s.label,
            }
        )
    return rows


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def samples_to_records(samples: list[COPASample]) -> list[dict[str, Any]]:
    return [asdict(s) for s in samples]


def compute_split_statistics(
    build: list[COPASample],
    validation: list[COPASample],
    test: list[COPASample],
    calibration_full: list[COPASample],
) -> dict[str, Any]:
    splits = {
        "build_400": build,
        "validation_100": validation,
        "calibration_full_500": calibration_full,
        "test_500": test,
    }
    stats: dict[str, Any] = {
        "dataset_source": DATASET_SOURCE,
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "splits": {},
        "leakage": leakage_check(
            ("build_400", build),
            ("validation_100", validation),
            ("test_500", test),
            ("calibration_full_500", calibration_full),
        ),
    }
    for name, samples in splits.items():
        stats["splits"][name] = {
            "n": len(samples),
            "by_question": count_by_question(samples),
            "by_label": count_by_label(samples),
            "idx_min": min(s.idx for s in samples) if samples else None,
            "idx_max": max(s.idx for s in samples) if samples else None,
            "idx_list": [s.idx for s in samples],
        }
    return stats
