#!/usr/bin/env python3
"""
COPA smoke checks that do not require a GPU / full model load:
  - XML load
  - split sizes / stratification
  - leakage
  - contrastive metadata
  - prompt template stability
  - method registry names

For an 8-sample end-to-end model smoke (loads Llama3.1-8B):
  uv run python -u scripts/copa/copa_run.py mode=smoke
"""

from __future__ import annotations

from odesteer.steer import get_steer_model
from odesteer.utils.copa import (
    COPA_PROMPT_TEMPLATE,
    build_contrastive_metadata,
    compute_split_statistics,
    format_copa_prompt,
    load_copa_official,
    split_dev_for_validation,
)


def main() -> None:
    print("→ Loading official COPA XML ...")
    dev, test = load_copa_official()
    assert len(dev) == 500 and len(test) == 500
    build, val = split_dev_for_validation(dev)
    assert len(build) == 400 and len(val) == 100
    stats = compute_split_statistics(build, val, test, calibration_full=dev)
    print("  split sizes:", {k: v["n"] for k, v in stats["splits"].items()})
    print("  build cause/effect:", stats["splits"]["build_400"]["by_question"])
    print("  build labels:", stats["splits"]["build_400"]["by_label"])

    leaks = stats["leakage"]["pairwise_idx_overlap"]
    text_leaks = stats["leakage"]["pairwise_text_overlap"]
    for key in (
        "build_400_vs_test_500",
        "validation_100_vs_test_500",
        "test_500_vs_calibration_full_500",
    ):
        assert leaks.get(key, []) == [], f"idx leak at {key}: {leaks.get(key)}"
        assert text_leaks.get(key, 0) == 0, f"text leak at {key}"
    print("✓ No idx/text leakage between calibration/build/val and test")

    meta = build_contrastive_metadata(dev[:3])
    for row, s in zip(meta, dev[:3]):
        assert row["prompt"] == format_copa_prompt(s)
        assert row["prompt"].endswith("Answer:")
        assert "\nAnswer:" in row["prompt"]
        assert row["correct_letter"] in ("A", "B")
        assert row["incorrect_letter"] != row["correct_letter"]
        # template must match the frozen string (no chat wrapping)
        assert COPA_PROMPT_TEMPLATE.split("{premise}")[0] in row["prompt"]
    print("✓ Contrastive metadata / prompt template OK")

    # Method registry (GSCSteer is CRFDSteer + manifold flags in the runner)
    for name in ("NoSteer", "RepE", "ITI", "CAA", "MiMiC", "LinAcT", "ODESteer", "CRFDSteer"):
        if name == "NoSteer":
            assert get_steer_model(name) is None
        else:
            # instantiate with manifold flags off (no V_k)
            obj = get_steer_model(
                name,
                use_manifold=False,
                use_manifold_retraction=False,
                vk_path=None,
                manifold_dim=128,
                **(
                    {"solver": "euler", "steps": 10, "n_components": 64, "degree": 2, "gamma": 0.1, "coef0": 1.0}
                    if name in ("ODESteer", "CRFDSteer")
                    else {}
                ),
                **({"ot_reg": 1e-2} if name == "MiMiC" else {}),
            )
            assert obj is not None
            print(f"  registered {name}: {type(obj).__name__}")
    print("✓ All 8 method entry points resolve")
    print("✓ COPA smoke (data + registry) passed")
    print("Next: uv run python -u scripts/copa/copa_run.py mode=smoke")


if __name__ == "__main__":
    main()
