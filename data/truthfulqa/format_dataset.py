import ast
import os
from pathlib import Path

# 在导入 datasets/huggingface_hub 前设置镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
from datasets import load_dataset

from odesteer.utils import get_project_dir

gen_dir = get_project_dir() / 'data' / 'truthfulqa' / 'texts'
gen_dir.mkdir(parents = True, exist_ok = True)

# 离线模式：HF_HUB_OFFLINE=1 或 ODESteer_OFFLINE=1 时仅用本地缓存
_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("ODESteer_OFFLINE", "0") == "1"

# ModelScope 下载目录：设置 TRUTHFULQA_LOCAL_DIR 或使用项目内默认路径 data/truthfulqa/truthful_qa_raw
_default_local = get_project_dir() / 'data' / 'truthfulqa' / 'truthful_qa_raw'
_TRUTHFULQA_LOCAL_DIR = os.environ.get("TRUTHFULQA_LOCAL_DIR", "") or (
    str(_default_local) if _default_local.exists() else ""
)


def _load_from_local_dir(local_dir: Path) -> pd.DataFrame:
    """从 ModelScope 等下载的本地目录加载 generation 格式数据，返回单表。"""
    required = {"question", "correct_answers", "incorrect_answers"}
    for ext, read_fn in [
        ("csv", lambda p: pd.read_csv(p)),
        ("parquet", lambda p: pd.read_parquet(p) if hasattr(pd, "read_parquet") else None),
        ("jsonl", lambda p: pd.read_json(p, lines=True, orient="records")),
        ("json", lambda p: pd.read_json(p)),
    ]:
        for f in local_dir.rglob(f"*.{ext}"):
            try:
                df = read_fn(f)
                if df is not None and required.issubset(df.columns):
                    return df
            except Exception:
                continue
    raise FileNotFoundError(
        f"在 {local_dir} 下未找到包含 question/correct_answers/incorrect_answers 的 csv/parquet/json 文件。"
    )


def format_truthfulqa_generation():
    local_dir = Path(_TRUTHFULQA_LOCAL_DIR).resolve() if _TRUTHFULQA_LOCAL_DIR else None

    if local_dir and local_dir.exists():
        # 优先：从 ModelScope 等 --local_dir 下载的目录加载
        df = _load_from_local_dir(local_dir)
        ds_train_test = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = len(ds_train_test) // 2
        df1 = ds_train_test.iloc[:split_idx]
        df2 = ds_train_test.iloc[split_idx:]
    else:
        try:
            ds = load_dataset(
                "truthfulqa/truthful_qa",
                "generation",
                split="validation",
                trust_remote_code=True,
                local_files_only=_OFFLINE,
            )
        except (ConnectionError, OSError) as e:
            if not _OFFLINE:
                try:
                    ds = load_dataset(
                        "truthfulqa/truthful_qa",
                        "generation",
                        split="validation",
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    raise RuntimeError(
                        "无法连接 Hugging Face，且本地无缓存。可改用 ModelScope 下载并设置 TRUTHFULQA_LOCAL_DIR，见 README。"
                    ) from e
            else:
                raise
        ds = ds.train_test_split(test_size=0.5, seed=42)
        df1, df2 = ds['train'].to_pandas(), ds['test'].to_pandas()

    for df_idx, df in enumerate([df1, df2]):
        pos_pairs, neg_pairs = [], []
        for idx, row in df.iterrows():
            correct_list = row.correct_answers
            incorrect_list = row.incorrect_answers
            if isinstance(correct_list, str):
                try:
                    correct_list = ast.literal_eval(correct_list)
                except Exception:
                    correct_list = [correct_list]
            if isinstance(incorrect_list, str):
                try:
                    incorrect_list = ast.literal_eval(incorrect_list)
                except Exception:
                    incorrect_list = [incorrect_list]
            for correct_answer in correct_list:
                pos_pairs.append({
                    'idx': idx,
                    'question': str(row.question).strip(),
                    'answer': correct_answer,
                })
            for incorrect_answer in incorrect_list:
                neg_pairs.append({
                    'idx': idx,
                    'question': str(row.question).strip(),
                    'answer': incorrect_answer,
                })
        pos_df = pd.DataFrame(pos_pairs)
        neg_df = pd.DataFrame(neg_pairs)
        pos_df.to_json(gen_dir / f'pos_{df_idx}.jsonl', orient = 'records', lines = True)
        neg_df.to_json(gen_dir / f'neg_{df_idx}.jsonl', orient = 'records', lines = True)
        print(f'Saved {len(pos_df)} positive pairs and {len(neg_df)} negative pairs for {df_idx}th split')

    
if __name__ == '__main__':
    format_truthfulqa_generation()