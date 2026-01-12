import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from comet import download_model, load_from_checkpoint  # pip: unbabel-comet


@dataclass
class Example:
    idx: int
    src: str
    ref: str
    base: str
    debate: str
    path: str

def load_json_robust(path):
    """
    Robust JSON loader for Windows:
    try UTF-8 first, fallback to GBK.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            return json.load(f)

def load_examples(output_dir: str) -> List[Example]:
    """
    读取 output_dir 下的 0.json, 1.json...（只认纯数字文件名）
    跳过 0-config.json 等。
    需要字段：source / reference / base_translation / debate_translation
    """
    files = glob.glob(os.path.join(output_dir, "*.json"))

    numbered = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if name.isdigit():
            numbered.append(p)

    numbered.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    if not numbered:
        raise RuntimeError(f"No numbered json files found in {output_dir}")

    examples: List[Example] = []
    for p in numbered:
        idx = int(os.path.splitext(os.path.basename(p))[0])
        obj: Dict[str, Any] = load_json_robust(p)

        src = str(obj.get("source", "")).strip()
        ref = str(obj.get("reference", "")).strip()
        base = str(obj.get("base_translation", "")).strip()
        debate = str(obj.get("debate_translation", "")).strip()

        # COMET(da) 必须有 reference；base/debate 为空也没法算
        if not src or not ref or not base or not debate:
            continue

        examples.append(Example(idx, src, ref, base, debate, p))

    if not examples:
        raise RuntimeError(
            "No valid examples. Check that each result json has non-empty "
            "'source', 'reference', 'base_translation', 'debate_translation'."
        )

    return examples


def predict_scores(model, srcs: List[str], mts: List[str], refs: List[str], batch_size: int) -> List[float]:
    data = [{"src": s, "mt": mt, "ref": r} for s, mt, r in zip(srcs, mts, refs)]
    gpus = 1 if getattr(model, "device", None) is not None and getattr(model.device, "type", "") == "cuda" else 0
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return list(out.scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="例如：data/lexical_output 或 lexical_output")
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da", help="论文常用 reference-based COMET")
    ap.add_argument("--batch_size", type=int, default=8, help="显存/内存不够可调小到 4/2")
    ap.add_argument("--out_csv", default="comet_scores.csv", help="输出 CSV 文件名（默认写在 output_dir 里）")
    args = ap.parse_args()

    examples = load_examples(args.output_dir)
    examples = sorted(examples, key=lambda x: x.idx)

    srcs = [e.src for e in examples]
    refs = [e.ref for e in examples]
    base_mts = [e.base for e in examples]
    debate_mts = [e.debate for e in examples]

    print(f"Scoring {len(examples)} examples from: {args.output_dir}")
    print(f"COMET model: {args.model}")

    ckpt = download_model(args.model)
    model = load_from_checkpoint(ckpt)

    base_scores = predict_scores(model, srcs, base_mts, refs, batch_size=args.batch_size)
    debate_scores = predict_scores(model, srcs, debate_mts, refs, batch_size=args.batch_size)

    df = pd.DataFrame({
        "id": [e.idx for e in examples],
        "source": srcs,
        "reference": refs,
        "base_translation": base_mts,
        "debate_translation": debate_mts,
        "comet_base": base_scores,
        "comet_debate": debate_scores,
    })
    df["delta_debate_minus_base"] = df["comet_debate"] - df["comet_base"]

    mean_base = float(df["comet_base"].mean())
    mean_debate = float(df["comet_debate"].mean())
    mean_delta = float(df["delta_debate_minus_base"].mean())
    win_rate = float((df["delta_debate_minus_base"] > 0).mean())

    print("\n===== COMET Summary (paper-style) =====")
    print(f"N = {len(df)}")
    print(f"COMET(base)   mean = {mean_base:.4f}")
    print(f"COMET(debate) mean = {mean_debate:.4f}")
    print(f"Δ (debate-base) mean = {mean_delta:.4f}")
    print(f"Win-rate (debate > base) = {win_rate:.2%}")

    out_csv = args.out_csv
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(args.output_dir, out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved per-sentence scores to: {out_csv}")


if __name__ == "__main__":
    main()
