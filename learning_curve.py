"""
データ量とモデル性能の関係を検証する学習曲線実験。

データセットの25%, 50%, 100%で学習し、損失の変化をプロットする。
AIエンジニアリング Ch8 p.386 の理論を自分のデータで実証。
"""

import json
import subprocess
import sys
from pathlib import Path


DATA_DIR = Path("data/combined")
MODEL = "mlx-community/gemma-3-4b-it-4bit"
RESULTS_FILE = Path("learning_curve_results.json")


def count_samples(filepath: Path) -> int:
    """JSONLファイルのサンプル数をカウント。"""
    with open(filepath) as f:
        return sum(1 for _ in f)


def create_subset(ratio: float, output_dir: Path):
    """トレーニングデータのサブセットを作成。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # train.jsonl のサブセットを作成
    train_path = DATA_DIR / "train.jsonl"
    with open(train_path) as f:
        lines = f.readlines()

    subset_size = max(1, int(len(lines) * ratio))
    subset_lines = lines[:subset_size]

    with open(output_dir / "train.jsonl", "w") as f:
        f.writelines(subset_lines)

    # valid.jsonl と test.jsonl はそのままコピー
    for filename in ["valid.jsonl", "test.jsonl"]:
        src = DATA_DIR / filename
        dst = output_dir / filename
        if src.exists():
            dst.write_text(src.read_text())

    return subset_size


def run_training(data_dir: Path, adapter_path: Path, iters: int = 100) -> dict:
    """LoRA FTを実行し、結果を返す。"""
    cmd = [
        "uv", "run", "mlx_lm.lora",
        "--model", MODEL,
        "--train",
        "--data", str(data_dir),
        "--batch-size", "1",
        "--num-layers", "4",
        "--learning-rate", "1e-5",
        "--iters", str(iters),
        "--max-seq-length", "256",
        "--adapter-path", str(adapter_path),
        "--grad-checkpoint",
    ]

    env = {"MLX_USE_CPU": "1", "PATH": "/opt/homebrew/bin:/usr/bin:/bin"}
    result = subprocess.run(
        cmd, capture_output=True, text=True, env={**__import__("os").environ, "MLX_USE_CPU": "1"}
    )

    # 出力から最終損失を抽出
    output = result.stdout + result.stderr
    train_loss = None
    val_loss = None

    for line in output.split("\n"):
        if "Train loss" in line and f"Iter {iters}:" in line:
            parts = line.split("Train loss ")
            if len(parts) > 1:
                train_loss = float(parts[1].split(",")[0])
        if "Val loss" in line and f"Iter {iters}:" in line:
            parts = line.split("Val loss ")
            if len(parts) > 1:
                val_loss = float(parts[1].split(",")[0])

    return {"train_loss": train_loss, "val_loss": val_loss, "output": output[-500:]}


def main():
    ratios = [0.25, 0.5, 1.0]
    results = []

    total_samples = count_samples(DATA_DIR / "train.jsonl")
    print(f"全トレーニングデータ: {total_samples} サンプル\n")

    for ratio in ratios:
        subset_dir = Path(f"data/subset_{int(ratio * 100)}")
        adapter_dir = Path(f"adapters-curve-{int(ratio * 100)}")

        subset_size = create_subset(ratio, subset_dir)
        print(f"=== {int(ratio * 100)}% ({subset_size} サンプル) ===")

        result = run_training(subset_dir, adapter_dir)
        result["ratio"] = ratio
        result["samples"] = subset_size
        results.append(result)

        print(f"  Train loss: {result['train_loss']}")
        print(f"  Val loss:   {result['val_loss']}")
        print()

    # 結果を保存
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== 学習曲線の結果 ===")
    print(f"{'データ量':>10} {'Train Loss':>12} {'Val Loss':>12}")
    print("-" * 36)
    for r in results:
        print(f"{r['samples']:>10} {r['train_loss'] or 'N/A':>12} {r['val_loss'] or 'N/A':>12}")

    print(f"\n結果を {RESULTS_FILE} に保存しました。")


if __name__ == "__main__":
    main()
