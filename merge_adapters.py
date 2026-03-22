"""
2つのLoRAアダプターを線形マージして統合アダプターを作成する。

LoRA-A (AI Engineering) + LoRA-B (Management) → Merged Adapter
"""

import json
from pathlib import Path

import mlx.core as mx

ADAPTER_A = Path("adapters")
ADAPTER_B = Path("adapters-management")
OUTPUT = Path("adapters-merged")


def merge_adapters(
    path_a: Path, path_b: Path, output: Path, weight_a: float = 0.5
):
    """2つのアダプターを加重平均でマージする。"""
    weight_b = 1.0 - weight_a

    # アダプターの重みを読み込み
    weights_a = mx.load(str(path_a / "adapters.safetensors"))
    weights_b = mx.load(str(path_b / "adapters.safetensors"))

    print(f"LoRA-A: {len(weights_a)} tensors ({path_a})")
    print(f"LoRA-B: {len(weights_b)} tensors ({path_b})")

    # キーが一致することを確認
    keys_a = set(weights_a.keys())
    keys_b = set(weights_b.keys())

    if keys_a != keys_b:
        common = keys_a & keys_b
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        print(f"Warning: キーが一致しません")
        print(f"  共通: {len(common)}, A only: {len(only_a)}, B only: {len(only_b)}")
        # 共通キーのみマージ
        keys = common
    else:
        keys = keys_a

    # 加重平均でマージ
    merged = {}
    for key in sorted(keys):
        merged[key] = weight_a * weights_a[key] + weight_b * weights_b[key]

    print(f"マージ完了: {len(merged)} tensors (weight_a={weight_a}, weight_b={weight_b})")

    # 保存
    output.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output / "adapters.npz"), **merged)

    # safetensors形式で保存
    mx.save_safetensors(str(output / "adapters.safetensors"), merged)

    # adapter_config.json をコピー（LoRA-Aのものを使用）
    config_a = json.loads((path_a / "adapter_config.json").read_text())
    (output / "adapter_config.json").write_text(
        json.dumps(config_a, indent=2, ensure_ascii=False)
    )

    print(f"保存先: {output}")
    return merged


if __name__ == "__main__":
    import sys

    weight_a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    merge_adapters(ADAPTER_A, ADAPTER_B, OUTPUT, weight_a=weight_a)
