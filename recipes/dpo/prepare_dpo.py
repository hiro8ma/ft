"""
DPO 用の選好データセット（prompt / chosen / rejected）を train/valid に分割する。

入力: recipes/dpo/preference_sample.jsonl（chosen=良い回答, rejected=悪い回答）
出力: data_dpo/train.jsonl, data_dpo/valid.jsonl
       （data_dpo/ は .gitignore 済。mlx_lm_lora.train --data data_dpo が読む）

mlx-lm-lora が受け付ける選好フォーマット:
  {"prompt": ..., "chosen": ..., "rejected": ...}
  {"system": ..., "prompt": ..., "chosen": ..., "rejected": ...}
"""

import json
import random
from pathlib import Path

SRC = Path(__file__).parent / "preference_sample.jsonl"
OUT_DIR = Path(__file__).parent.parent.parent / "data_dpo"
VALID_RATIO = 0.2
SEED = 42


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    for r in rows:
        if not {"prompt", "chosen", "rejected"} <= r.keys():
            raise ValueError(f"missing chosen/rejected keys: {r}")

    random.Random(SEED).shuffle(rows)
    n_valid = max(1, int(len(rows) * VALID_RATIO))
    valid, train = rows[:n_valid], rows[n_valid:]

    OUT_DIR.mkdir(exist_ok=True)
    for name, split in [("train", train), ("valid", valid)]:
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(split):>3} rows -> {path}")


if __name__ == "__main__":
    main()
