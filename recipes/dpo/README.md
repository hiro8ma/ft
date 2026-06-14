# DPO レシピ（MLX / Apple Silicon）

`mlx-lm-lora` を使った DPO（選好最適化）の最小レシピ。設計と検証状況の詳細は `../../docs/dpo-mlx.md`。

## ファイル
- `preference_sample.jsonl` — chosen/rejected の選好サンプル（6 件）
- `prepare_dpo.py` — サンプルを `data_dpo/{train,valid}.jsonl` に分割（出力は gitignore 済）

## 手順
```bash
uv add mlx-lm-lora   # 本体 mlx-lm は据え置き
make prepare-dpo     # 選好データを分割
make train-dpo       # DPO 実行（adapters-dpo/ に LoRA アダプタ保存）
```

## データ形式
```json
{"system": "あなたは専門家です", "prompt": "質問", "chosen": "良い回答", "rejected": "悪い回答"}
```
`system` は任意。`prompt` / `chosen` / `rejected` が必須。
