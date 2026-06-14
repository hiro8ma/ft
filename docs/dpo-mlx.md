# DPO（選好最適化）を Apple Silicon の MLX で回す

このリポ（MLX + LoRA で Gemma 3 4B / Qwen3 を Apple Silicon ローカル fine-tuning）に、SFT の次段である DPO を追加するためのレシピと前提整理。

## 結論 — MLX で DPO は現状できるか

| 手段 | DPO 対応 | 備考 |
|---|---|---|
| `mlx_lm.lora`（本リポが SFT で使用、v0.31.1） | 非対応 | データ形式は `text` / `messages`(chat) / `prompt`+`completion` のみ。`chosen`/`rejected` は読めない。`--fine-tune-type` は `lora`/`dora`/`full`、損失は cross-entropy と KL/JS 蒸留のみ |
| `mlx-lm-lora`（コミュニティ拡張パッケージ、PyPI v2.1.0） | 対応 | `mlx_lm_lora.train --train-mode dpo` で DPO/ORPO/CPO/GRPO/Online DPO を Apple Silicon ネイティブに実行。`mlx_lm>=0.30.6` に依存 |

結論: **mlx-lm 本体だけでは DPO は不可。`mlx-lm-lora` を追加導入すれば Mac ローカルで DPO が回せる。** 本リポはこの拡張を使う前提でレシピを用意した。`docs/post-training-spectrum.md` の「MLX は最新 RL が手薄」は、`mlx-lm-lora` の登場で DPO/GRPO まで埋まりつつある（2026 時点）。

## DPO とは（SFT との違い）

- SFT は「正解の回答」を 1 本与えて模倣させる（教師あり）。
- DPO は **同じ prompt に対する 2 本の回答（chosen=良い / rejected=悪い）** を与え、chosen の確率を上げ rejected を下げる。報酬モデルも PPO ループも不要で、選好を二値分類として直接最適化する。RLHF 比で計算 40〜75% 減・学習が安定。
- `beta` は参照モデル（学習前の方策）からの逸脱への KL ペナルティ強度。小さいほど選好に強く合わせ、大きいほど元の挙動を保つ。初期値 0.1。

## データ形式（mlx-lm-lora の選好フォーマット）

JSONL 1 行 1 サンプル。`prompt` / `chosen` / `rejected` が必須。`system` は任意。

```json
{"prompt": "質問", "chosen": "良い回答", "rejected": "悪い回答"}
{"system": "あなたは専門家です", "prompt": "質問", "chosen": "良い回答", "rejected": "悪い回答"}
```

本リポのサンプル: `recipes/dpo/preference_sample.jsonl`（6 件、AI エンジニアリング Q&A の chosen/rejected ペア）。

## 実行手順

```bash
# 1. mlx-lm-lora を追加（uv。本体 mlx-lm は据え置き）
uv add mlx-lm-lora

# 2. 選好データを train/valid に分割（data_dpo/ に出力。gitignore 済）
make prepare-dpo

# 3. DPO 実行（LoRA アダプタとして学習。adapters-dpo/ に保存）
make train-dpo

# 4. 学習後の対話確認（既存の chat.py を adapters-dpo 向けに使う）
uv run python chat.py   # ADAPTER_PATH を adapters-dpo に向ける場合は chat.py 側を調整
```

`make train-dpo` が実行するコマンド（Makefile）:

```bash
uv run mlx_lm_lora.train \
  --model mlx-community/gemma-3-4b-it-4bit \
  --train \
  --train-mode dpo \
  --train-type lora \
  --data data_dpo \
  --beta 0.1 \
  --batch-size 1 \
  --num-layers 8 \
  --learning-rate 1e-5 \
  --iters 100 \
  --max-seq-length 512 \
  --adapter-path adapters-dpo \
  --grad-checkpoint
```

### 検証状況

- 確認済（公式 README）: `--train` / `--train-mode dpo` / `--train-type lora`（`lora`/`dora`/`full`）/ `--data` / `--beta` / `--model`。
- 要検証（mlx_lm.lora の慣習から流用。実機で `mlx_lm_lora.train --help` を確認すること）: `--num-layers` / `--iters` / `--max-seq-length` / `--grad-checkpoint` / `--batch-size` / `--learning-rate` の正確な引数名と既定値。
- 未実行: モデル DL と GPU 学習は本作業では回していない（ダウンロード/メモリが要るため）。`make prepare-dpo` のデータ生成のみ実行・確認済。

## LoRA / DPO ハイパラの目安

| パラメータ | 値 | 意図 |
|---|---|---|
| `train-type` | `lora` | 重み凍結 + 低ランク差分。Mac の統合メモリで 4B を学習可 |
| `lora rank (r)` | 8 | 本リポ SFT と揃える。小データなら 8〜16 |
| `lora alpha` | 16（= 2r 目安） | スケーリング係数。r の 2 倍が定番 |
| `beta` | 0.1 | KL ペナルティ。選好に強く寄せたいなら下げる |
| `num-layers` | 8 | 後段レイヤのみアダプタを差す（メモリ節約） |
| `learning-rate` | 1e-5 | DPO は SFT より小さめが安定 |

## MLX で DPO が使えない / 重い場合の代替

1. **SFT で chosen のみ学習**: 選好の「良い側」だけを通常の SFT データに変換して `mlx_lm.lora` で学習。DPO の効果（悪い回答を明示的に下げる）は得られないが、本体だけで完結。
2. **ORPO に切替**: `mlx-lm-lora` の `--train-mode orpo`。参照モデル不要で、SFT と選好最適化を 1 段で行うためメモリが軽い。Mac の RAM が厳しいときの第一候補。
3. **外部 CUDA で DPO**: `trl` の `DPOTrainer`（HuggingFace + PEFT、CUDA / bitsandbytes）で学習し、できた LoRA アダプタを Mac に持ち帰って推論。`docs/post-training-spectrum.md` の「Mac=実験、本番 RL=CUDA」分担に沿う。

## 関連

- `docs/post-training-spectrum.md` — CPT/SFT/PEFT/選好チューニングの全体スペクトル
- `docs/fine-tuning-vs-rag-vs-grounding.md` — FT を選ぶ前の判断
- `recipes/dpo/preference_sample.jsonl` — chosen/rejected サンプル
- `recipes/dpo/prepare_dpo.py` — train/valid 分割スクリプト
