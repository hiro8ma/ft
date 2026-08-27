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

### 検証状況（2026-08-27 実機で実行）

Apple Silicon で実際に回した。ドキュメント作成時は未実行だったが、依存を入れて通した。

```
mlx-lm-lora 3.1.2 を uv add
DPO_ITERS=10 make train-dpo

Trainable parameters: 0.077% (3.506M / 4551.516M)
Iter  1  Val loss 0.693  accuracy 0.000  margin 0.000
Iter 10  Val loss 0.001  accuracy 1.000  margin 6.772
peak_mem 6.272GB / 10 iter を約 10 秒
```

Val loss の初期値 0.693 は `ln(2)` にあたる。DPO の損失は
「chosen と rejected のどちらが好ましいか」の二値分類なので、
学習前は五分五分になる。理論値がそのまま出ている。

ただし選好データが 6 件しかないため、accuracy 1.000 は過学習を見ているだけになる。
確認できたのはパイプラインが通ることだけで、学習が成立したわけではない。

実行して分かった注意点が 2 つある。

**scale の既定値が mlx-lm と違う**

```
mlx-lm       scale 20.0   （SFT で使用）
mlx-lm-lora  scale 10.0   （DPO で使用）
```

Makefile は `scale` を渡していないため、SFT と DPO で LoRA の強さが 2 倍違う。
DPO は beta で参照モデルからの乖離を制御するので、scale と二重に効く。
揃えるなら `--lora-parameters` で明示する。

**学習後にフルモデルが自動で書き出される**

`mlx-lm-lora` は学習後にアダプタをベースへマージ（fuse）する。
しかも 4bit 量子化を解いた形で出すため、元モデルより大きくなる。

```
adapters.safetensors        13MB   ← アダプタ本体
model-00001-of-00002.safetensors  5.0GB
model-00002-of-00002.safetensors  3.5GB
```

`.gitignore` の `adapters-*/` で塞がっているが、
ディスクは消費する。アダプタだけ残すなら学習後に手で削る。

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
