# 小規模モデルの選び方と HF/CUDA → MLX 読み替え

書籍の fine-tuning 例は HuggingFace + PyTorch + `bitsandbytes`（4bit, CUDA 前提）で書かれることが多い。`bitsandbytes` は Triton/CUDA カーネル依存で **Apple Silicon では動かない**。本リポ（MLX + LoRA）で同等のことをやるための読み替えをまとめる。

## 小規模モデル（8B 未満）を選ぶ基準

### なぜ小規模か（利点）
- **単一デバイスで完結**: 4bit 量子化した 4B 級なら Apple Silicon の統合メモリに載り、ローカルで学習・推論できる。クラウド GPU もデータ送出も不要（プライバシー）。
- **反復が速い**: LoRA/DPO の 1 回が数分〜数十分。プロンプト改善やデータ修正のループを高速に回せる。
- **推論コストとレイテンシが低い**: 本番投入時の単価・応答速度で有利。
- **タスクを絞れば大規模に肉薄**: 分類・抽出・定型回答・特定ドメイン QA など範囲が狭いタスクでは、FT した小モデルが汎用大モデルに匹敵する。

### タスク別の選び方
| タスク | 推奨サイズ感 | 理由 |
|---|---|---|
| 分類 / 抽出 / ルーティング | 0.5B〜2B | 出力が短く構造的。小モデルで十分、レイテンシ最優先 |
| 定型ドメイン QA / スタイル固定 | 2B〜4B | SFT/DPO でフォーマットと口調を焼き付ける用途に最適 |
| 推論・多段思考が要る | 4B〜8B+ または大モデル | 小モデルは多段推論で崩れやすい。GRPO 等の RL か大モデルを検討 |
| 汎用アシスタント | FT より RAG/大モデル | 知識を重みに焼くより retrieval が有利 |

本リポの採用: **Gemma 3 4B IT 4bit**（メイン）、**Qwen3-0.6B 4bit**（Reasoning 比較用）。どちらも `mlx-community` の 4bit 版を使い、ダウンロード後すぐ LoRA できる。

### 選定チェックリスト
1. ライセンス（商用可否、蒸留教師に使う場合は教師モデルの ToS）
2. `mlx-community` に量子化版があるか（無ければ自前で `mlx_lm.convert -q`）
3. instruct 版か base 版か（SFT/DPO は通常 instruct 版を起点）
4. コンテキスト長と語彙が対象タスクに足りるか

## HF/CUDA 例 → MLX 読み替え表

| 書籍/HF 側（CUDA） | 本リポ MLX 側 | 補足 |
|---|---|---|
| `bitsandbytes` 4bit (`load_in_4bit=True`) | **MLX 量子化モデル**（`mlx-community/...-4bit`）または `mlx_lm.convert -q --q-bits 4` | bitsandbytes は Mac 不可。MLX は量子化済み重みをそのままロード |
| `transformers.AutoModelForCausalLM` | `mlx_lm.load(model)` | MLX 形式の重みをロード |
| `peft.LoraConfig(r=8, lora_alpha=16)` | `--lora-rank 8`（`mlx_lm.lora`） / `--train-type lora`（`mlx_lm_lora.train`） | alpha は MLX 既定に従う。rank を CLI 指定 |
| `peft` `target_modules=["q_proj","v_proj",...]` | `--num-layers N`（後段 N 層にアダプタ）。モジュール粒度指定は `--config` の YAML で | MLX は「どの層に差すか」を層数で制御するのが標準。細粒度は config |
| `fine_tune_type` 相当 | `--fine-tune-type {lora,dora,full}`（mlx_lm.lora） / `--train-type`（mlx_lm_lora） | DoRA も両方で対応 |
| `trl.SFTTrainer` | `mlx_lm.lora --train`（SFT） | データは `messages` chat 形式 or `prompt`/`completion` |
| `trl.DPOTrainer(beta=0.1)` | `mlx_lm_lora.train --train-mode dpo --beta 0.1` | **mlx_lm.lora 本体は DPO 非対応**。拡張パッケージが必要（`docs/dpo-mlx.md`） |
| `trl.ORPOTrainer` | `mlx_lm_lora.train --train-mode orpo` | 参照モデル不要でメモリ軽 |
| `trl.GRPOTrainer` | `mlx_lm_lora.train --train-mode grpo` | 検証可能報酬タスク向け |
| `TrainingArguments(num_train_epochs=3)` | `--epochs 3` または `--iters N` | MLX は iters 指定も可 |
| `gradient_checkpointing=True` | `--grad-checkpoint` | メモリ削減。Mac の RAM が厳しいとき有効 |
| `device_map="auto"` / `accelerate` | 不要 | MLX は統合メモリ前提でデバイス分散の概念が無い |

### DPO データ形式の対応
| HF (`trl` DPO) | MLX (`mlx_lm_lora`) |
|---|---|
| `{"prompt":..., "chosen":..., "rejected":...}` | 同一（`system` も任意で付与可） |

DPO の選好データ形式は HF と MLX でほぼ同じなので、`trl` 用に作った選好データセットを **そのまま `mlx_lm_lora` に流用できる**。

## まとめの方針
- 書籍が `bitsandbytes` を出してきたら「Mac 不可 → mlx-community の 4bit 版 or `mlx_lm.convert -q`」と即読み替える。
- `SFTTrainer` → `mlx_lm.lora --train`、`DPOTrainer` → `mlx_lm_lora.train --train-mode dpo`。
- `target_modules` の細粒度指定は MLX では `--num-layers` + config YAML に丸める。

## 関連
- `docs/dpo-mlx.md` — DPO レシピと検証状況
- `docs/post-training-spectrum.md` — 事後学習スペクトル全体
- `docs/transformer-internals.md` — q_proj/v_proj 等が何を指すか
