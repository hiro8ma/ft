# 事後学習スペクトル — CPT / SFT / PEFT / 選好チューニングと Unsloth vs MLX

このリポ（MLX + LoRA で Gemma 3 4B / Qwen3 を Apple Silicon ローカル fine-tuning）が事後学習スペクトルのどこに位置するかの整理。

## 事後学習の段階的スタック（2026）

単一手法でなくモジュラーなスタックが標準。

1. **CPT（Continued Pre-Training / 継続事前学習）**: ドメインコーパス（業界用語・社内文書）で事前学習を継続。基盤モデルが持たない語彙・知識を注入。Next Token Prediction を続ける
2. **SFT（Supervised Fine-Tuning / 指示チューニング）**: (prompt, response) ペアで教師あり学習。指示追従を獲得。instruction tuning とも
3. **PEFT（Parameter-Efficient Fine-Tuning）**: 効率的に SFT/適応
   - **LoRA**: 重みを凍結し低ランク行列（全体の 1-5%）だけ学習。full FT の 90-95% 品質
   - **QLoRA**: 4bit 量子化 + LoRA。full FT の 80-90%。7B を RTX 4090（24GB）で学習可（full FT は 100GB+ VRAM）
   - **DoRA**: 重みを「方向」と「大きさ」に分解。ノイズ・少データに強い。QDoRA は full FT/QLoRA を上回る報告あり（このリポに `adapters-dora` 実験あり）
4. **選好チューニング（Preference Tuning）**: 出力ペアの良し悪しで学習
   - **RLHF(PPO)**: 報酬モデル + PPO。重く不安定（概念的基盤）
   - **DPO**: 選好を分類問題として直接最適化、報酬モデル不要。RLHF 比で計算 40-75% 減・安定。一般アラインメントの標準
   - **KTO**: 👍/👎 の二値フィードバック（ペア収集が難しい現場向け）
   - **GRPO**: critic 廃止、K 個生成→検証可能報酬→グループ正規化 advantage。数学/コードなど機械検証できる推論タスクで主役（DeepSeek-R1 で有名化）

選択軸: **データの形（ペア/二値）・計算予算・出力が検証可能か**。

## 必要データ量の目安
- ICL（few/many-shot）: 数個〜数百例（重み不変）
- PEFT（LoRA/QLoRA）: 最小実用 1,000〜5,000 サンプル
- full FT: 大量。数学・コードなど精密調整が要る複雑ドメインで優位

## Unsloth（CUDA）vs MLX-LM（Apple Silicon）← このリポの位置

| 軸 | Unsloth | MLX-LM（本リポ） |
|---|---|---|
| 対象 HW | NVIDIA CUDA GPU | Apple Silicon（M 系） |
| 依存 | Triton カーネル → **Mac では動かない** | Apple 純正 MLX、CUDA toolkit 不要 |
| 速度 | 最速（RTX 3090 は M2 Max の 2-4 倍） | 遅いが統合メモリで大 model ロード可・静音 |
| 対応手法 | SFT/LoRA/QLoRA + 最新 RL（GRPO/DPO） | LoRA/QLoRA 中心、最新 RL は手薄 |
| 立ち位置 | 本番・大規模学習 | **ローカル実験・プライバシー・CUDA 構築回避** |

このリポは MLX + LoRA で Apple Silicon ローカル学習 = 「Mac = 実験/プライバシー、本番スケールや最新 RL は CUDA(Unsloth)」という 2026 の役割分担の**前者側**。橋渡しに `mlx-tune` / `unsloth-mlx`（Unsloth 互換 API）があり、Mac で書いた `FastLanguageModel` スクリプトを CUDA クラスタへ移せる。

## マネージド FT vs セルフホスト
- 学習単価例: Together AI LoRA $0.48/1M（最安）/ OpenAI GPT-4.1 $3/1M / Vertex Gemini 2.0 Flash $3/1M
- 推論: **Vertex は tuned model も base 価格据え置き**が強み。OpenAI は fine-tuned 推論にプレミアム
- break-even: クラウド FT は週 40 時間以上恒常運用まで割安、超えたら自前
- 指針: 実験・反復はマネージド、本番スケールは spot GPU 自前

## ICL → FT 移行の判断（4 軸）
1. **データ量**: 数千件以上あるか（無ければ ICL/many-shot 継続）
2. **推論スケール**: 高頻度なら学習に前払いして推論を軽くする FT が経済的（ICL は毎回例トークン送信）
3. **挙動の根本変更**: スタイル/フォーマットを恒久的に変えたいなら FT（ICL は文脈依存で揮発）
4. **ミッションクリティカル度**: 実験は ICL、本番は FT

**まず many-shot ICL で fine-tuned 同等に届くか先に検証**（long-context で FT を回避できる領域が拡大）してから FT 着手を決める。

## 社内一次回答基盤一般への含意
一次回答系のシステムは grounding + RAG を先に固め、FT は「retrieval は十分なのに generation が頭打ち」が eval で実証されてから。個人情報を重みに焼かない（RAG は取得時アクセス制御で対処、重みに残らない）。蒸留で小型化する場合は教師モデルのライセンス確認が必須ゲート（他社クローズド出力を教師にすると ToS 違反）。

## 関連
- `fine-tuning-vs-rag-vs-grounding.md` — FT/RAG/grounding の使い分けマトリクス
- `four-tradeoffs-cost-angle.md` — コスト軸での FT 位置づけ
- ai/knowledge/transformer/reasoning-model-training-pipeline.md — R1 の RL/GRPO/蒸留（学習リポ側）
