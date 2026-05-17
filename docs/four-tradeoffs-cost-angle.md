# エージェントシステム設計 4 トレードオフ — コスト最適化の極としての fine-tuning リポ

## TL;DR

エージェントシステム設計の 4 軸（性能 / 拡張性 / 信頼性 / コスト）のうち、このリポ（MLX + LoRA で Gemma 3 4B を fine-tune）は**コスト最適化の極**に位置する。**「軽量モデル + ローカルハードウェア」の組み合わせで推論コストを 0 に振り切る選択肢**として、隣接する LLMOps 寄りエージェント実装群と対をなす。

## 4 トレードオフの整理

| 軸 | 内容 | このリポの立ち位置 |
|---|---|---|
| 性能 | スピード vs 精度 | 領域特化で精度を稼ぐ（汎用 7-8B より特化 4B が勝つ局面） |
| 拡張性 | GPU/リソース最適化 | Apple Silicon 統合メモリで「ローカル GPU 0% 遊休」を実現 |
| 信頼性 | ロバスト性 | adapter 切替で fallback 構成、データ漏洩リスクゼロ |
| **コスト** | **性能 vs 費用** | **「軽量モデル + ローカル」で最適化の極** |

## 2026 の収束パターン（コスト軸の最前線）

- **Small Model First** — 失敗のみ大型へ escalate（cascade routing と接続）
- **Prompt Caching 標準化** — Anthropic 90% / Gemini 75% / OpenAI 50%（read 側割引）
- **LLMLingua-2 で 2-5x 圧縮** — latency 1.6-2.9x 改善
- **Batching API** — 50% 追加割引
- **Cost Dashboard 必須化** — Helicone / Langfuse で per-user / per-feature 配賦

これらは「外部 API を呼ぶエージェント側」の最適化。**このリポは別軸で「そもそも外部 API を呼ばない選択肢」を提供する**。

## コスト軸の二択（このリポを置く意義）

| 戦略 | 主役 | コスト構造 | 適用 |
|---|---|---|---|
| **A. 外部基盤モデル + 最適化** | Vertex AI / Bedrock / OpenAI | 推論 token 従量、cache 90% off | 汎用タスク、低頻度、高品質要求 |
| **B. ローカル軽量モデル + fine-tune** | MLX + LoRA + Gemma 3 4B | ハードウェア固定費のみ | 領域特化、高頻度、データ機密 |

A と B はトレードオフではなく **使い分け**。本リポは B の専用実験場として、A のエージェントと組み合わせる時の「fallback / first-pass / 機密データ処理」の引き出しになる。

## このリポの強み（コスト軸で何を押さえているか）

| 観点 | 状態 |
|---|---|
| ローカル推論コスト 0 | ✅ MLX で Apple Silicon ローカル完結 |
| ハードウェア固定費のみ | ✅ M シリーズ統合メモリで GPU 別途調達不要 |
| データ送信ゼロ | ✅ 機密データを外部 API に送らない |
| adapter 切替 | ✅ `adapters-*` ディレクトリ別の version 管理 |
| 学習効率可視化 | ✅ `learning_curve.py` でデータ量別 loss 計測 |
| Base vs FT 比較 | ✅ `eval.py` で出力比較 |

## 他 3 軸のギャップ

### 性能（スピード vs 精度）

- adapter ごとの latency 計測なし
- prompt engineering との性能比較なし（fine-tune を選んだ妥当性を都度検証する仕組みなし）

### 拡張性（リソース最適化）

- 単一 Apple Silicon 上で完結、複数 GPU / マルチノード対応は不要だが「学習中に推論できない」リソース競合あり
- batching / quantization で推論速度を上げる余地

### 信頼性（ロバスト性）

- **adapter 自動評価ハーネス未整備**（現状は手動 5 問採点）
- adapter version registry はディレクトリ命名規則のみ、メタデータ未管理
- regression test なし

## 最適化戦略のハイブリッド設計

このリポと外部 API エージェントを組み合わせる時の理想形:

```
ユーザーリクエスト
  ↓
[ルーター] 機密データを含むか？
  ├ Yes → ローカル MLX + LoRA adapter（このリポ）
  └ No  → cascade routing
            ├ first-pass: small model（Haiku / Flash）
            └ fallback : large model（Sonnet / Opus）
              ↑ Prompt Caching で 90% 割引
```

これで「機密データ処理コスト ≒ 電気代」「汎用処理コスト ≒ token 課金 × cache hit ratio」の二段構えになる。

## 隣接する LLMOps スタック（2026）

agent 側で考慮するスタック。fine-tune 結果を最終的に推論で使う時にこれらと接続:

- **Observability**: OpenTelemetry GenAI Semantic Conventions + Langfuse / Phoenix / Datadog LLM Obs
- **Eval**: τ²-bench の trajectory metrics + Braintrust / Promptfoo / DeepEval / Latitude
- **Prompt Registry**: Braintrust / PromptLayer / Maxim AI / Langfuse Prompt Mgmt
- **Guardrails**: NeMo Guardrails / LlamaGuard 3 8B / Llama Prompt Guard 2 86M の 3 段
- **Routing**: OpenRouter / Martian / Not Diamond（cost-aware）

## 参考

- [Speculative Cascades (Google Research)](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)
- [Prompt Caching Guide 2026](https://tokenmix.ai/blog/prompt-caching-guide)
- [LLMLingua-2](https://llmlingua.com/llmlingua2.html)
- [τ²-bench / AI Agents 2026 architecture](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails)
- [GPU Cost Governance 2026](https://www.digiusher.com/blog/gpu-cost-governance-for-azure-openai-aws-bedrock-and-google-vertex-ai/)
- [Langfuse Token & Cost Tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
