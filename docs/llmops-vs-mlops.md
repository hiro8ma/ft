# MLOps と LLMOps の違い — このリポ（fine-tuning 実験場）の位置づけ

## TL;DR

このリポ（`ft/` = MLX + LoRA で Gemma 3 4B を fine-tune する実験場）は **MLOps 寄り**の活動。隣接する agent 実装リポ群が LLMOps 寄りで、両方を回すことで「基盤モデルを使い倒す層」と「自前で重みを動かす層」の両方を経験できる構成にしている。

## 同じ Ops じゃない

| 観点 | MLOps（このリポの世界） | LLMOps（agent リポ群の世界） |
|---|---|---|
| 主役 | 自前で訓練したモデル / adapter | 他社の基盤モデル（Claude / GPT / Gemini） |
| 改善の単位 | データ + 重み（再学習が必要） | prompt + コンテキスト + tool（再デプロイなし） |
| バージョン管理 | データセット + モデル重み + ハイパラ | prompt + RAG index + tool schema + model ID |
| デプロイ | adapter を serving に乗せる | prompt を更新、API キー切り替え、registry の version label 切替 |
| コスト構造 | 学習 GPU が支配、推論は安い | 推論 token が支配、学習は基本発生しない |
| 評価 | accuracy / F1 / loss curve（数値で一意） | LLM-as-Judge / rubric / human eval（多次元・揺らぎあり） |
| 失敗モード | overfitting / data drift | hallucination / prompt injection / context rot |
| 改善サイクル | 週〜月（再学習回す） | 時間〜日（prompt diff） |

## 2026 時点での整理

- LLM 時代になって MLOps が消えたわけではなく、「層が違う」整理が主流
- 基盤モデル（外部 API）の上で agentic system を作り、必要な部分だけ自前 fine-tune する hybrid が普通
- このリポは hybrid 構成のうち「自前 fine-tune」レイヤを担う

## このリポで MLOps として効いている要素

- `learning_curve.py` — データ量別の loss を取って学習効率を可視化
- `eval.py` — base モデルと FT 後 adapter の出力比較
- `adapters-*` ディレクトリ別の version 管理（手動だが registry 的）
- `learning_curve_results.json` — 実験結果の永続化

## まだ薄い・足したい要素

- adapter の自動評価 harness（現状は手動 5 問採点）
- adapter version registry（命名規則だけ、メタデータ未管理）
- agent 側 eval との接続（fine-tune した adapter を agent で使った時の品質を統一指標で見る）

## 隣接する LLMOps スタック（2026）

参考までに、agent 側で考慮するスタックを記録しておく。fine-tune 結果を最終的に推論で使うとき、これらと接続する。

- **Observability**: OpenTelemetry GenAI Semantic Conventions（事実上の標準）+ Langfuse / Phoenix / Datadog LLM Obs
- **Eval**: Braintrust / Promptfoo / DeepEval / Latitude（LLM-as-Judge は pairwise + cross-model judge で運用）
- **Prompt Registry**: Braintrust / PromptLayer / Maxim AI / Langfuse Prompt Mgmt
- **Gateway**: LiteLLM / Portkey / Helicone Gateway（cache / cost cap / multi-provider failover）
- **Caching**: OpenAI / Anthropic / Gemini いずれも cached input 約 90% 割引が標準

## 参考

- ZenML — MLOps vs LLMOps: https://www.zenml.io/blog/mlops-vs-llmops
- Jozu — AIOps vs DevOps vs MLOps vs LLMOps: https://jozu.com/blog/aiops-devops-mlops-llmops-whats-the-difference/
- OpenTelemetry GenAI Semantic Conventions: https://dev.to/x4nent/opentelemetry-genai-semantic-conventions-the-standard-for-llm-observability-1o2a
