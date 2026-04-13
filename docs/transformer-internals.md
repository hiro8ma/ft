---
title: "Transformer デコーダの内部構造と推論最適化"
source: "Software Design 2026年4月号"
date: "2026-04-13"
tags: [transformer, decoder, vLLM, PagedAttention, tokenization, Qwen3]
---

# Transformer デコーダの内部構造と推論最適化

## 1. Qwen3-0.6B のアーキテクチャ

Qwen3-0.6B は 0.6B パラメータの小型デコーダモデル。内部構造を理解することで、LoRA FT 時にどの層が何をしているかを把握できる。

### レイヤー構成

```
Embedding (151,936 tokens → 1,024次元)
  ↓
DecoderLayer × 28層
  ├── RMSNorm (Pre-Norm)
  ├── Self-Attention (16 heads, 64次元/head)
  │   └── RotaryEmbedding (RoPE: 位置情報をベクトル回転で注入)
  ├── RMSNorm
  └── MLP (1,024 → 3,072 → 1,024, SiLU活性化)
  ↓
RMSNorm (Final)
  ↓
lm_head (1,024 → 151,936: 次トークン確率)
```

### 各コンポーネントの役割

| コンポーネント | 役割 | 補足 |
|---|---|---|
| **Embedding** | トークンID → 1,024次元ベクトルに変換 | 語彙数 151,936 は多言語対応のため大きい |
| **RMSNorm** | LayerNorm の簡易版。平均を引かず二乗平均のみで正規化 | 計算効率が良い。Pre-Norm 構造（Attention/MLP の前に適用） |
| **RotaryEmbedding (RoPE)** | Query/Key ベクトルを位置に応じて回転させる | 相対位置関係を内積で自然に表現。外挿性能が高い |
| **Self-Attention** | 16ヘッド × 64次元。各ヘッドが異なる関係性パターンを学習 | GQA (Grouped Query Attention) でKV共有による効率化 |
| **MLP** | Gate + Up → SiLU → Down の3層構造 (1,024→3,072→1,024) | SwiGLU 活性化。知識の記憶・変換を担う |
| **lm_head** | 最終層の1,024次元 → 151,936語彙の確率分布に変換 | Embedding の重みを共有（weight tying）する場合もある |

### LoRA FT との接続

LoRA は主に Self-Attention の Q/V プロジェクション行列に低ランク行列を挿入する。28層 × (Q + V) = 56箇所のアダプタが追加される。MLP にも適用可能だが、パラメータ数とのトレードオフがある。

## 2. Tokenization の仕組み

### apply_chat_template

チャットモデルでは、ユーザー入力をそのまま渡すのではなく、モデル固有のテンプレートで整形する。

```python
messages = [{"role": "user", "content": "こんにちは"}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # アシスタント応答の開始トークンを付与
    enable_thinking=True,        # Qwen3: <think>トークンを有効化
)
```

- **add_generation_prompt=True**: `<|im_start|>assistant\n` のようなプレフィックスを末尾に追加。これがないとモデルは応答を開始しない
- **enable_thinking=True**: Qwen3 固有。`<think>...</think>` で思考過程を出力してから回答する Reasoning モード

### attention_mask

```python
attention_mask = [1, 1, 1, 1, 1, 0, 0]  # 1=有効, 0=パディング
```

バッチ処理時、系列長が異なる入力をパディングで揃える。`attention_mask` で「どのトークンに注意を向けるか」を制御する。パディング部分は 0 にして Attention 計算から除外する。

## 3. vLLM の PagedAttention

推論時のボトルネックは KV キャッシュのメモリ管理。vLLM はこれを OS の仮想メモリ管理と同じ発想で解決した。

### 従来の問題

- KV キャッシュは系列長に比例してメモリを消費する
- 事前に最大系列長分のメモリを確保する必要がある → メモリの無駄遣い
- 複数リクエストの同時処理でメモリ断片化が発生

### PagedAttention の仕組み

```
従来: [=====KV Cache=====][    未使用領域    ]  ← 最大長分を確保
                                                  （実際は半分しか使わない）

PagedAttention:
  Page Table: [Page0] → Block A
              [Page1] → Block D    ← 必要な分だけ物理ブロックを割当
              [Page2] → Block F
```

- KV キャッシュを固定サイズの「ブロック」に分割
- OS のページテーブルと同様に、論理ページ → 物理ブロックのマッピングで管理
- 連続したメモリ確保が不要 → メモリ断片化を解消
- 同じプロンプトの KV キャッシュをリクエスト間で共有可能（Copy-on-Write）

### Continuous Batching

従来のバッチ処理では、バッチ内の全リクエストが終了するまで次のバッチを処理できなかった。

```
従来の Static Batching:
  Request A: [============]
  Request B: [====]          ← B が終わっても A を待つ
  Request C: [========]      ← C も A を待つ
  → スループットが低い

Continuous Batching:
  Request A: [============]
  Request B: [====][D: new]  ← B 完了後すぐに D を開始
  Request C: [========][E:]  ← C 完了後すぐに E を開始
  → GPU 稼働率を最大化
```

- リクエストが完了するたびに新しいリクエストをバッチに追加
- GPU の空き時間を最小化し、スループットを向上

## 4. 継続事前学習 vs LoRA FT の使い分け

ファインチューニングには大きく2つのアプローチがある。目的に応じて使い分ける。

### 比較表

| 項目 | 継続事前学習 (CPT) | LoRA FT (SFT) |
|---|---|---|
| **目的** | ドメイン知識の注入 | 特定タスクへの適応 |
| **更新範囲** | フルパラメータ | 低ランク行列のみ（全体の0.1-1%程度） |
| **データ形式** | 大量の生テキスト（数GB〜） | 指示-応答ペア（数百〜数千件） |
| **計算コスト** | 高い（GPU クラスタ必要） | 低い（単一GPU / Apple Silicon で可能） |
| **破壊的忘却** | リスク高い | リスク低い（元の重みは凍結） |
| **典型的ユースケース** | 医療・法律・金融のドメイン特化 | Q&A、要約、分類などのタスク特化 |

### 選択フロー

```
モデルがドメイン知識を持っていない？
  → Yes: 継続事前学習でドメイン知識を注入
    → その後、LoRA FT でタスク適応
  → No: LoRA FT のみでタスク適応
```

実務では「継続事前学習 → LoRA FT」のパイプラインが多い。まずドメイン知識を入れ、次にタスク形式に合わせる。

## 5. ft/ リポでの実践との接続

このリポジトリでは LoRA FT を Apple Silicon ローカルで実践している。

### 実装済みの内容

| 項目 | 実装 | 対応する理論 |
|---|---|---|
| **モデル** | Qwen3-0.6B (4bit量子化) | 上記アーキテクチャの実物 |
| **FT手法** | MLX + LoRA | Self-Attention の Q/V に低ランク行列を挿入 |
| **データ** | AIエンジニアリング面接Q&A | SFT 用の指示-応答ペア |
| **推論** | chat_qwen3.py (enable_thinking対応) | apply_chat_template + Reasoning モード |
| **評価** | eval.py + learning_curve.py | データ量 vs 損失の関係を検証 |
| **手法比較** | LoRA vs DoRA (Makefile) | PEFT 手法の実践的比較 |

### 今後の拡張余地

- **vLLM での推論サーバー構築**: PagedAttention + Continuous Batching で推論スループットを向上（ただし Apple Silicon では vLLM 非対応のため、Linux 環境が必要）
- **継続事前学習の実験**: ドメイン特化テキストでの CPT → LoRA FT パイプラインの検証
- **KV キャッシュの可視化**: 推論時のメモリ使用量をプロファイリングし、PagedAttention の効果を定量的に確認
