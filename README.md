# Fine-tuning LLMs on Apple Silicon

MLX + LoRA/QLoRA を使った、Apple Silicon ローカルでのファインチューニング実験。

## 環境

- MacBook Pro M2 Max / 64GB
- MLX (Apple 純正 ML フレームワーク)
- ベースモデル: Gemma 3 4B IT 4bit

## セットアップ

```bash
make setup
```

## 使い方

```bash
# 1. knowledge/ からトレーニングデータを生成
make prepare

# 2. LoRA ファインチューニング実行
make train

# 3. FT済みモデルで対話テスト
make chat

# 4. ベースモデル vs FTモデルの比較
make test
```

## ディレクトリ構成

```
ft/
├── prepare_data.py    # データ準備（knowledge/ → JSONL）
├── chat.py            # 対話テスト
├── eval.py            # ベース vs FT 比較評価
├── data/              # 生成されたトレーニングデータ
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── adapters/          # LoRA アダプター（FT結果）
├── pyproject.toml
└── Makefile
```
