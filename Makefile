.PHONY: help setup prepare train train-dora chat test clean

# === 設定 ===
MODEL ?= mlx-community/gemma-3-4b-it-4bit
DATA_DIR = data
ADAPTER_DIR = adapters
EPOCHS ?= 3
BATCH_SIZE ?= 1
LORA_RANK ?= 8
LEARNING_RATE ?= 1e-5

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 依存関係をインストール
	uv sync

prepare: ## knowledge/ からトレーニングデータを生成
	uv run python prepare_data.py

train: ## LoRA ファインチューニングを実行
	uv run mlx_lm.lora \
		--model $(MODEL) \
		--train \
		--data $(DATA_DIR) \
		--batch-size $(BATCH_SIZE) \
		--lora-rank $(LORA_RANK) \
		--num-layers 8 \
		--learning-rate $(LEARNING_RATE) \
		--epochs $(EPOCHS) \
		--adapter-path $(ADAPTER_DIR)

train-dora: ## DoRA ファインチューニングを実行（LoRA比較用）
	uv run mlx_lm.lora \
		--model $(MODEL) \
		--train \
		--data $(DATA_DIR) \
		--fine-tune-type dora \
		--batch-size $(BATCH_SIZE) \
		--num-layers 4 \
		--learning-rate $(LEARNING_RATE) \
		--iters 100 \
		--max-seq-length 256 \
		--adapter-path adapters-dora \
		--grad-checkpoint

chat: ## FT済みモデルで対話テスト
	uv run python chat.py

test: ## ベースモデル vs FTモデルの比較テスト
	uv run python eval.py

clean: ## アダプターとキャッシュを削除
	rm -rf $(ADAPTER_DIR) __pycache__ .cache
