"""ベースモデル vs FTモデルの比較テスト。"""

import json
from pathlib import Path

from mlx_lm import generate, load

MODEL = "mlx-community/gemma-3-4b-it-4bit"
ADAPTER_PATH = "adapters"
TEST_DATA = Path("data/test.jsonl")

SYSTEM_PROMPT = "あなたはAIエンジニアリングの専門家です。技術的に正確で、実務に即した回答をしてください。"


def load_test_data() -> list[dict]:
    """テストデータを読み込む。"""
    data = []
    with open(TEST_DATA, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def run_inference(model, tokenizer, question: str) -> str:
    """モデルで推論を実行する。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=256)


def main():
    test_data = load_test_data()
    print(f"テストデータ: {len(test_data)} サンプル\n")

    # 最大5サンプルで比較
    samples = test_data[: min(5, len(test_data))]

    # ベースモデル
    print("=" * 60)
    print("ベースモデルをロード中...")
    base_model, base_tokenizer = load(MODEL)
    print("ロード完了!\n")

    # FTモデル
    print("FTモデルをロード中...")
    ft_model, ft_tokenizer = load(MODEL, adapter_path=ADAPTER_PATH)
    print("ロード完了!\n")

    for i, sample in enumerate(samples):
        question = sample["messages"][1]["content"]
        expected = sample["messages"][2]["content"][:200]

        print(f"{'=' * 60}")
        print(f"Q{i + 1}: {question}\n")

        base_answer = run_inference(base_model, base_tokenizer, question)
        ft_answer = run_inference(ft_model, ft_tokenizer, question)

        print(f"【ベースモデル】\n{base_answer[:300]}\n")
        print(f"【FTモデル】\n{ft_answer[:300]}\n")
        print(f"【期待する回答（冒頭）】\n{expected}\n")

    print("=" * 60)
    print("比較テスト完了!")


if __name__ == "__main__":
    main()
