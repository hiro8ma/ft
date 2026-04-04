"""Qwen3-0.6B ベースモデルで対話テストを行う（Reasoning モード対応）。"""

import re

from mlx_lm import generate, load

MODEL = "mlx-community/Qwen3-0.6B-4bit"

SYSTEM_PROMPT = "あなたはAIエンジニアリングの専門家です。面接で聞かれた質問に対して、実務経験に基づいた具体的な回答をしてください。"


def split_thinking(text: str) -> tuple[str, str]:
    """思考トークン(<think>...</think>)と最終回答を分離する。"""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text[match.end() :].strip()
        return thinking, answer
    return "", text.strip()


def main():
    print("モデルをロード中...")
    model, tokenizer = load(MODEL)
    print("ロード完了!\n")
    print("質問を入力してください（'quit' で終了）:")

    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=8192,
        )

        thinking, answer = split_thinking(response)

        if thinking:
            print(f"\n💭 思考:\n{thinking}")
            print(f"\n📝 回答:\n{answer}")
        else:
            print(f"\n{answer}")


if __name__ == "__main__":
    main()
