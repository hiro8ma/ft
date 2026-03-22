"""FT済みモデルで対話テストを行う。"""

from mlx_lm import generate, load

MODEL = "mlx-community/gemma-3-4b-it-4bit"
ADAPTER_PATH = "adapters"

SYSTEM_PROMPT = "あなたはAIエンジニアリングの専門家です。面接で聞かれた質問に対して、実務経験に基づいた具体的な回答をしてください。"


def main():
    print("モデルをロード中...")
    model, tokenizer = load(MODEL, adapter_path=ADAPTER_PATH)
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
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            temp=0.7,
        )
        print(f"\n{response}")


if __name__ == "__main__":
    main()
