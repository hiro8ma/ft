"""
knowledge/ ディレクトリのナレッジファイルからFT用データセットを生成する。

出力形式: JSONL（mlx_lm.lora が期待する chat format）
- train.jsonl (80%)
- valid.jsonl (10%)
- test.jsonl (10%)
"""

import json
import os
import random
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent.parent / "ai" / "knowledge"
OUTPUT_DIR = Path(__file__).parent / "data"

# ナレッジファイルから Q&A ペアを生成するためのテンプレート
QUESTION_TEMPLATES = [
    "{title}について説明してください。",
    "{title}の重要なポイントは何ですか？",
    "{title}を面接で聞かれたらどう答えますか？",
    "{title}の実務への適用方法を教えてください。",
]


def extract_frontmatter(content: str) -> dict:
    """YAML フロントマターを抽出する。"""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    frontmatter = {}
    for line in content[3:end].strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter


def extract_sections(content: str) -> list[dict]:
    """Markdown のセクションを抽出する。"""
    sections = []
    current_title = ""
    current_body = []

    # フロントマターをスキップ
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3 :]

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_title and current_body:
                body = "\n".join(current_body).strip()
                if len(body) > 50:  # 短すぎるセクションはスキップ
                    sections.append({"title": current_title, "body": body})
            current_title = line.lstrip("#").strip()
            current_body = []
        elif current_title:
            current_body.append(line)

    # 最後のセクション
    if current_title and current_body:
        body = "\n".join(current_body).strip()
        if len(body) > 50:
            sections.append({"title": current_title, "body": body})

    return sections


def create_qa_pairs(filepath: Path) -> list[dict]:
    """ナレッジファイルから Q&A ペアを生成する。"""
    content = filepath.read_text(encoding="utf-8")
    frontmatter = extract_frontmatter(content)
    sections = extract_sections(content)

    pairs = []
    book = frontmatter.get("book", "")
    file_title = frontmatter.get("title", filepath.stem)

    for section in sections:
        title = section["title"]
        body = section["body"]

        # 面接で語れるポイントセクションは特別扱い
        if "面接" in title:
            pairs.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたはAIエンジニアリングの専門家です。面接で聞かれた質問に対して、実務経験に基づいた具体的な回答をしてください。",
                        },
                        {
                            "role": "user",
                            "content": f"{file_title}について、面接でどう語りますか？",
                        },
                        {"role": "assistant", "content": body},
                    ]
                }
            )
            continue

        # 通常のセクション
        for template in random.sample(
            QUESTION_TEMPLATES, min(2, len(QUESTION_TEMPLATES))
        ):
            question = template.format(title=title)
            context = f"（{book}より）\n\n" if book else ""
            pairs.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたはAIエンジニアリングの専門家です。技術的に正確で、実務に即した回答をしてください。",
                        },
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"{context}{body}"},
                    ]
                }
            )

    return pairs


def main():
    all_pairs = []

    # knowledge/ 配下の全 .md ファイルを処理
    if not KNOWLEDGE_DIR.exists():
        print(f"Error: {KNOWLEDGE_DIR} が見つかりません")
        return

    md_files = list(KNOWLEDGE_DIR.rglob("*.md"))
    print(f"ナレッジファイル数: {len(md_files)}")

    for filepath in md_files:
        if filepath.name == "README.md":
            continue
        pairs = create_qa_pairs(filepath)
        if pairs:
            print(f"  {filepath.relative_to(KNOWLEDGE_DIR)}: {len(pairs)} ペア")
        all_pairs.extend(pairs)

    print(f"\n合計 Q&A ペア数: {len(all_pairs)}")

    if not all_pairs:
        print("Error: Q&A ペアが生成されませんでした")
        return

    # シャッフル & 分割 (80/10/10)
    random.seed(42)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train.jsonl": all_pairs[:train_end],
        "valid.jsonl": all_pairs[train_end:valid_end],
        "test.jsonl": all_pairs[valid_end:],
    }

    # 出力
    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, data in splits.items():
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {filename}: {len(data)} サンプル")

    print("\nデータ準備完了!")


if __name__ == "__main__":
    main()
