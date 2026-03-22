"""
複数のナレッジソースからFT用データセットを生成する。

データソース:
- ai/knowledge/ — AIエンジニアリング書籍ナレッジ
- management/docs/ — マネジメント知識ベース

出力形式: JSONL（mlx_lm.lora が期待する chat format）
- train.jsonl (80%)
- valid.jsonl (10%)
- test.jsonl (10%)
"""

import json
import os
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "data"

# データソース定義（ディレクトリ, システムプロンプト, ラベル）
DATA_SOURCES = [
    {
        "dir": BASE_DIR / "ai" / "knowledge",
        "system_prompt": "あなたはAIエンジニアリングの専門家です。技術的に正確で、実務に即した回答をしてください。",
        "label": "ai-engineering",
    },
    {
        "dir": BASE_DIR / "management" / "docs",
        "system_prompt": "あなたはエンジニアリングマネジメントの専門家です。チーム運営、評価、組織設計について、実務に即した回答をしてください。",
        "label": "management",
    },
]

# ナレッジファイルから Q&A ペアを生成するためのテンプレート
QUESTION_TEMPLATES = [
    "{title}について説明してください。",
    "{title}の重要なポイントは何ですか？",
    "{title}を聞かれたらどう答えますか？",
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


def create_qa_pairs(filepath: Path, system_prompt: str, label: str) -> list[dict]:
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

        # 通常のセクション
        for template in random.sample(
            QUESTION_TEMPLATES, min(2, len(QUESTION_TEMPLATES))
        ):
            question = template.format(title=title)
            context = f"（{book}より）\n\n" if book else ""
            pairs.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"{context}{body}"},
                    ]
                }
            )

    return pairs


def main():
    all_pairs = []

    for source in DATA_SOURCES:
        source_dir = source["dir"]
        system_prompt = source["system_prompt"]
        label = source["label"]

        if not source_dir.exists():
            print(f"Warning: {source_dir} が見つかりません。スキップします。")
            continue

        md_files = list(source_dir.rglob("*.md"))
        print(f"\n[{label}] ナレッジファイル数: {len(md_files)}")

        source_count = 0
        for filepath in md_files:
            if filepath.name == "README.md":
                continue
            pairs = create_qa_pairs(filepath, system_prompt, label)
            if pairs:
                print(f"  {filepath.relative_to(source_dir)}: {len(pairs)} ペア")
                source_count += len(pairs)
            all_pairs.extend(pairs)

        print(f"[{label}] 小計: {source_count} ペア")

    print(f"\n合計 Q&A ペア数: {len(all_pairs)}")

    if not all_pairs:
        print("Error: Q&A ペアが生成されませんでした")
        return

    # シャッフル & 分割 (80/10/10)
    random.seed(42)
    random.shuffle(all_pairs)

    write_splits(all_pairs, OUTPUT_DIR / "combined")


def prepare_per_source():
    """データソースごとに個別のデータセットを生成する（モデルマージ用）。"""
    for source in DATA_SOURCES:
        source_dir = source["dir"]
        system_prompt = source["system_prompt"]
        label = source["label"]
        # ラベルからディレクトリ名を生成
        dir_name = label.lower().replace(" ", "-")

        if not source_dir.exists():
            print(f"Warning: {source_dir} が見つかりません。スキップします。")
            continue

        pairs = []
        md_files = list(source_dir.rglob("*.md"))
        print(f"\n[{label}] ナレッジファイル数: {len(md_files)}")

        for filepath in md_files:
            if filepath.name == "README.md":
                continue
            file_pairs = create_qa_pairs(filepath, system_prompt, label)
            if file_pairs:
                print(f"  {filepath.relative_to(source_dir)}: {len(file_pairs)} ペア")
            pairs.extend(file_pairs)

        print(f"[{label}] 合計: {len(pairs)} ペア")

        random.seed(42)
        random.shuffle(pairs)
        write_splits(pairs, OUTPUT_DIR / dir_name)


def write_splits(pairs: list[dict], output_dir: Path):
    """データをtrain/valid/testに分割して書き出す。"""
    n = len(pairs)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train.jsonl": pairs[:train_end],
        "valid.jsonl": pairs[train_end:valid_end],
        "test.jsonl": pairs[valid_end:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in splits.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {output_dir.name}/{filename}: {len(data)} サンプル")

    print(f"データ準備完了! → {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--per-source":
        prepare_per_source()
    else:
        main()
