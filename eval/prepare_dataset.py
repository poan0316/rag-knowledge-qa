"""半自动生成评测集：对文档分块，让 LLM 生成 (问题, 标准答案)。

生成后请人工抽查修正，再运行 evaluate.py。
用法：
    python eval/prepare_dataset.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.generation import get_llm
from src.ingestion import load_documents, split_documents

EVAL_DATA = Path(__file__).resolve().parent / "eval_set.json"

QG_PROMPT = """请根据下面的文本片段，生成 1 个可以用它来回答的问题，并给出标准答案。
只输出一行 JSON，不要其他文字：{"question": "问题", "ground_truth": "标准答案"}

文本片段：
{text}
"""


def _parse(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main(limit: int = 30):
    docs = load_documents(config.DOCS_DIR)
    if not docs:
        raise RuntimeError(f"{config.DOCS_DIR} 下没有文档，请先放入 PDF/Word/文本")

    chunks = split_documents(docs)[:limit]
    llm = get_llm(temperature=0.3)
    records = []

    for i, c in enumerate(chunks, 1):
        raw = llm.invoke(QG_PROMPT.format(text=c.page_content)).content
        parsed = _parse(raw)
        if parsed and parsed.get("question") and parsed.get("ground_truth"):
            parsed["contexts"] = [c.page_content]
            records.append(parsed)
            print(f"[{i}/{len(chunks)}] {parsed['question']}")

    with open(EVAL_DATA, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n已生成 {len(records)} 条评测数据 → {EVAL_DATA}")


if __name__ == "__main__":
    main()
