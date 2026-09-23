"""Agent 工具：搜索（本地知识库检索）+ 文件读写。

说明：这里用普通函数实现工具，节点内确定性调用（不依赖 7b 的 function-calling，
更稳定）。若将来换支持 function-calling 的模型，可给它们套 `@tool` 装饰器交给 LLM 自动调用。
"""
import warnings
from pathlib import Path

from src.retrieval import retrieve_with_rerank

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _retrieve(query: str, top_k: int):
    with warnings.catch_warnings():
        # bge-m3 的余弦分数可能超出 [0,1]，触发的 UserWarning 属正常现象，屏蔽掉
        warnings.filterwarnings("ignore", message=".*Relevance scores.*")
        return retrieve_with_rerank(query, top_k=top_k)


def _format_chunks(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = Path(d.metadata.get("source", "未知来源")).name
        parts.append(f"[来源 {i}]（{src}）\n{d.page_content}")
    return "\n\n".join(parts)


def search(query: str, top_k: int = 4) -> str:
    """在本地知识库检索与 query 最相关的文档片段，返回带 [来源 N] 编号的文本。"""
    docs = _retrieve(query, top_k)
    if not docs:
        return "（知识库中没有检索到相关内容）"
    return _format_chunks(docs)


def search_with_sources(query: str, top_k: int = 4):
    """检索并同时返回 (文本, 来源文件名列表)，供定稿时生成准确来源清单。"""
    docs = _retrieve(query, top_k)
    if not docs:
        return "（知识库中没有检索到相关内容）", []
    filenames = [Path(d.metadata.get("source", "未知来源")).name for d in docs]
    return _format_chunks(docs), filenames


def save_report(text: str, filename: str = "report.md") -> str:
    """把报告文本写入文件，返回保存路径。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    path.write_text(text, encoding="utf-8")
    return str(path)
