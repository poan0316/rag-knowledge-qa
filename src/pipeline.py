"""RAG 主链路：检索 + 生成。"""
from .generation import generate
from .retrieval import retrieve, retrieve_with_rerank


def ask(question: str, use_rerank: bool = True, top_k: int = None) -> dict:
    """入口：传入问题，返回 {answer, sources}。"""
    if use_rerank:
        docs = retrieve_with_rerank(question, top_k=top_k)
    else:
        docs = retrieve(question, top_k=top_k)

    if not docs:
        return {"answer": "没有检索到相关内容。", "sources": []}
    return generate(question, docs)
