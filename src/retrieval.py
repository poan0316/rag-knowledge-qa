"""检索：top-k 相似度检索 + 可选轻量重排。"""
from langchain_chroma import Chroma

import config
from .ingestion import get_embeddings


def get_vectorstore():
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
    )


def retrieve(query: str, top_k: int = None):
    """纯向量相似度检索，返回 top_k 个文档分块。"""
    top_k = top_k or config.TOP_K
    return get_vectorstore().similarity_search(query, k=top_k)


def _keyword_score(query: str, text: str) -> float:
    """极简关键词重合度，作为重排的弱信号（可替换为 cross-encoder）。"""
    q_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & text_words) / len(q_words)


def retrieve_with_rerank(query: str, top_k: int = None, fetch_k: int = None):
    """先取 fetch_k 个候选，再按 0.7*向量分 + 0.3*关键词重合 重排取 top_k。"""
    top_k = top_k or config.TOP_K
    fetch_k = fetch_k or top_k * 3
    vs = get_vectorstore()
    scored = vs.similarity_search_with_relevance_scores(query, k=fetch_k)
    reranked = sorted(
        scored,
        key=lambda t: t[1] * 0.7 + _keyword_score(query, t[0].page_content) * 0.3,
        reverse=True,
    )
    return [doc for doc, _ in reranked[:top_k]]
