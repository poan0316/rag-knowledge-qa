"""文档解析 → 分块 → embedding → 写入向量库。"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def get_embeddings():
    """根据 LLM_PROVIDER 返回对应的 embedding 模型。"""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        client_kwargs={"trust_env": False},
    )


def load_document(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix == ".docx":
        return Docx2txtLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"暂不支持的格式：{suffix}（支持 {SUPPORTED_SUFFIXES}）")


def load_documents(docs_dir: Path):
    docs = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                docs.extend(load_document(p))
            except Exception as e:  # 单文档失败不阻断整体
                print(f"[跳过] {p.name}: {e}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs_dir: Path = None):
    """解析 docs_dir 下所有文档并写入向量库。"""
    docs_dir = docs_dir or config.DOCS_DIR
    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(f"{docs_dir} 下没有找到可解析的文档（{SUPPORTED_SUFFIXES}）")

    chunks = split_documents(docs)
    print(f"解析到 {len(docs)} 篇文档，切成 {len(chunks)} 个分块")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
    )
    print(f"已写入向量库：{config.CHROMA_DIR}")
    return vs
