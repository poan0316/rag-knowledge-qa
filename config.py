"""集中配置：从 .env 读取，缺省给「本地 Ollama 零成本」方案。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------- LLM ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | openai

# Ollama（本地免费）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EVAL_MODEL = os.getenv("EVAL_MODEL", "qwen2.5:14b")  # 评测官模型，独立于生成模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# OpenAI（可选，云端付费）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ---------- 向量库 ----------
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ai_interview_kb")

# ---------- RAG 参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))

# ---------- 数据目录 ----------
DOCS_DIR = BASE_DIR / "data/docs"
