"""CLI 入库：把 data/docs 下的文档写入向量库。

用法：
    python scripts/run_ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import build_vectorstore

if __name__ == "__main__":
    build_vectorstore()
