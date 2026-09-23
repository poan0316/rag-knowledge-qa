"""Ragas 评测：忠实度 + 答案相关性。

注意：ragas 不同版本 API 差异较大（本工程锁 0.2.x）。
若导入报错，请先 `pip install -r requirements.txt` 确认 ragas 版本。

用法：
    python eval/evaluate.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from src.generation import get_eval_llm
from src.ingestion import get_embeddings
from src.pipeline import ask

EVAL_DATA = Path(__file__).resolve().parent / "eval_set.json"
REPORT = Path(__file__).resolve().parent / "eval_report.csv"


def load_eval_set():
    with open(EVAL_DATA, encoding="utf-8") as f:
        return json.load(f)


def build_samples(records):
    samples = []
    for r in records:
        result = ask(r["question"], use_rerank=True)
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r.get("contexts", []),
                response=result["answer"],
                reference=r.get("ground_truth", ""),
            )
        )
    return samples


def main():
    records = load_eval_set()

    # 冒烟测试：MAX_SAMPLES>0 时只跑前 N 条，先验证评测官能出分，再跑全量
    max_samples = int(os.getenv("MAX_SAMPLES", "0") or "0")
    if max_samples > 0:
        records = records[:max_samples]
        print(f"[冒烟测试] 只评测前 {max_samples} 条")

    samples = build_samples(records)
    dataset = EvaluationDataset(samples=samples)

    llm = LangchainLLMWrapper(get_eval_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm),
    ]

    # max_workers=1：串行跑。本地 Ollama 是串行处理请求，若并发过高会排队超时，
    # 导致 faithfulness 全 NaN（此前 16 路并发就是这个坑）。
    result = evaluate(
        dataset,
        metrics=metrics,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1),
    )
    df = result.to_pandas()
    print(df)
    df.to_csv(REPORT, index=False, encoding="utf-8-sig")
    print(f"\n评测报告已保存 → {REPORT}")


if __name__ == "__main__":
    main()
