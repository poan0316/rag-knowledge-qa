"""多智能体报告生成 CLI 入口。

用法：
    python agents/main.py "RAG 是什么"
"""
import sys
from pathlib import Path

# 把 knowledge-qa 根目录加入 sys.path，使 config、src.*、agents.* 可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph import build_graph
from agents.tools import save_report


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else input("请输入报告主题：").strip()
    if not topic:
        print("主题不能为空")
        return

    graph = build_graph()
    print(f"[研究员] 正在围绕「{topic}」搜集资料……\n")
    result = graph.invoke(
        {
            "topic": topic,
            "research_brief": "",
            "sources": [],
            "draft": "",
            "feedback": "",
            "verdict": "",
            "iteration": 0,
            "final_report": "",
        }
    )

    report = result["final_report"]
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50 + "\n")

    path = save_report(report)
    print(f"报告已保存 → {path}")


if __name__ == "__main__":
    main()
