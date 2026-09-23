"""LangGraph 多智能体工作流：研究员 → 写手 → 审校（可回退重写）→ 定稿。"""
from typing import Literal

from langgraph.graph import END, StateGraph

from src.generation import get_llm
from .prompts import AgentState, RESEARCHER_PROMPT, REVIEWER_PROMPT, WRITER_PROMPT
from .tools import search_with_sources

MAX_ITERATIONS = 2  # 最多重写次数，避免死循环（成本控制）


def researcher(state: AgentState) -> dict:
    """研究员：检索资料 → LLM 汇总成研究简报，并记录来源文件名。"""
    topic = state["topic"]
    raw, sources = search_with_sources(topic)
    brief = get_llm(0.1).invoke(
        RESEARCHER_PROMPT.format(topic=topic, research=raw)
    ).content
    return {"research_brief": brief, "sources": sources}


def writer(state: AgentState) -> dict:
    """写手：根据研究简报（+ 上一轮审校意见）撰写报告。"""
    feedback = state.get("feedback", "")
    iteration = state.get("iteration", 0)
    if feedback:
        iteration += 1  # 有审校意见说明这是重写轮

    extra = f"\n\n【上一轮审校意见，请据此修改】\n{feedback}" if feedback else ""
    draft = get_llm(0.3).invoke(
        WRITER_PROMPT.format(topic=state["topic"], brief=state["research_brief"]) + extra
    ).content
    return {"draft": draft, "iteration": iteration}


def reviewer(state: AgentState) -> dict:
    """审校：检查报告，输出修改建议 + 结论。"""
    out = get_llm(0.0).invoke(
        REVIEWER_PROMPT.format(draft=state["draft"], brief=state["research_brief"])
    ).content
    return {"feedback": out, "verdict": _parse_verdict(out)}


def finalize(state: AgentState) -> dict:
    """定稿：去掉写手可能自写的「来源」节，统一附加准确的来源清单。"""
    draft = state["draft"].rstrip()
    if "## 来源" in draft:
        draft = draft.split("## 来源")[0].rstrip()

    sources = state.get("sources", [])
    unique = list(dict.fromkeys(sources))  # 去重保序
    src_section = (
        "\n\n## 来源\n" + "\n".join(f"{i}. {name}" for i, name in enumerate(unique, 1))
        if unique
        else ""
    )
    return {"final_report": draft + src_section}


def _parse_verdict(text: str) -> str:
    """从审校输出里解析结论：找「结论」行，含「修改」→ 需修改，否则通过。"""
    for line in reversed(text.strip().splitlines()):
        if "结论" in line:
            return "需修改" if "修改" in line else "通过"
    return "通过"  # 解析失败默认通过，保证不卡死


def route_after_review(state: AgentState) -> Literal["writer", "finalize"]:
    """审校后的条件路由：需修改且未超次数 → 回写手，否则 → 定稿。"""
    if state.get("verdict") == "需修改" and state.get("iteration", 0) < MAX_ITERATIONS:
        return "writer"
    return "finalize"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("reviewer", reviewer)
    g.add_node("finalize", finalize)

    g.set_entry_point("researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()
