"""组装 prompt → 生成 → 引用溯源。"""
import config

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请仅根据下面提供的「上下文」回答问题：
1. 如果上下文里有答案，直接回答并给出引用来源。
2. 如果上下文不足以回答，明确说「根据现有资料无法回答」，不要编造。
3. 回答末尾用 [来源 N] 标注引用的片段编号。"""


def get_llm(temperature: float = 0.1):
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=temperature,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        client_kwargs={"trust_env": False},
    )


def get_eval_llm(temperature: float = 0.0):
    """评测官（judge）专用 LLM：用更大的模型打分，输出 JSON 更稳定。

    生成和评测分开：生成走 LLM_MODEL（7b，快），评测走 EVAL_MODEL（14b，稳）。
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.EVAL_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        client_kwargs={"trust_env": False},
    )


def build_prompt(question: str, docs) -> str:
    context_parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "未知来源")
        page = d.metadata.get("page")
        loc = src + (f"（第 {page + 1} 页）" if page is not None else "")
        context_parts.append(f"[片段 {i}] {loc}\n{d.page_content}")
    context = "\n\n".join(context_parts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}\n"
        f"回答（并用 [来源 N] 标注引用）："
    )


def generate(question: str, docs) -> dict:
    llm = get_llm()
    answer = llm.invoke(build_prompt(question, docs)).content

    sources = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        sources.append(
            {
                "index": i,
                "source": d.metadata.get("source", "未知来源"),
                "page": page + 1 if page is not None else None,
                "snippet": d.page_content[:200],
            }
        )
    return {"answer": answer, "sources": sources}
