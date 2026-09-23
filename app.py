"""Streamlit 入口：可交互的知识库问答界面。

运行：
    streamlit run app.py
"""
import streamlit as st

from src.pipeline import ask

st.set_page_config(page_title="智能知识库问答", page_icon="📚")
st.title("📚 智能知识库问答系统")
st.caption("主线项目 · RAG 问答 | LangChain + Chroma + Ollama/OpenAI")

question = st.text_input("请输入你的问题", placeholder="例如：这份文档讲了什么？")

if question:
    with st.spinner("检索并生成中…"):
        result = ask(question)

    st.markdown("### 回答")
    st.write(result["answer"])

    if result["sources"]:
        st.markdown("### 引用来源")
        for s in result["sources"]:
            loc = s["source"] + (f" · 第 {s['page']} 页" if s.get("page") else "")
            with st.expander(f"[来源 {s['index']}] {loc}"):
                st.write(s["snippet"])
    else:
        st.info("未检索到相关来源。")
