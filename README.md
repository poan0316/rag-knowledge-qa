# 📚 智能知识库问答系统（RAG）

> 主线项目：基于 **LangChain + Chroma + Ollama/OpenAI** 的私有文档问答系统，支持 PDF/Word/文本解析、分块、向量检索、引用溯源与 Ragas 自动评测。

---

## 架构

```
用户 ──▶ Streamlit 前端
              │
              ▼
    ┌─────────────────────────────┐
    │  编排层：LangChain          │
    │  · 文档解析（PDF/Word）     │
    │  · 分块 chunking            │
    │  · Embedding 向量化         │
    │  · 检索 + 重排 + 生成       │
    │  · 引用溯源                 │
    └──────────────┬──────────────┘
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
  向量库 Chroma            LLM（Ollama 本地 / OpenAI API）
                              │
                              ▼
                         Ragas 评测层
```

## 目录结构

```
knowledge-qa/
├── app.py                  # Streamlit 交互界面
├── config.py               # 集中配置（读 .env）
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion.py        # 解析 / 分块 / 入库
│   ├── retrieval.py        # 向量检索 + 轻量重排
│   ├── generation.py       # prompt 组装 + 生成 + 引用溯源
│   └── pipeline.py         # RAG 主链路 ask()
├── scripts/
│   └── run_ingest.py       # CLI 入库
├── eval/
│   ├── prepare_dataset.py  # 半自动生成评测集
│   ├── evaluate.py         # Ragas 评测
│   └── eval_set.json       # 评测数据
└── data/
    └── docs/               # 放你的 PDF/Word/文本
```

## 快速开始

### 1. 环境准备

> ⚠️ 建议使用 **Python 3.11 或 3.12**（3.14 部分依赖可能没有 wheel）。

```bash
cd knowledge-qa
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # Windows 用 copy .env.example .env
```

### 2. 跑通本地模型（Ollama，零成本）

安装 [Ollama](https://ollama.com/) 后拉取模型：

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

> 16G 内存跑 7B 可以；跑不动可换成更小的 `qwen2.5:3b` 或 `phi3:mini`，改 `.env` 里的 `LLM_MODEL` 即可。

### 3. 放文档 + 入库

把你的 PDF / Word / 文本放进 `data/docs/`，然后：

```bash
python scripts/run_ingest.py
```

### 4. 启动问答

```bash
streamlit run app.py
```

浏览器打开后输入问题即可，回答会带 `[来源 N]` 引用。

### 5. 评测（可选，加分项）

```bash
# 半自动生成评测集（对文档分块生成 问题+标准答案）
python eval/prepare_dataset.py
# 人工抽查修正 eval/eval_set.json 后运行评测
python eval/evaluate.py
```

输出忠实度（faithfulness）与答案相关性（answer relevancy）指标。

---

## 可选：切换到 OpenAI API

编辑 `.env`：

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
```

即可用 `gpt-4o-mini` + `text-embedding-3-small`（需要付费）。

---

## 进阶项目：多智能体报告生成（LangGraph）

让「研究员 → 写手 → 审校」三个 Agent 协作：输入一个主题，自动检索本地知识库，产出一份带来源标注的 Markdown 报告。

```
[研究员] 检索+简报 ──▶ [写手] 撰写报告 ──▶ [审校] 检查 ──(通过)──▶ 定稿
                            ▲                    │
                            └───(需修改，最多重写2次)──┘
```

### 运行

```bash
# 需用安装了 langgraph 的 Python 环境（本项目为 eval-venv，Python 3.12）
python agents/main.py "RAG 是什么"
```

报告保存到 `agents/reports/report.md`。

### 目录

```
agents/
├── main.py        # CLI 入口
├── graph.py       # LangGraph 工作流编排（含条件循环边）
├── prompts.py     # 三个角色 prompt + 状态定义
└── tools.py       # 工具：search（复用 RAG 检索）+ save_report
```

面试可聊的点：角色分工、LangGraph 状态机与条件路由、Agent 间通信、失败重试（审校回退重写）、成本控制（限制重写次数）、工具调用。

---

## 可调参数（.env）

| 参数 | 默认 | 说明 |
|------|------|------|
| `CHUNK_SIZE` | 500 | 分块大小 |
| `CHUNK_OVERLAP` | 50 | 分块重叠 |
| `TOP_K` | 4 | 检索返回条数 |

面试可聊的优化方向：分块策略、embedding 选型、top-k 调参、重排（当前是关键词弱重排，可换 cross-encoder）、引用溯源、如何抑制幻觉、评测闭环。
