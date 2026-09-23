# 环境安装引导（Windows）

> 目标：在 Windows 11 上跑通「智能知识库问答系统」。全程零成本，用本地 Ollama 模型。

---

## 0. 先做决策：Python 版本

你本机目前是 **Python 3.14.4**。chromadb / langchain / pydantic 等库对 3.14 的 wheel 支持可能不全，安装时容易报「找不到匹配版本」或编译失败。

**建议：装一个 Python 3.12（或 3.11）专门跑这个项目。**

先看看本机已经有哪些 Python 版本：

```bash
py -0p
```

如果列表里已经有 `3.12` 或 `3.11`，直接跳到第 2 步；没有就按第 1 步装。

---

## 1. 安装 Python 3.12

任选一种方式：

**方式 A：winget（推荐，最省事）**
```bash
winget install Python.Python.3.12
```
装完**重开一个终端**再继续。

**方式 B：官网下载**
打开 https://www.python.org/downloads/release/python-3120/ 下载 Windows installer（64-bit），安装时**勾选「Add python.exe to PATH」**。

**方式 C：已装 conda / miniconda**
```bash
conda create -n rag python=3.12 -y
conda activate rag
```
（用 conda 的话下面「创建 venv」一步可跳过，直接用这个环境。）

---

## 2. 创建并激活虚拟环境

进入项目目录（Git Bash）：

```bash
cd D:/rag检索/knowledge-qa
```

用 **py 启动器指定 3.12** 建虚拟环境：

```bash
py -3.12 -m venv .venv
```

激活（**注意你用的是哪个终端**）：

```bash
# Git Bash（你现在这个）
source .venv/Scripts/activate

# 或 PowerShell
.venv\Scripts\Activate.ps1

# 或 cmd
.venv\Scripts\activate.bat
```

激活成功后，命令行前面会出现 `(.venv)` 前缀。

---

## 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 这一步最花时间（chromadb、langchain 等比较大）。如果报「Building wheel ... error」或「Could not find a version」，多半是 Python 版本问题，回到第 0/1 步用 3.12 重试。

---

## 4. 安装 Ollama（本地模型运行时）

**方式 A：winget**
```bash
winget install Ollama.Ollama
```

**方式 B：官网**
打开 https://ollama.com/download/windows 下载安装包。

装完 **启动 Ollama 应用**（任务栏会出现羊驼图标），它是后台服务，监听 `http://localhost:11434`。

---

## 5. 拉取模型（两个：一个生成、一个 embedding）

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

验证是否成功：

```bash
ollama list
```

> `nomic-embed-text` 是 embedding 模型，入库必须用它把文档向量化。
> 如果 7B 太大/太慢，可换小的：`ollama pull qwen2.5:3b`，并把 `.env` 里的 `LLM_MODEL` 改成 `qwen2.5:3b`。

---

## 6. 配置 .env

```bash
cp .env.example .env
```

默认就是本地 Ollama 方案（`LLM_PROVIDER=ollama`），**不用改、不用 API Key**。除非你想用 OpenAI，才去填 `OPENAI_API_KEY` 并改 `LLM_PROVIDER=openai`。

---

## 7. 放文档 + 入库 + 跑起来

1. 把你的 **PDF / Word / txt / md** 文件放进 `data/docs/` 目录。
2. 入库：
   ```bash
   python scripts/run_ingest.py
   ```
   看到「解析到 X 篇文档，切成 Y 个分块」「已写入向量库」即成功。
3. 启动界面：
   ```bash
   streamlit run app.py
   ```
   浏览器会自动打开 `http://localhost:8501`，输入问题即可。

---

## 8. 常见报错排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `pip install` 报 `Could not find a version` / `Building wheel ... error` | Python 3.14 缺 wheel | 换 Python 3.12 重建 venv |
| `Microsoft Visual C++ 14.0 or greater is required` | 某包需要本地编译 | 换 3.12 用预编译 wheel，或装 VS Build Tools |
| 连接 `localhost:11434` 被拒绝 / `Connection error` | Ollama 没启动 | 打开 Ollama 应用，`ollama list` 能出结果再跑 |
| 模型加载后卡死 / 内存爆 | 7B 模型内存不够 | 换 `qwen2.5:3b`，改 `.env` |
| `streamlit: command not found` | venv 没激活 | 用 `python -m streamlit run app.py` 或重新 `source .venv/Scripts/activate` |
| 入库时出现 `[跳过] xxx.pdf` | 单个文件解析失败 | 看具体报错；换个文件/格式试，不影响其它文档 |
| `ModuleNotFoundError: rag...` 或 `langchain...` | 依赖没装全 | 回到第 3 步重装 |

---

## 快速自检清单

- [ ] `python --version` 是 3.12（或 3.11）
- [ ] 命令行有 `(.venv)` 前缀
- [ ] `ollama list` 能列出 `qwen2.5:7b` 和 `nomic-embed-text`
- [ ] `.env` 已存在（`cp .env.example .env`）
- [ ] `data/docs/` 里放了至少一个文档
- [ ] `python scripts/run_ingest.py` 显示「已写入向量库」
- [ ] `streamlit run app.py` 能打开网页并回答问题
