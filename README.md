# 无线通信知识智能助手

基于 **RAG + Agent** 的无线通信协议智能问答系统，支持上传 PDF/Word 扩展知识库。

## 技术栈

Python / LangChain / 通义千问 / ChromaDB / Streamlit

## 功能

- **知识库检索**：RAG 检索增强生成，回答基于知识库内容并标注来源
- **公式计算**：香农容量、误码率、路径损耗等通信公式
- **概念对比**：自动对比两个通信技术的异同
- **文件上传**：支持上传 PDF/Word/Markdown 扩展知识库

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env  # 编辑 .env 填入通义千问 API Key

# 3. 构建知识库
python main.py --build

# 4. 启动
python main.py
```

浏览器打开 `http://localhost:8501`，在侧边栏加载知识库后即可提问。

## 示例问题

- OFDM 中循环前缀有什么作用？
- 计算 20MHz 带宽、10dB 信噪比下的信道容量
- 对比 LDPC 码和 Polar 码

## 项目结构

```
├── data/               # 知识库文件
├── src/
│   ├── agent.py        # Agent 调度核心
│   ├── vector_store.py # 向量存储与检索
│   ├── rag_chain.py    # RAG 问答链
│   ├── document_loader.py  # 文档加载（MD/PDF/Word）
│   └── tools/          # Agent 工具集
└── main.py             # 入口
```
