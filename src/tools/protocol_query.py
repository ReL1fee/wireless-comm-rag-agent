"""
知识库检索工具 — Agent 通过此工具调用 RAG 检索通信知识库
"""
from langchain_core.tools import tool
from src.vector_store import load_vector_store, retrieve


@tool
def search_knowledge(query: str) -> str:
    """搜索无线通信知识库，获取协议、技术、原理等详细信息。
    当需要了解某个通信概念的原理、参数、帧结构、工作方式时使用此工具。
    例如：OFDM的原理、5G NR的子载波间隔、MIMO的信道容量公式、循环前缀的作用。
    参数 query 为自然语言查询问题。"""
    try:
        vectorstore = load_vector_store()
    except FileNotFoundError:
        return "错误：知识库尚未构建，请先运行 python main.py --build 构建向量库"

    docs = retrieve(vectorstore, query, k=4)
    if not docs:
        return f"知识库中未找到与 '{query}' 相关的内容，请尝试换个关键词。"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        results.append(f"[来源{i}: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)
