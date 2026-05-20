"""
RAG 问答模块
组合检索与 LLM 生成，实现带引用的智能问答
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL, RETRIEVAL_K
)
from src.vector_store import retrieve


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """获取通义千问 LLM"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "请设置 DASHSCOPE_API_KEY 环境变量。\n"
            "复制 .env.example 为 .env 并填入你的 API Key"
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=temperature,
    )


RAG_SYSTEM_PROMPT = """你是一个专业的无线通信技术知识助手。请根据以下检索到的知识库内容回答用户的问题。

要求：
1. 基于知识库内容回答，并标注信息来源
2. 回答要专业但易懂，适当使用公式和例子
3. 如果知识库内容不足以回答问题，可以结合你的专业知识补充，但要说明"""


def ask_rag(vectorstore, question: str) -> dict:
    """RAG 问答：检索 + 生成"""
    # 1. 检索相关文档
    docs = retrieve(vectorstore, question)

    # 2. 构建上下文
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        context_parts.append(f"[来源{i}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. 调用 LLM 生成回答
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "知识库内容：\n{context}\n\n用户问题：{question}"),
    ])
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "contexts": docs,
    }
