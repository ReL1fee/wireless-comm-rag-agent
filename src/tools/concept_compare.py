"""
概念对比工具 — 基于知识库对比两个通信概念的异同
"""
from langchain_core.tools import tool
from src.rag_chain import get_llm
from src.vector_store import load_vector_store, retrieve


COMPARE_PROMPT = """你是一个无线通信技术专家。请根据知识库内容对比以下两个概念/技术。

知识库相关内容：
{context}

请从定义、原理、应用场景、优缺点等方面对比：
- 概念 A: {concept_a}
- 概念 B: {concept_b}

要求：
1. 优先使用知识库内容，用表格形式呈现对比结果
2. 如果知识库中缺少某个概念的信息，说明"知识库中未收录该概念详情"
3. 标注信息来源"""


@tool
def compare_concepts(concept_a: str, concept_b: str) -> str:
    """对比两个通信相关概念或技术。当你需要比较两个技术（如OFDM vs OFDMA、
    LDPC vs Polar码、4G vs 5G）的异同时使用此工具。
    参数 concept_a 为第一个概念，concept_b 为第二个概念。"""
    if not concept_a or not concept_b:
        return "错误：请提供两个需要对比的概念"
    if concept_a == concept_b:
        return f"提示：你输入了两个相同的概念 '{concept_a}'，不需要对比。"

    # 先检索知识库中两个概念的相关内容
    try:
        vectorstore = load_vector_store()
    except FileNotFoundError:
        return "错误：知识库尚未构建，请先运行 python main.py --build 构建向量库"

    docs_a = retrieve(vectorstore, concept_a, k=3)
    docs_b = retrieve(vectorstore, concept_b, k=3)

    # 合并去重
    seen = set()
    all_context_parts = []
    for doc in docs_a + docs_b:
        key = doc.page_content[:60]
        if key not in seen:
            seen.add(key)
            source = doc.metadata.get("source", "未知")
            all_context_parts.append(f"[来源: {source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(all_context_parts)

    # 基于检索结果让 LLM 生成对比
    llm = get_llm(temperature=0)
    prompt = COMPARE_PROMPT.format(
        context=context,
        concept_a=concept_a,
        concept_b=concept_b,
    )
    response = llm.invoke([("human", prompt)])
    return response.content
