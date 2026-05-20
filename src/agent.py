"""
Agent 智能代理（统一入口）
通过 RAG 检索 + 工具调用处理所有用户问题

核心流程：
1. 用户提问 → Agent 分析问题类型
2. 需要查知识 → 调用 search_knowledge（RAG检索知识库）
3. 需要计算 → 调用 calculate_formula
4. 需要对比 → 调用 compare_concepts（内部也用RAG）
5. 简单闲聊 → 直接回答
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.rag_chain import get_llm
from src.tools.protocol_query import search_knowledge
from src.tools.formula_calc import calculate_formula
from src.tools.concept_compare import compare_concepts

TOOLS = [search_knowledge, calculate_formula, compare_concepts]

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """你是一个无线通信技术智能助手，基于 RAG 检索增强技术构建。你可以使用以下工具：

1. **search_knowledge** — 搜索通信知识库（基于RAG向量检索）
   适用：协议细节、技术原理、参数查询。如"OFDM的循环前缀有什么作用"、"5G NR的子载波间隔"、"MIMO的信道容量公式"

2. **calculate_formula** — 计算通信公式
   适用：香农容量、误码率、路径损耗、奈奎斯特速率、波长。如"计算20MHz带宽10dB信噪比的信道容量"

3. **compare_concepts** — 对比两个通信概念（内部也使用RAG检索知识库）
   适用：两个技术的异同。如"LDPC和Polar码的区别"、"4G和5G帧结构对比"

使用规则：
- 涉及通信专业知识查询 → 必须用 search_knowledge 检索知识库，不要凭自己记忆回答
- 涉及数值计算 → 用 calculate_formula
- 涉及两个概念对比 → 用 compare_concepts
- 普通闲聊或与通信无关的问题可以直接回答

回答要求：专业但易懂，适合通信工程学生理解。"""


def chat(question: str) -> str:
    """统一对话入口：用户提问，Agent 自动选择合适的工具处理"""
    llm = get_llm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for _ in range(5):
        response = llm.invoke(messages)
        messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"  [Agent] 调用工具: {tool_name}")

                func = TOOL_MAP.get(tool_name)
                if func:
                    try:
                        result = func.invoke(tool_args)
                    except Exception as e:
                        result = f"工具执行错误: {e}"
                else:
                    result = f"未知工具: {tool_name}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
        else:
            return response.content

    return "已达到最大工具调用次数，请简化您的问题。"
