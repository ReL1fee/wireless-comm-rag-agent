"""
Streamlit 前端界面 — 支持文件上传扩展知识库
"""
import streamlit as st
import sys
import os
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DASHSCOPE_API_KEY, RETRIEVAL_K, UPLOAD_DIR
from src.document_loader import get_chunks, process_uploaded_file
from src.vector_store import create_vector_store, load_vector_store, retrieve, add_chunks
from src.agent import chat

st.set_page_config(
    page_title="无线通信知识智能助手",
    page_icon="📡",
    layout="wide",
)

# ---- 确保上传目录存在 ----
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # [(filename, time, chunk_count)]


def init_kb():
    if not DASHSCOPE_API_KEY:
        return False
    try:
        st.session_state.vectorstore = load_vector_store()
        st.session_state.kb_ready = True
        return True
    except FileNotFoundError:
        return False


def build_kb():
    with st.spinner("正在加载文档..."):
        chunks = get_chunks()
    with st.spinner("正在向量化并存储..."):
        st.session_state.vectorstore = create_vector_store(chunks)
        st.session_state.kb_ready = True
    st.success(f"知识库构建完成！共 {len(chunks)} 个文本块")


def handle_uploaded_file(uploaded_file) -> bool:
    """处理用户上传的文件：保存 → 分块 → 入库"""
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    # 验证格式
    if ext not in (".pdf", ".docx", ".doc", ".md", ".txt"):
        st.error(f"不支持的格式: {ext}")
        return False

    # 保存到 uploads 目录（加时间戳避免重名）
    base, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{base}_{timestamp}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 处理文件
    try:
        with st.spinner(f"正在解析 {filename} ..."):
            chunks = process_uploaded_file(saved_path)
        with st.spinner(f"正在将 {len(chunks)} 个文本块写入知识库 ..."):
            if st.session_state.kb_ready:
                add_chunks(chunks)
            else:
                st.session_state.vectorstore = create_vector_store(chunks)
                st.session_state.kb_ready = True
        st.success(f"已入库: {filename} → {len(chunks)} 个文本块")
        st.session_state.uploaded_files.append((filename, timestamp, len(chunks)))
        return True
    except Exception as e:
        st.error(f"处理失败: {e}")
        # 删除无效文件
        if os.path.exists(saved_path):
            os.remove(saved_path)
        return False


# ---- 侧边栏 ----
with st.sidebar:
    st.title("📡 通信知识助手")

    st.divider()

    # === 知识库管理 ===
    st.subheader("📚 知识库管理")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("构建知识库", type="primary", use_container_width=True,
                      help="基于 data/ 目录下的预置文档构建"):
            if not DASHSCOPE_API_KEY:
                st.error("请先配置 API Key")
            else:
                build_kb()
    with col2:
        if st.button("加载已有", use_container_width=True,
                      help="加载已构建的知识库"):
            if init_kb():
                st.success("已加载")
            else:
                st.warning("不存在，请先构建")

    if st.session_state.kb_ready:
        st.success("✅ 知识库就绪")
    else:
        st.info("请先构建或加载知识库")

    st.divider()

    # === 文件上传 ===
    st.subheader("📤 扩展知识库")

    uploaded_file = st.file_uploader(
        "拖拽或选择文件上传",
        type=["pdf", "docx", "doc", "md", "txt"],
        help="支持 PDF、Word、Markdown、TXT 文件",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if not DASHSCOPE_API_KEY:
            st.error("请先配置 API Key")
        elif not st.session_state.kb_ready:
            st.warning("请先构建或加载知识库")
        else:
            handle_uploaded_file(uploaded_file)

    # 已上传文件列表
    if st.session_state.uploaded_files:
        with st.expander(f"已上传文件 ({len(st.session_state.uploaded_files)})"):
            for fname, ftime, fchunks in st.session_state.uploaded_files:
                st.caption(f"📄 {fname} ({fchunks}块)")

    st.divider()

    # === 检索设置 ===
    st.subheader("⚙️ 检索设置")
    k_value = st.slider("检索文档数", 1, 10, RETRIEVAL_K)

    st.divider()
    st.caption("Agent 自动选择：检索知识库 / 计算 / 对比")

    with st.expander("💡 示例问题"):
        st.markdown("""
        - OFDM中循环前缀有什么作用？
        - 计算20MHz带宽、10dB信噪比下的信道容量
        - 对比LDPC码和Polar码
        - 5G NR的子载波间隔配置是怎样的？
        - 100m距离、2.4GHz的自由空间路径损耗？
        """)


# ---- 主界面 ----
st.title("无线通信知识智能助手")
st.caption("基于 RAG + Agent — 支持上传 PDF/Word 扩展知识库")

if not DASHSCOPE_API_KEY:
    st.warning("""
    ⚠️ 请先配置通义千问 API Key：
    1. 访问 [DashScope控制台](https://dashscope.console.aliyun.com/) 获取 API Key
    2. 复制 `.env.example` 为 `.env`，填入 `DASHSCOPE_API_KEY=你的key`
    """)

# 聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 参考来源"):
                for i, src in enumerate(msg["sources"], 1):
                    source_name = src.metadata.get("source", f"来源{i}")
                    st.caption(f"**[{i}] {source_name}**")
                    c = src.page_content
                    st.text(c[:300] + "..." if len(c) > 300 else c)
                    st.divider()

# 用户输入
if question := st.chat_input("输入你的通信技术问题..."):
    if not st.session_state.kb_ready:
        st.error("请先在侧边栏构建或加载知识库")
    elif not DASHSCOPE_API_KEY:
        st.error("请先配置 API Key")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    answer = chat(question)

                    try:
                        sources = retrieve(
                            st.session_state.vectorstore, question, k=k_value
                        )
                    except Exception:
                        sources = []

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 参考来源"):
                            for i, src in enumerate(sources, 1):
                                source_name = src.metadata.get("source", f"来源{i}")
                                st.caption(f"**[{i}] {source_name}**")
                                c = src.page_content
                                st.text(c[:300] + "..." if len(c) > 300 else c)
                                st.divider()

                except Exception as e:
                    st.error(f"发生错误: {str(e)}")
                    answer = f"抱歉，处理您的问题时出错了：{e}"
                    sources = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
