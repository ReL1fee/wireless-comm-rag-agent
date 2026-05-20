"""
向量存储与检索模块
将文本块向量化并存储到 ChromaDB，提供检索接口
"""
import os
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from src.config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL,
    CHROMA_DB_DIR, RETRIEVAL_K
)


class DashScopeEmbeddings(Embeddings):
    """通义千问 Embedding 封装，直接使用 OpenAI SDK 调用"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        embeddings = []
        # 逐条调用，避免 DashScope 兼容层批处理的问题
        for text in texts:
            if not text or not text.strip():
                text = " "  # 空文本用空格占位
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding


def get_embeddings() -> DashScopeEmbeddings:
    """获取 Embedding 模型"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "请设置 DASHSCOPE_API_KEY 环境变量。\n"
            "方式1: 复制 .env.example 为 .env 并填入你的 API Key\n"
            "方式2: 在终端执行 export DASHSCOPE_API_KEY=你的key"
        )
    return DashScopeEmbeddings(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        model=EMBEDDING_MODEL,
    )


def create_vector_store(chunks: list, persist_dir: str = CHROMA_DB_DIR) -> Chroma:
    """将文本块向量化并存入 ChromaDB"""
    print("[向量存储] 正在向量化并存入 ChromaDB ...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metadatas,
        persist_directory=persist_dir,
    )
    print(f"[向量存储] 已存储 {len(chunks)} 个文本块到 {persist_dir}")
    return vectorstore


def load_vector_store(persist_dir: str = CHROMA_DB_DIR) -> Chroma:
    """加载已存在的向量库"""
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        raise FileNotFoundError(
            f"向量库目录 {persist_dir} 不存在或为空，请先运行 --build 构建向量库"
        )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def retrieve(vectorstore: Chroma, query: str, k: int = RETRIEVAL_K) -> list:
    """检索与查询最相关的文档块"""
    return vectorstore.similarity_search(query, k=k)


def add_chunks(chunks: list, persist_dir: str = CHROMA_DB_DIR) -> int:
    """向已有向量库增量添加文本块"""
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        # 向量库不存在则新建
        create_vector_store(chunks, persist_dir)
        return len(chunks)

    vectorstore = load_vector_store(persist_dir)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print(f"[向量存储] 增量添加 {len(chunks)} 个文本块")
    return len(chunks)
