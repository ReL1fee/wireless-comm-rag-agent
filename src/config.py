"""
项目配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 通义千问 API 配置（通过 DashScope 兼容 OpenAI 接口）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LLM 模型名
LLM_MODEL = "qwen-plus"

# Embedding 模型名
EMBEDDING_MODEL = "text-embedding-v3"

# 知识库文件目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# 用户上传文件目录
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")

# 向量库持久化目录
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# 文档分块参数
CHUNK_SIZE = 500      # 每块字符数
CHUNK_OVERLAP = 80    # 块之间重叠字符数

# 检索参数
RETRIEVAL_K = 4       # 每次检索返回的文档块数
