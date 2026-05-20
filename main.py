"""
项目主入口
用法：
  python main.py              → 启动 Streamlit 界面
  python main.py --build      → 仅构建知识库
  python main.py --ask "问题"  → 命令行问答
"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_build():
    """命令行构建知识库"""
    from src.document_loader import get_chunks
    from src.vector_store import create_vector_store

    print("=" * 50)
    print("  无线通信知识库构建工具")
    print("=" * 50)
    print()

    chunks = get_chunks()
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"  [{i+1}] {source} | {preview}...")
    print()

    vectorstore = create_vector_store(chunks)
    print()
    print("知识库构建完成！现在可以运行 python main.py 启动问答界面")


def cmd_ask(question: str):
    """命令行问答 — 统一走 Agent"""
    from src.agent import chat

    print()
    print("=" * 60)
    print(f"问题: {question}")
    print("=" * 60)
    print()
    answer = chat(question)
    print(answer)
    print()


def cmd_ui():
    """启动 Web 界面"""
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, "src", "ui.py")
    subprocess.run(["streamlit", "run", ui_path])


if __name__ == "__main__":
    if "--build" in sys.argv:
        cmd_build()
    elif "--ask" in sys.argv:
        idx = sys.argv.index("--ask")
        if idx + 1 < len(sys.argv):
            cmd_ask(sys.argv[idx + 1])
        else:
            print("用法: python main.py --ask \"你的问题\"")
    else:
        cmd_ui()
