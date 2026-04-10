"""chromadb_ingest 工具 —— 将上传的 Word 文档解析、分块后存入 ChromaDB。

AI 调用此工具时传入文件路径（/mnt/user-data/uploads/ 下的 .doc/.docx），
工具自动完成：转文本 -> 分块 -> 写入向量数据库。
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from langchain.tools import tool

from deerflow.chromadb import ChromaDBManager

# 分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 支持的扩展名
WORD_EXTENSIONS = {".doc", ".docx"}


def _get_manager() -> ChromaDBManager:
    """从环境变量创建 ChromaDBManager。"""
    return ChromaDBManager.from_env(os.environ)


def _convert_to_text(file_path: str) -> str:
    """用 markitdown 将 Word 文件转为纯文本。"""
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content or ""


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将长文本按固定大小分块，尽量在换行处断开。"""
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        if end < length:
            newline_pos = text.rfind("\n", start + chunk_size - chunk_overlap, end)
            if newline_pos > start:
                end = newline_pos + 1
        chunks.append(text[start:end].strip())
        start = max(start + 1, end - chunk_overlap)
    return [c for c in chunks if c]


def _make_chunk_id(file_path: str, chunk_index: int) -> str:
    """生成分块唯一 ID：文件路径哈希 + 块序号。"""
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
    return f"{path_hash}_{chunk_index}"


@tool("chromadb_ingest", parse_docstring=True)
def chromadb_ingest_tool(
    file_path: str,
    collection_name: str = "default",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> str:
    """Parse a Word document (.doc/.docx) into chunks and store them in ChromaDB for semantic retrieval.

    Use this tool when the user uploads a Word document and wants to:
    - Import it into the knowledge base for later Q&A
    - Store document content for semantic search via chromadb_query

    The tool reads the file, converts it to text, splits it into chunks,
    and upserts all chunks into the specified ChromaDB collection with metadata
    (source filename, chunk index). Re-running on the same file safely overwrites
    previous chunks (upsert semantics).

    Args:
        file_path: Path to the Word file. Typically under /mnt/user-data/uploads/.
        collection_name: Target ChromaDB collection name. Defaults to "default".
        chunk_size: Maximum characters per chunk. Defaults to 500.
        chunk_overlap: Overlap characters between adjacent chunks. Defaults to 50.
    """
    # 校验扩展名
    p = Path(file_path)
    if p.suffix.lower() not in WORD_EXTENSIONS:
        return f"Error: 不支持的文件类型 '{p.suffix}'，仅支持 .doc / .docx"

    # 转换为文本
    try:
        text = _convert_to_text(file_path)
    except Exception as exc:
        return f"Error: 文件转换失败 — {exc}"

    if not text.strip():
        return "Error: 文件内容为空，无法导入"

    # 分块
    chunks = _split_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return "Error: 分块后无有效内容"

    # 构建 ids 和 metadata
    ids = [_make_chunk_id(file_path, i) for i in range(len(chunks))]
    metadatas: list[dict[str, Any]] = [
        {
            "source": file_path,
            "filename": p.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    # 写入 ChromaDB
    try:
        manager = _get_manager()
        manager.upsert_texts(
            collection_name=collection_name,
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as exc:
        return f"Error: 写入 ChromaDB 失败 — {exc}"

    return json.dumps(
        {
            "status": "ok",
            "file": p.name,
            "collection": collection_name,
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        ensure_ascii=False,
    )
