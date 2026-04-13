"""批量导入文档到 ChromaDB 向量数据库。

支持 PDF、Word、PPT、Excel 等 markitdown 能处理的格式。
复用项目已有的 ChromaDBManager 和分块逻辑，无需启动 agent。

用法:
    #先把要存的文档拷贝到容器中（宿主机中执行，容器卸载之后，拷贝的临时文件会删除）
    docker cp ~/agentdev/TD_Docs_1 deer-flow-gateway:/tmp/docs

    #(python虚拟环境需要用backend/.venv/bin/python)
    
    # 导入整个目录
    uv run scripts/ingest_to_chromadb.py /tmp/docs

    # 指定 collection
    uv run scripts/ingest_to_chromadb.py /tmp/docs --collection my_kb

    # 导入单个文件
    uv run scripts/ingest_to_chromadb.py /tmp/docs/file.pdf

    # 自定义分块参数
    uv run scripts/ingest_to_chromadb.py /tmp/docs--chunk-size 1000 --chunk-overlap 100

    # 查询数据库记录的内容：
    cd /app/backend && env | grep CHROMA; uv run python -c "from deerflow.chromadb import ChromaDBManager; m = ChromaDBManager.from_os_env(); print('连接:', m.check_connection()); [print(f'  collection: {c.name if hasattr(c,\"name\") else c}, 文档数: {m.client.get_collection(c.name if hasattr(c,\"name\") else c).count()}') for c in m.list_collections()]"


环境变量:
    CHROMA_URL          完整 URL，如 http://chromadb:8000
    CHROMA_HOST          主机名（默认 chromadb）
    CHROMA_PORT          端口（默认 8000）
    CHROMA_AUTH_TOKEN    认证令牌（可选）
    CHROMA_TENANT        多租户标识（可选）
    CHROMA_DATABASE      数据库名称（可选）
"""

import argparse
import hashlib
import io
import re
import sys
from pathlib import Path

from markitdown import MarkItDown

from deerflow.chromadb import ChromaDBManager
from chromadb_registry import is_registered, register_file

# markitdown 输出的 Markdown 图片语法：![alt](src)
# 包括普通路径和 data:image base64 两种形式
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

# markitdown 支持的文档格式
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}

# 默认分块参数
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def replace_images(text: str) -> str:
    """将 Markdown 图片语法替换为 [Image] 占位符。"""
    return _IMAGE_RE.sub("[Image]", text)


def convert_to_text(file_path: Path) -> str:
    """用 markitdown 将文档转为纯文本，图片替换为 [Image] 占位符。

    转换过程中 pdfminer 等库会输出大量重复警告到 stderr，
    这里捕获 stderr 并对每个文件只显示去重后的警告各一次。
    """
    md = MarkItDown()
    old_stderr = sys.stderr
    captured = io.StringIO()
    sys.stderr = captured
    try:
        result = md.convert(str(file_path))
    finally:
        sys.stderr = old_stderr

    # 将捕获的 stderr 去重后输出（每种警告只显示一次）
    seen: set[str] = set()
    for line in captured.getvalue().splitlines():
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            print(f"  [WARN] {stripped}", file=sys.stderr)

    text = result.text_content or ""
    return replace_images(text)


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """生成分块唯一 ID：文件路径哈希 + 块序号。"""
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
    return f"{path_hash}_{chunk_index}"


def collect_files(path: Path) -> list[Path]:
    """收集路径下所有支持格式的文件。"""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        return []
    files: set[Path] = set()
    for ext in SUPPORTED_EXTENSIONS:
        files.update(path.rglob(f"*{ext}"))
    return sorted(files)


def ingest_file(file_path: Path, manager: ChromaDBManager, collection_name: str, chunk_size: int, chunk_overlap: int) -> tuple[bool, int]:
    """处理单个文件：转换 -> 分块 -> 写入。返回 (是否成功, 块数)。"""
    # 转换
    try:
        text = convert_to_text(file_path)
    except Exception as exc:
        print(f"  [FAIL] 转换失败: {exc}")
        return False, 0

    if not text.strip():
        print(f"  [SKIP] 文件内容为空")
        return False, 0

    # 分块
    chunks = split_text(text, chunk_size, chunk_overlap)
    if not chunks:
        print(f"  [SKIP] 分块后无有效内容")
        return False, 0

    # 构建 ids 和 metadata
    file_key = str(file_path.resolve())
    ids = [make_chunk_id(file_key, i) for i in range(len(chunks))]
    metadatas = [
        {
            "source": file_key,
            "filename": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    # 写入
    try:
        manager.upsert_texts(
            collection_name=collection_name,
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as exc:
        print(f"  [FAIL] 写入 ChromaDB 失败: {exc}")
        return False, 0

    return True, len(chunks)


def main():
    parser = argparse.ArgumentParser(description="批量导入文档到 ChromaDB 向量数据库")
    parser.add_argument("path", type=Path, help="文件或目录路径")
    parser.add_argument("--collection", default="default", help="ChromaDB collection 名称 (默认: default)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"分块大小 (默认: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"分块重叠 (默认: {DEFAULT_CHUNK_OVERLAP})")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"错误: 路径不存在 — {args.path}")
        sys.exit(1)

    # 收集文件
    files = collect_files(args.path)
    if not files:
        print(f"未找到支持的文档文件。支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    print(f"找到 {len(files)} 个文档文件")
    print(f"Collection: {args.collection}")
    print(f"分块参数: size={args.chunk_size}, overlap={args.chunk_overlap}")
    print()

    # 连接 ChromaDB
    try:
        manager = ChromaDBManager.from_os_env()
        ok, info = manager.check_connection()
        if not ok:
            print(f"错误: ChromaDB 连接失败 — {info}")
            sys.exit(1)
        print(f"ChromaDB: {info}")
    except Exception as exc:
        print(f"错误: 无法创建 ChromaDB 客户端 — {exc}")
        sys.exit(1)

    print()

    # 逐文件处理
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_chunks = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name}")

        # 检查注册表，跳过已导入的同名文档
        if is_registered(args.collection, file_path.name):
            print(f"  [SKIP] 已存在于 collection '{args.collection}'，跳过")
            skip_count += 1
            continue

        ok, chunks = ingest_file(file_path, manager, args.collection, args.chunk_size, args.chunk_overlap)
        if ok:
            success_count += 1
            total_chunks += chunks
            register_file(args.collection, file_path.name)
            print(f"  [OK] {chunks} 个分块已写入")
        else:
            fail_count += 1

    # 统计
    print()
    print("=" * 40)
    print(f"完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}, 总分块数: {total_chunks}")


if __name__ == "__main__":
    main()
