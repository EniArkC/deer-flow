"""ChromaDB 文档注册表 — 记录已导入的文档，避免重复导入。

注册表文件默认存储在 backend/.deer-flow/chromadb_added_files.json，格式：

{
    "collection_name": ["doc1.pdf", "doc2.docx", ...],
    ...
}

由 ingest_to_chromadb.py 和 clear_chromadb.py 共同维护。
"""

import json
from pathlib import Path

# 默认注册表路径：backend/.deer-flow/chromadb_added_files.json
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / ".deer-flow" / "chromadb_added_files.json"


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, list[str]]:
    """读取注册表，文件不存在则返回空 dict。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_registry(registry: dict[str, list[str]], path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """写入注册表（先写临时文件再 rename）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_registered(collection: str, filename: str, path: Path = DEFAULT_REGISTRY_PATH) -> bool:
    """判断某个 collection 中是否已记录过指定文件名。"""
    registry = load_registry(path)
    return filename in registry.get(collection, [])


def register_file(collection: str, filename: str, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """将文件名添加到指定 collection 的记录中（去重）。"""
    registry = load_registry(path)
    files = registry.setdefault(collection, [])
    if filename not in files:
        files.append(filename)
    save_registry(registry, path)


def register_files(collection: str, filenames: list[str], path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """批量注册文件名（去重）。"""
    registry = load_registry(path)
    files = registry.setdefault(collection, [])
    for name in filenames:
        if name not in files:
            files.append(name)
    save_registry(registry, path)


def remove_collection(collection: str, path: Path = DEFAULT_REGISTRY_PATH) -> bool:
    """从注册表中删除整个 collection 的记录。返回是否实际删除了记录。"""
    registry = load_registry(path)
    if collection in registry:
        del registry[collection]
        save_registry(registry, path)
        return True
    return False


def remove_all_collections(path: Path = DEFAULT_REGISTRY_PATH) -> int:
    """清空注册表中所有 collection 记录，返回被清除的 collection 数量。"""
    registry = load_registry(path)
    count = len(registry)
    if count > 0:
        save_registry({}, path)
    return count
