"""清空 ChromaDB 数据库（删除全部或指定 collection）。

用法:
    # 清空所有 collection
    PYTHONPATH=packages/harness uv run python scripts/clear_chromadb.py

    # 只删除指定 collection
    PYTHONPATH=packages/harness uv runpython scripts/clear_chromadb.py --collection my_kb

    # 跳过确认提示（用于脚本/CI）
    PYTHONPATH=packages/harness uv runpython scripts/clear_chromadb.py --yes

环境变量:
    CHROMA_URL          完整 URL，如 http://chromadb:8000
    CHROMA_HOST          主机名（默认 chromadb）
    CHROMA_PORT          端口（默认 8000）
    CHROMA_AUTH_TOKEN    认证令牌（可选）
    CHROMA_TENANT        多租户标识（可选）
    CHROMA_DATABASE      数据库名称（可选）
"""

import argparse
import sys

from deerflow.chromadb import ChromaDBManager


def main():
    parser = argparse.ArgumentParser(description="清空 ChromaDB 数据库")
    parser.add_argument("--collection", help="只删除指定 collection（不指定则删除全部）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    # 连接
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

    # 确定要删除的 collection
    if args.collection:
        targets = [args.collection]
    else:
        collections = manager.list_collections()
        # chromadb 不同版本返回格式不同：可能是字符串列表或对象列表
        targets = [c.name if hasattr(c, "name") else str(c) for c in collections]

    if not targets:
        print("数据库为空，没有 collection 需要删除。")
        return

    print(f"\n即将删除以下 {len(targets)} 个 collection:")
    for name in targets:
        print(f"  - {name}")

    # 确认
    if not args.yes:
        answer = input("\n确认删除？(y/N) ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消。")
            return

    # 执行删除
    print()
    deleted = 0
    for name in targets:
        try:
            manager.delete_collection(name)
            print(f"  [OK] 已删除 {name}")
            deleted += 1
        except Exception as exc:
            print(f"  [FAIL] 删除 {name} 失败: {exc}")

    print(f"\n完成! 已删除 {deleted}/{len(targets)} 个 collection。")


if __name__ == "__main__":
    main()
