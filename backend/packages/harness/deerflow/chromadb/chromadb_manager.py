from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import chromadb


# ============================================================
# ChromaDB 连接配置（不可变数据类）
# 支持通过完整 URL 或分字段的环境变量两种方式构建
# ============================================================
@dataclass(frozen=True)
class ChromaConnectionConfig:
    host: str = "chromadb"          # ChromaDB 服务主机名
    port: int = 8000                # 服务端口
    ssl: bool = False               # 是否启用 SSL
    headers: Mapping[str, str] | None = None  # 自定义请求头（如认证令牌）
    tenant: str | None = None       # 多租户标识（可选）
    database: str | None = None     # 数据库名称（可选）

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "ChromaConnectionConfig":
        """从环境变量映射中构建配置。

        优先读取 CHROMA_URL（完整 URL），若未设置则回退到
        CHROMA_HOST / CHROMA_PORT / CHROMA_SSL 分字段配置。
        CHROMA_AUTH_TOKEN 会被转为 Bearer 认证头。
        """
        raw_url = (environ.get("CHROMA_URL") or "").strip()
        auth_token = (environ.get("CHROMA_AUTH_TOKEN") or "").strip()
        tenant = (environ.get("CHROMA_TENANT") or "").strip() or None
        database = (environ.get("CHROMA_DATABASE") or "").strip() or None

        # 如果提供了认证令牌，构造 Bearer 认证头
        headers: dict[str, str] | None = None
        if auth_token:
            headers = {"Authorization": f"Bearer {auth_token}"}

        # 方式一：通过完整 URL 配置（如 http://chromadb:8000）
        if raw_url:
            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(f"Unsupported CHROMA_URL scheme: {parsed.scheme!r}")
            if not parsed.hostname:
                raise ValueError("CHROMA_URL must include a hostname")
            return cls(
                host=parsed.hostname,
                port=parsed.port or (443 if parsed.scheme == "https" else 80),
                ssl=parsed.scheme == "https",
                headers=headers,
                tenant=tenant,
                database=database,
            )

        # 方式二：通过分字段环境变量配置
        host = (environ.get("CHROMA_HOST") or "").strip() or "chromadb"
        port = cls._parse_port((environ.get("CHROMA_PORT") or "").strip() or "8000")
        ssl = cls._parse_bool((environ.get("CHROMA_SSL") or "").strip(), default=False)
        return cls(host=host, port=port, ssl=ssl, headers=headers, tenant=tenant, database=database)

    @classmethod
    def from_os_env(cls) -> "ChromaConnectionConfig":
        """从 os.environ 构建配置的快捷方法。"""
        return cls.from_env(os.environ)

    @staticmethod
    def _parse_port(raw_port: str) -> int:
        """解析并校验端口号，必须为 1-65535 的整数。"""
        try:
            port = int(raw_port)
        except ValueError as e:
            raise ValueError(f"Invalid CHROMA_PORT: {raw_port!r}") from e
        if port <= 0 or port > 65535:
            raise ValueError(f"CHROMA_PORT out of range: {port}")
        return port

    @staticmethod
    def _parse_bool(raw_value: str, *, default: bool) -> bool:
        """将字符串解析为布尔值，支持常见的真/假表示。"""
        normalized = raw_value.strip().lower()
        if not normalized:
            return default
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw_value!r}")


# ============================================================
# ChromaDB 管理器
# 轻量级 HTTP 客户端封装，集中管理连接配置，
# 并提供 collection 操作的便捷方法
# ============================================================
class ChromaDBManager:
    """
    ChromaDB HTTP 客户端的最小化封装。

    设计上保持轻量：集中管理连接配置，
    并为常见的 collection 操作提供便捷方法。
    """

    def __init__(self, config: ChromaConnectionConfig | None = None):
        self._config = config or ChromaConnectionConfig()
        self._client = self._create_client(self._config)

    @property
    def client(self) -> Any:
        """获取底层 chromadb HttpClient 实例。"""
        return self._client

    @property
    def config(self) -> ChromaConnectionConfig:
        """获取当前连接配置。"""
        return self._config

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "ChromaDBManager":
        """从环境变量映射创建管理器实例。"""
        return cls(config=ChromaConnectionConfig.from_env(environ))

    @classmethod
    def from_os_env(cls) -> "ChromaDBManager":
        """从 os.environ 创建管理器实例的快捷方法。"""
        return cls(config=ChromaConnectionConfig.from_os_env())

    # ---------- 客户端初始化 ----------

    @staticmethod
    def _create_client(config: ChromaConnectionConfig) -> Any:
        """根据连接配置创建 chromadb HttpClient。"""
        kwargs: dict[str, Any] = {
            "host": config.host,
            "port": config.port,
            "ssl": config.ssl,
        }
        if config.headers:
            kwargs["headers"] = dict(config.headers)
        # tenant/database 在较新版本的 chromadb 中支持，按需传入以保持兼容性
        if config.tenant is not None:
            kwargs["tenant"] = config.tenant
        if config.database is not None:
            kwargs["database"] = config.database
        return chromadb.HttpClient(**kwargs)

    # ---------- 连接健康检查 ----------

    def heartbeat(self) -> Any:
        """发送心跳请求，返回 ChromaDB 服务端的心跳值。"""
        return self._client.heartbeat()

    def check_connection(self) -> tuple[bool, str]:
        """验证 ChromaDB 连接是否正常，返回 (是否连通, 描述信息)。"""
        try:
            heartbeat_value = self.heartbeat()
            return True, f"connected (heartbeat={heartbeat_value})"
        except Exception as exc:
            return False, f"disconnected ({type(exc).__name__}: {exc})"

    # ---------- Collection 管理 ----------

    def list_collections(self) -> Any:
        """列出所有 collection。"""
        return self._client.list_collections()

    def get_collection(self, name: str, *, embedding_function: Any | None = None) -> Any:
        """获取已有的 collection，不存在时抛出异常。"""
        return self._client.get_collection(name=name, embedding_function=embedding_function)

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> Any:
        """获取或创建 collection，适用于写入场景。"""
        return self._client.get_or_create_collection(name=name, metadata=dict(metadata) if metadata else None, embedding_function=embedding_function)

    def delete_collection(self, name: str) -> Any:
        """删除指定 collection。"""
        return self._client.delete_collection(name=name)

    # ---------- 文档写入 ----------

    def add_texts(
        self,
        *,
        collection_name: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        """向 collection 中添加文档（ID 已存在时会报错）。"""
        col = self.get_or_create_collection(collection_name)
        return col.add(ids=list(ids), documents=list(documents), metadatas=[dict(m) for m in metadatas] if metadatas else None)

    def upsert_texts(
        self,
        *,
        collection_name: str,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        """向 collection 中插入或更新文档（ID 已存在则覆盖）。"""
        col = self.get_or_create_collection(collection_name)
        return col.upsert(ids=list(ids), documents=list(documents), metadatas=[dict(m) for m in metadatas] if metadatas else None)

    # ---------- 文档查询 ----------

    def query_texts(
        self,
        *,
        collection_name: str,
        query_texts: Sequence[str],
        n_results: int = 5,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
    ) -> Any:
        """对 collection 进行语义查询。collection 不存在时抛出异常，避免静默创建空集合。"""
        col = self.get_collection(collection_name)
        return col.query(
            query_texts=list(query_texts),
            n_results=int(n_results),
            where=dict(where) if where else None,
            where_document=dict(where_document) if where_document else None,
            include=list(include) if include else None,
        )

    # ---------- 文档删除 ----------

    def delete(
        self,
        *,
        collection_name: str,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
    ) -> Any:
        """从 collection 中删除文档。必须至少指定一个过滤条件，防止意外全量删除。"""
        if ids is None and where is None and where_document is None:
            raise ValueError("At least one of ids, where, or where_document must be provided to avoid deleting all documents")
        col = self.get_collection(collection_name)
        return col.delete(ids=list(ids) if ids else None, where=dict(where) if where else None, where_document=dict(where_document) if where_document else None)

