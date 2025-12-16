from typing import List, Optional, Dict, Any

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

from .config import MilvusSettings
from .log import get_logger

logger = get_logger(__name__)


class MilvusClient:
    """Milvus 客户端封装（基础 CRUD + 集合管理 + 搜索）。"""

    def __init__(self, settings: Optional[MilvusSettings] = None, connection_name: str = "default"):
        self.settings = settings or MilvusSettings()
        self.connection_name = connection_name
        self.collection_name = self.settings.collection_name

    # -------------------- 连接管理 -------------------- #
    def connect(self):
        """连接到 Milvus 服务器（幂等）。"""
        connections.connect(
            alias=self.connection_name,
            host=self.settings.host,
            port=self.settings.port,
        )
        logger.info("已连接 Milvus: %s:%s", self.settings.host, self.settings.port)

    def disconnect(self):
        """断开连接。"""
        try:
            connections.disconnect(self.connection_name)
            logger.info("已断开 Milvus 连接")
        except Exception as exc:
            logger.warning("断开连接时发生异常: %s", exc)

    def is_connected(self) -> bool:
        """检查连接状态。"""
        try:
            connections.get_connection_addr(self.connection_name)
            return True
        except Exception:
            return False

    # -------------------- 集合管理 -------------------- #
    def create_collection(
        self,
        dimension: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> Collection:
        """创建集合（如已存在则直接返回）。"""
        if collection_name:
            self.collection_name = collection_name
        if dimension is None:
            dimension = self.settings.dimension

        if utility.has_collection(self.collection_name):
            logger.info("集合已存在: %s", self.collection_name)
            return Collection(self.collection_name)

        fields = [
            FieldSchema(
                name=self.settings.id_field,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=self.settings.auto_id,
            ),
            FieldSchema(
                name=self.settings.text_field,
                dtype=DataType.VARCHAR,
                max_length=self.settings.max_length,
            ),
            FieldSchema(
                name=self.settings.anns_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
            ),
        ]
        schema = CollectionSchema(fields, description="Document similarity search")
        collection = Collection(self.collection_name, schema)

        collection.create_index(self.settings.anns_field, self.settings.index_params())
        logger.info("已创建集合: %s", self.collection_name)
        return collection

    def get_collection(self) -> Collection:
        """获取集合对象，不存在时抛出异常。"""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"集合 '{self.collection_name}' 不存在，请先创建集合")
        return Collection(self.collection_name)

    def list_collections(self) -> List[str]:
        return utility.list_collections()

    def drop_collection(self, collection_name: Optional[str] = None):
        name = collection_name or self.collection_name
        if not utility.has_collection(name):
            raise ValueError(f"集合 '{name}' 不存在")
        utility.drop_collection(name)
        logger.info("已删除集合: %s", name)

    def clear_collection(self, collection_name: Optional[str] = None):
        """删除集合中的所有数据（保留 schema）。"""
        name = collection_name or self.collection_name
        if not utility.has_collection(name):
            raise ValueError(f"集合 '{name}' 不存在")

        collection = Collection(name)
        collection.load()
        results = collection.query(expr="id >= 0", output_fields=[self.settings.id_field])
        if not results:
            logger.info("集合 %s 已为空", name)
            return

        ids = [doc[self.settings.id_field] for doc in results]
        expr = f"{self.settings.id_field} in [{','.join(map(str, ids))}]"
        collection.delete(expr=expr)
        collection.flush()
        logger.info("已清空集合 %s，删除 %d 条记录", name, len(ids))

    # -------------------- 数据操作 -------------------- #
    def insert_documents(self, texts: List[str], embeddings: List[List[float]]):
        if len(texts) != len(embeddings):
            raise ValueError("文本和向量数量必须一致")

        collection = self.get_collection()
        entities = [texts, embeddings]
        collection.insert(entities)
        collection.flush()
        logger.info("已插入 %d 条文档", len(texts))

    def delete_document(self, doc_id: int):
        collection = self.get_collection()
        collection.delete(expr=f"{self.settings.id_field} == {doc_id}")
        collection.flush()
        logger.info("已删除文档 ID: %s", doc_id)

    def delete_documents(self, doc_ids: List[int]):
        if not doc_ids:
            return
        collection = self.get_collection()
        expr = f"{self.settings.id_field} in [{','.join(map(str, doc_ids))}]"
        collection.delete(expr=expr)
        collection.flush()
        logger.info("已批量删除 %d 条文档", len(doc_ids))

    def update_document(self, doc_id: int, text: str, embedding: List[float]):
        """
        更新文档：当前 schema 使用 auto_id=True，因此更新采用“删除+插入”模式，
        新记录会获得新的 ID。如果需要保持 ID 不变，请改用 auto_id=False 的 schema。
        """
        # 确认存在
        existing = self.get_document(doc_id)
        if not existing:
            raise ValueError(f"文档 {doc_id} 不存在")

        collection = self.get_collection()
        collection.delete(expr=f"{self.settings.id_field} == {doc_id}")
        collection.flush()

        entities = [[text], [embedding]]
        collection.insert(entities)
        collection.flush()
        logger.info("已更新文档（原 ID: %s，auto_id 会产生新 ID）", doc_id)

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        collection = self.get_collection()
        collection.load()
        results = collection.query(
            expr=f"{self.settings.id_field} == {doc_id}",
            output_fields=[self.settings.id_field, self.settings.text_field, self.settings.anns_field],
        )
        return results[0] if results else None

    def query_by_ids(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        collection = self.get_collection()
        collection.load()
        expr = f"{self.settings.id_field} in [{','.join(map(str, doc_ids))}]"
        return collection.query(
            expr=expr,
            output_fields=[self.settings.id_field, self.settings.text_field, self.settings.anns_field],
        )

    # -------------------- 搜索与统计 -------------------- #
    def search(self, query_embedding: List[float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        collection = self.get_collection()
        collection.load()
        params = self.settings.search_params()
        k = top_k or self.settings.top_k_default

        results = collection.search(
            data=[query_embedding],
            anns_field=self.settings.anns_field,
            param=params,
            limit=k,
            output_fields=[self.settings.text_field],
        )

        formatted: List[Dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                formatted.append(
                    {
                        "id": hit.id,
                        "text": hit.entity.get(self.settings.text_field),
                        "distance": hit.distance,
                        "score": 1 / (1 + hit.distance),
                    }
                )
        return formatted

    def get_collection_stats(self) -> Dict[str, Any]:
        collection = self.get_collection()
        collection.load()
        return {
            "collection_name": self.collection_name,
            "num_entities": collection.num_entities,
        }

