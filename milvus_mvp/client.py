"""
Milvus 客户端封装（领域层）

核心职责：
- 负责与 Milvus Server 的直接交互：连接管理、集合管理、CRUD、搜索、统计。
- 不处理向量化/业务流程，保持“单一职责”：只管把指令翻译成 PyMilvus 调用。

推荐阅读顺序：
1) 连接管理  connect / disconnect / is_connected
2) 集合管理  create_collection / get_collection / list_collections / drop_collection / clear_collection
3) 数据操作  insert / delete / update / get / query_by_ids
4) 搜索与统计 search / get_collection_stats
"""

from typing import List, Optional, Dict, Any

from pymilvus import (
    connections,  # 连接管理
    Collection,   # 集合对象
    FieldSchema,  # 字段定义
    CollectionSchema,  # 集合 schema 定义
    DataType,     # 字段类型枚举
    utility,      # 工具函数：集合存在检测 / 列表等
)

from .config import MilvusSettings
from .log import get_logger

logger = get_logger(__name__)


class MilvusClient:
    """
    Milvus 客户端封装（基础 CRUD + 集合管理 + 搜索）。

    设计要点：
    - 通过 MilvusSettings 统一配置（host/port/collection/index/search 等），支持环境变量覆盖。
    - 保持幂等：connect 多次不会报错；create_collection 如果存在直接返回。
    - 不做向量化与业务流程，方便被 service 层复用或被其他入口调用。
    """

    def __init__(self, settings: Optional[MilvusSettings] = None, connection_name: str = "default"):
        self.settings = settings or MilvusSettings()
        self.connection_name = connection_name
        self.collection_name = self.settings.collection_name

    # -------------------- 连接管理 -------------------- #
    def connect(self):
        """
        这是 MilvusClient 类中的 connect 方法，其作用是建立与 Milvus 服务器的连接，且具有幂等性。

        详细讲解如下：

        1. 幂等性设计：
           该方法设计为“幂等”的，意味着无论你调用多少次 connect，
           只要连接参数一样，就不会重复建立多余的连接。PyMilvus 库本身在内部做了连接管理，
           相同 alias 名称的连接只有一个，后续调用只是复用。

        2. 连接信息来源：
           - alias：连接的别名（用于区分多连接场景），这里用 self.connection_name。
           - host：Milvus 服务所在的主机名或 IP 地址，取自 self.settings.host。
           - port：Milvus 服务监听的端口号，取自 self.settings.port。
           这些参数全部由 MilvusSettings 配置对象提供，方便通过环境变量灵活配置。

        3. 具体实现：
           - `connections.connect(...)` 调用是 PyMilvus 库提供的连接管理函数。
             它会根据 alias/host/port 建立与 Milvus 服务端的实际连接。
           - 调用完成后，通过 logger 记录一条 info 日志，标明连接已建立。

        4. 错误与异常：
           - 如果提供的 host 不可达、端口错误、网络故障等，`connections.connect` 会抛出异常。
           - 本方法本身不做异常捕获，而是把异常留给上层处理（设计哲学是让调用方明确处理失败场景）。

        5. 使用场景：
           - 一般在程序启动或首次需要与 Milvus 交互前调用。
           - 如果已经连上了，重复调用也不会有副作用。

        总结：本方法是与 Milvus 建立基础连接的入口，细节交给 PyMilvus 封装，参数来源于配置对象，设计典型而清晰。
        """
        connections.connect(
            alias=self.connection_name,
            host=self.settings.host,
            port=self.settings.port,
        )
        logger.info("已连接 Milvus: %s:%s", self.settings.host, self.settings.port)

    def disconnect(self):
        """
        断开连接。

        注意：
        - 如果连接不存在，会捕获异常并记录 warning，不中断流程。
        - 上层一般在 finally 中调用，确保资源释放。
        """
        try:
            connections.disconnect(self.connection_name)
            logger.info("已断开 Milvus 连接")
        except Exception as exc:
            logger.warning("断开连接时发生异常: %s", exc)

    def is_connected(self) -> bool:
        """
        检查连接状态（查询连接地址）。

        返回：
        - True：连接存在且可获取地址
        - False：未连接或连接失效（内部捕获异常）
        """
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
        """
        创建集合（如已存在则直接返回）。

        - 字段设计：
          * id：INT64 主键，auto_id 取决于配置（MVP 默认 True）
          * text：VARCHAR，用于存储原文
          * embedding：FLOAT_VECTOR，用于向量索引和检索
        - 索引参数：由 settings.index_params() 给出，默认 IVF_FLAT + L2
        参数：
        - dimension：向量维度，默认取 settings.dimension
        - collection_name：集合名，默认取 settings.collection_name
        返回：
        - Collection 实例（已创建或已存在）
        异常：
        - pymilvus 抛出的网络/权限/参数异常需上层处理
        """
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
        """列出当前 Milvus 实例中的所有集合名称。"""
        return utility.list_collections()

    def drop_collection(self, collection_name: Optional[str] = None):
        name = collection_name or self.collection_name
        if not utility.has_collection(name):
            raise ValueError(f"集合 '{name}' 不存在")
        utility.drop_collection(name)
        logger.info("已删除集合: %s", name)

    def clear_collection(self, collection_name: Optional[str] = None):
        """
        清空集合数据（保留 schema）。

        做法：
        - 先 query 出所有 id
        - 构造删除表达式批量删除
        - flush 确保落盘
        返回：
        - None（成功时无返回），若集合不存在抛 ValueError
        """
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
        """
        插入文本与向量。

        约束：
        - texts 与 embeddings 长度必须一致
        - embeddings 维度需与集合 schema 一致（由调用方保证）
        流程：
        - get_collection -> insert -> flush
        返回：
        - None，异常向上抛出
        """
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
        更新文档：当前 schema 使用 auto_id=True，因此采取“删除+插入”模式。

        影响：
        - 新记录会生成新的 ID（auto_id=True 的特性）
        - 如果需要保持 ID 不变，需改 schema 为 auto_id=False 并在插入时显式传 ID。
        流程：
        - 检查存在 -> delete 原记录 -> flush -> insert 新记录 -> flush
        返回：
        - None，若文档不存在抛 ValueError，其他异常由 PyMilvus 抛出
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
        """
        向量搜索，返回格式化后的结果列表。

        - params 来自 settings.search_params()，默认 metric=L2，nprobe 可配置。
        - 输出字段仅 text（可按需扩展）。
        - score 为简单的 1/(1+distance) 近似相似度。
        参数：
        - query_embedding：单条查询向量
        - top_k：返回条数，默认取 settings.top_k_default
        返回：
        - List[dict]，含 id/text/distance/score
        """
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
        """
        返回集合名与实体数量，便于快速查看数据规模。
        返回：
        - dict: {"collection_name": str, "num_entities": int}
        """
        collection = self.get_collection()
        collection.load()
        return {
            "collection_name": self.collection_name,
            "num_entities": collection.num_entities,
        }

