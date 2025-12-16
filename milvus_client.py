"""
Milvus客户端模块
用于连接Milvus数据库并管理集合
"""
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from typing import List, Optional
import numpy as np


class MilvusClient:
    """Milvus客户端封装类"""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        """
        初始化Milvus客户端
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
        """
        self.host = host
        self.port = port
        self.connection_name = "default"
        self.collection_name = "document_collection"
        
    def connect(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(
                alias=self.connection_name,
                host=self.host,
                port=self.port
            )
            print(f"✓ 成功连接到Milvus服务器 ({self.host}:{self.port})")
        except Exception as e:
            print(f"✗ 连接Milvus失败: {e}")
            raise
    
    def disconnect(self):
        """断开Milvus连接"""
        try:
            connections.disconnect(self.connection_name)
            print("✓ 已断开Milvus连接")
        except Exception as e:
            print(f"✗ 断开连接失败: {e}")
    
    def create_collection(self, dimension: int = 384, collection_name: Optional[str] = None):
        """
        创建集合
        
        Args:
            dimension: 向量维度（默认384，适用于sentence-transformers的all-MiniLM-L6-v2模型）
            collection_name: 集合名称，默认使用self.collection_name
        """
        if collection_name:
            self.collection_name = collection_name
        
        # 检查集合是否已存在
        if utility.has_collection(self.collection_name):
            print(f"集合 '{self.collection_name}' 已存在")
            return Collection(self.collection_name)
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        # 创建schema
        schema = CollectionSchema(fields, "文档相似性搜索集合")
        
        # 创建集合
        collection = Collection(self.collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        print(f"✓ 成功创建集合 '{self.collection_name}'")
        return collection
    
    def get_collection(self) -> Collection:
        """获取集合对象"""
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"集合 '{self.collection_name}' 不存在，请先创建集合")
        return Collection(self.collection_name)
    
    def insert_documents(self, texts: List[str], embeddings: List[List[float]]):
        """
        插入文档和向量
        
        Args:
            texts: 文档文本列表
            embeddings: 对应的向量列表
        """
        if len(texts) != len(embeddings):
            raise ValueError("文本和向量数量必须一致")
        
        collection = self.get_collection()
        
        # 准备数据
        entities = [
            texts,
            embeddings
        ]
        
        # 插入数据
        collection.insert(entities)
        collection.flush()
        
        print(f"✓ 成功插入 {len(texts)} 条文档")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回最相似的k个结果
            
        Returns:
            搜索结果列表，每个结果包含id、text、distance等信息
        """
        collection = self.get_collection()
        collection.load()
        
        # 执行搜索
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        # 格式化结果
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "distance": hit.distance,
                    "score": 1 / (1 + hit.distance)  # 将距离转换为相似度分数
                })
        
        return search_results
    
    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        collection = self.get_collection()
        collection.load()
        
        num_entities = collection.num_entities
        return {
            "collection_name": self.collection_name,
            "num_entities": num_entities
        }

