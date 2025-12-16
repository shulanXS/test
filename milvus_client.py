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
    
    def is_connected(self) -> bool:
        """检查是否已连接到Milvus"""
        try:
            connections.get_connection_addr(self.connection_name)
            return True
        except:
            return False
    
    def delete_document(self, doc_id: int) -> bool:
        """
        根据ID删除单个文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否删除成功
        """
        collection = self.get_collection()
        try:
            collection.delete(expr=f"id == {doc_id}")
            collection.flush()
            print(f"✓ 成功删除文档 ID: {doc_id}")
            return True
        except Exception as e:
            print(f"✗ 删除文档失败: {e}")
            return False
    
    def delete_documents(self, doc_ids: List[int]) -> int:
        """
        批量删除文档
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            成功删除的文档数量
        """
        if not doc_ids:
            return 0
        
        collection = self.get_collection()
        try:
            # 构建删除表达式
            ids_str = ",".join([str(id) for id in doc_ids])
            expr = f"id in [{ids_str}]"
            
            collection.delete(expr=expr)
            collection.flush()
            
            deleted_count = len(doc_ids)
            print(f"✓ 成功删除 {deleted_count} 条文档")
            return deleted_count
        except Exception as e:
            print(f"✗ 批量删除文档失败: {e}")
            return 0
    
    def update_document(self, doc_id: int, text: str, embedding: List[float]) -> bool:
        """
        更新文档内容
        
        注意：由于当前schema使用auto_id=True，更新操作会先删除原文档再插入新文档，
        新文档会获得新的ID。如果需要保持ID不变，请修改schema使用auto_id=False。
        
        Args:
            doc_id: 文档ID
            text: 新的文档文本
            embedding: 新的向量
            
        Returns:
            是否更新成功
        """
        collection = self.get_collection()
        try:
            # 检查文档是否存在
            existing_doc = self.get_document(doc_id)
            if not existing_doc:
                print(f"✗ 文档 ID: {doc_id} 不存在")
                return False
            
            # Milvus的更新操作：先删除再插入
            # 注意：由于auto_id=True，新插入的文档会有新的ID
            collection.delete(expr=f"id == {doc_id}")
            collection.flush()
            
            # 插入更新后的数据
            entities = [
                [text],
                [embedding]
            ]
            collection.insert(entities)
            collection.flush()
            
            print(f"✓ 成功更新文档（原ID: {doc_id}，新文档已插入）")
            print("  注意：由于使用auto_id，新文档会有新的ID")
            return True
        except Exception as e:
            print(f"✗ 更新文档失败: {e}")
            return False
    
    def get_document(self, doc_id: int) -> Optional[dict]:
        """
        根据ID查询单个文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档信息字典，包含id、text、embedding，如果不存在则返回None
        """
        collection = self.get_collection()
        collection.load()
        
        try:
            # 使用query方法查询
            results = collection.query(
                expr=f"id == {doc_id}",
                output_fields=["id", "text", "embedding"]
            )
            
            if results:
                return results[0]
            else:
                print(f"未找到文档 ID: {doc_id}")
                return None
        except Exception as e:
            print(f"✗ 查询文档失败: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """
        列出所有集合名称
        
        Returns:
            集合名称列表
        """
        try:
            collections = utility.list_collections()
            return collections
        except Exception as e:
            print(f"✗ 列出集合失败: {e}")
            return []
    
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        删除集合
        
        Args:
            collection_name: 集合名称，默认使用self.collection_name
            
        Returns:
            是否删除成功
        """
        name = collection_name or self.collection_name
        
        if not utility.has_collection(name):
            print(f"集合 '{name}' 不存在")
            return False
        
        try:
            utility.drop_collection(name)
            print(f"✓ 成功删除集合 '{name}'")
            
            # 如果删除的是当前集合，重置集合名称
            if name == self.collection_name:
                self.collection_name = "document_collection"
            
            return True
        except Exception as e:
            print(f"✗ 删除集合失败: {e}")
            return False
    
    def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        清空集合中的所有数据（保留集合结构）
        
        Args:
            collection_name: 集合名称，默认使用self.collection_name
            
        Returns:
            是否清空成功
        """
        name = collection_name or self.collection_name
        
        if not utility.has_collection(name):
            print(f"集合 '{name}' 不存在")
            return False
        
        try:
            collection = Collection(name)
            collection.load()
            
            # 获取所有文档ID
            results = collection.query(
                expr="id >= 0",  # 查询所有文档
                output_fields=["id"]
            )
            
            if not results:
                print(f"集合 '{name}' 已经是空的")
                return True
            
            # 删除所有文档
            doc_ids = [doc["id"] for doc in results]
            ids_str = ",".join([str(id) for id in doc_ids])
            expr = f"id in [{ids_str}]"
            
            collection.delete(expr=expr)
            collection.flush()
            
            print(f"✓ 成功清空集合 '{name}'，删除了 {len(doc_ids)} 条文档")
            return True
        except Exception as e:
            print(f"✗ 清空集合失败: {e}")
            return False
    
    def query_by_ids(self, doc_ids: List[int]) -> List[dict]:
        """
        根据ID列表批量查询文档
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            文档信息列表
        """
        if not doc_ids:
            return []
        
        collection = self.get_collection()
        collection.load()
        
        try:
            # 构建查询表达式
            ids_str = ",".join([str(id) for id in doc_ids])
            expr = f"id in [{ids_str}]"
            
            results = collection.query(
                expr=expr,
                output_fields=["id", "text", "embedding"]
            )
            
            return results
        except Exception as e:
            print(f"✗ 批量查询文档失败: {e}")
            return []

