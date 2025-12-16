"""
Milvus MVP 包初始化

该包提供最小可行的向量检索功能（连接、集合管理、CRUD、搜索），
并保持结构化、可扩展的目录布局。
"""

from .config import MilvusSettings
from .client import MilvusClient
from .vectorizer import TextVectorizer

__all__ = ["MilvusSettings", "MilvusClient", "TextVectorizer"]

