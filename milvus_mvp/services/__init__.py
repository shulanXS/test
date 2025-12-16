"""Service 层：封装常用业务流程（向量化 + CRUD + 搜索）。"""

from .ingest import insert_texts, update_text, delete_by_ids, get_by_id, get_by_ids
from .search import search_texts

__all__ = [
    "insert_texts",
    "update_text",
    "delete_by_ids",
    "get_by_id",
    "get_by_ids",
    "search_texts",
]

