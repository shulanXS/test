"""
搜索服务层（service）

职责：组合“向量化 + Milvus 搜索”这两个步骤，对上层屏蔽细节。
注意：此处不做任何业务过滤/排序，只负责最基础的 search，保持 MVP 范围最小。
"""

from typing import List, Dict, Any

from ..client import MilvusClient
from ..vectorizer import TextVectorizer


def search_texts(
    client: MilvusClient,
    vectorizer: TextVectorizer,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    文本搜索入口：
    1) 用 TextVectorizer 将 query 文本编码为向量
    2) 调用 MilvusClient.search 执行向量检索

    返回：
        List[Dict]，每个 dict 含 id/text/distance/score
    """
    query_embedding = vectorizer.encode([query])[0]
    return client.search(query_embedding, top_k=top_k)

