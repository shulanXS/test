from typing import List, Dict, Any

from ..client import MilvusClient
from ..vectorizer import TextVectorizer


def search_texts(
    client: MilvusClient,
    vectorizer: TextVectorizer,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_embedding = vectorizer.encode([query])[0]
    return client.search(query_embedding, top_k=top_k)

