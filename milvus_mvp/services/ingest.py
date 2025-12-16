from typing import List, Optional

from ..client import MilvusClient
from ..vectorizer import TextVectorizer


def insert_texts(client: MilvusClient, vectorizer: TextVectorizer, texts: List[str]) -> int:
    embeddings = vectorizer.encode(texts)
    client.insert_documents(texts, embeddings)
    return len(texts)


def update_text(client: MilvusClient, vectorizer: TextVectorizer, doc_id: int, text: str):
    embedding = vectorizer.encode([text])[0]
    client.update_document(doc_id, text, embedding)


def delete_by_ids(client: MilvusClient, doc_ids: List[int]):
    if len(doc_ids) == 1:
        client.delete_document(doc_ids[0])
    else:
        client.delete_documents(doc_ids)


def get_by_id(client: MilvusClient, doc_id: int):
    return client.get_document(doc_id)


def get_by_ids(client: MilvusClient, doc_ids: List[int]):
    return client.query_by_ids(doc_ids)

