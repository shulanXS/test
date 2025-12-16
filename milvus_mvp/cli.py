import argparse
import sys
from typing import Optional, List

from .client import MilvusClient
from .config import MilvusSettings
from .log import get_logger
from .vectorizer import TextVectorizer
from .services import ingest, search as search_service

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Milvus 文档相似性搜索 MVP")
    parser.add_argument("--host", default=None, help="Milvus 服务器地址")
    parser.add_argument("--port", type=int, default=None, help="Milvus 服务器端口")
    parser.add_argument("--collection-name", help="集合名称")

    parser.add_argument(
        "--action",
        choices=[
            "insert",
            "search",
            "both",
            "delete",
            "update",
            "get",
            "stats",
            "list-collections",
            "drop-collection",
            "clear",
        ],
        default="both",
        help="执行的操作",
    )
    parser.add_argument("--query", default="什么是向量数据库？", help="搜索查询文本")
    parser.add_argument("--top-k", type=int, default=None, help="返回最相似的 k 个结果")
    parser.add_argument("--doc-id", type=int, help="文档 ID（用于 delete/update/get）")
    parser.add_argument("--doc-ids", help="文档 ID 列表，逗号分隔（用于批量删除）")
    parser.add_argument("--text", help="文档文本（用于 update）")

    return parser.parse_args()


def _build_settings(args: argparse.Namespace) -> MilvusSettings:
    settings = MilvusSettings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.collection_name:
        settings.collection_name = args.collection_name
    return settings


def _maybe_vectorizer(actions: List[str]) -> Optional[TextVectorizer]:
    """仅在需要向量化时加载模型，减少无关操作的开销。"""
    if any(act in actions for act in ["insert", "search", "both", "update"]):
        return TextVectorizer()
    return None


def run_action(args: argparse.Namespace):
    settings = _build_settings(args)
    client = MilvusClient(settings=settings)

    vectorizer = _maybe_vectorizer([args.action])

    try:
        client.connect()

        need_collection = args.action not in ["list-collections", "drop-collection"]
        if need_collection:
            client.create_collection(dimension=(vectorizer.get_dimension() if vectorizer else settings.dimension))

        if args.action in ["insert", "both"]:
            # 示例文档
            documents = [
                "Python是一种高级编程语言，广泛用于数据科学和机器学习。",
                "Milvus是一个开源的向量数据库，专为AI应用设计。",
                "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
                "向量数据库可以高效地存储和检索高维向量数据。",
                "自然语言处理是计算机科学和人工智能的一个领域。",
                "深度学习使用神经网络来模拟人脑的学习过程。",
                "数据科学结合了统计学、编程和领域专业知识来分析数据。",
                "相似性搜索是向量数据库的核心功能之一。",
            ]
            if not vectorizer:
                vectorizer = TextVectorizer()
            count = ingest.insert_texts(client, vectorizer, documents)
            logger.info("插入完成，数量: %s", count)

        if args.action in ["search", "both"]:
            if not vectorizer:
                vectorizer = TextVectorizer()
            top_k = args.top_k or settings.top_k_default
            results = search_service.search_texts(client, vectorizer, args.query, top_k=top_k)
            logger.info("搜索结果数量: %d", len(results))
            for idx, res in enumerate(results, 1):
                logger.info("[%d] ID: %s, score: %.4f, distance: %.4f, text: %s", idx, res["id"], res["score"], res["distance"], res["text"])

        if args.action == "delete":
            if args.doc_ids:
                ids = [int(i.strip()) for i in args.doc_ids.split(",") if i.strip()]
                ingest.delete_by_ids(client, ids)
            elif args.doc_id is not None:
                ingest.delete_by_ids(client, [args.doc_id])
            else:
                raise ValueError("delete 操作需要 --doc-id 或 --doc-ids")

        if args.action == "update":
            if args.doc_id is None or not args.text:
                raise ValueError("update 操作需要同时提供 --doc-id 和 --text")
            if not vectorizer:
                vectorizer = TextVectorizer()
            ingest.update_text(client, vectorizer, args.doc_id, args.text)

        if args.action == "get":
            if args.doc_id is None:
                raise ValueError("get 操作需要 --doc-id")
            doc = ingest.get_by_id(client, args.doc_id)
            if doc:
                logger.info("文档: %s", doc)
            else:
                logger.info("未找到文档 ID: %s", args.doc_id)

        if args.action == "stats":
            stats = client.get_collection_stats()
            logger.info("集合统计: %s", stats)

        if args.action == "list-collections":
            names = client.list_collections()
            logger.info("集合列表 (%d): %s", len(names), names)

        if args.action == "drop-collection":
            client.drop_collection(args.collection_name)

        if args.action == "clear":
            client.clear_collection(args.collection_name)

    except Exception as exc:
        logger.exception("操作失败: %s", exc)
        sys.exit(1)
    finally:
        client.disconnect()


def main():
    args = parse_args()
    run_action(args)


if __name__ == "__main__":
    main()

