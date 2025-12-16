"""
Milvus文档相似性搜索MVP应用
主程序入口
"""
from milvus_client import MilvusClient
from vectorizer import TextVectorizer
import argparse
import sys


def insert_documents(client: MilvusClient, vectorizer: TextVectorizer):
    """插入示例文档"""
    print("\n=== 插入文档 ===")
    
    # 示例文档
    documents = [
        "Python是一种高级编程语言，广泛用于数据科学和机器学习。",
        "Milvus是一个开源的向量数据库，专为AI应用设计。",
        "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
        "向量数据库可以高效地存储和检索高维向量数据。",
        "自然语言处理是计算机科学和人工智能的一个领域。",
        "深度学习使用神经网络来模拟人脑的学习过程。",
        "数据科学结合了统计学、编程和领域专业知识来分析数据。",
        "相似性搜索是向量数据库的核心功能之一。"
    ]
    
    # 向量化
    print("正在向量化文档...")
    embeddings = vectorizer.encode(documents)
    
    # 插入到Milvus
    client.insert_documents(documents, embeddings)
    
    # 显示统计信息
    stats = client.get_collection_stats()
    print(f"集合统计: {stats['num_entities']} 条文档")


def search_documents(client: MilvusClient, vectorizer: TextVectorizer, query: str, top_k: int = 5):
    """搜索相似文档"""
    print(f"\n=== 搜索查询: '{query}' ===")
    
    # 向量化查询
    query_embedding = vectorizer.encode([query])[0]
    
    # 搜索
    results = client.search(query_embedding, top_k=top_k)
    
    # 显示结果
    print(f"\n找到 {len(results)} 个相似文档:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [ID: {result['id']}, 相似度: {result['score']:.4f}, 距离: {result['distance']:.4f}]")
        print(f"   {result['text']}\n")


def delete_document(client: MilvusClient, doc_id: int):
    """删除单个文档"""
    print(f"\n=== 删除文档 ID: {doc_id} ===")
    client.delete_document(doc_id)
    stats = client.get_collection_stats()
    print(f"当前集合统计: {stats['num_entities']} 条文档")


def delete_documents_batch(client: MilvusClient, doc_ids: list):
    """批量删除文档"""
    print(f"\n=== 批量删除文档 ===")
    print(f"要删除的文档ID: {doc_ids}")
    deleted_count = client.delete_documents(doc_ids)
    stats = client.get_collection_stats()
    print(f"当前集合统计: {stats['num_entities']} 条文档")


def update_document(client: MilvusClient, vectorizer: TextVectorizer, doc_id: int, text: str):
    """更新文档"""
    print(f"\n=== 更新文档 ID: {doc_id} ===")
    print(f"新文本: {text}")
    
    # 向量化新文本
    embedding = vectorizer.encode([text])[0]
    
    # 更新文档
    client.update_document(doc_id, text, embedding)
    print("✓ 文档更新完成")


def get_document(client: MilvusClient, doc_id: int):
    """查询单个文档"""
    print(f"\n=== 查询文档 ID: {doc_id} ===")
    doc = client.get_document(doc_id)
    
    if doc:
        print(f"ID: {doc['id']}")
        print(f"文本: {doc['text']}")
        print(f"向量维度: {len(doc['embedding'])}")
    else:
        print("文档不存在")


def list_collections(client: MilvusClient):
    """列出所有集合"""
    print("\n=== 所有集合 ===")
    collections = client.list_collections()
    
    if collections:
        print(f"找到 {len(collections)} 个集合:")
        for i, name in enumerate(collections, 1):
            print(f"  {i}. {name}")
    else:
        print("没有找到任何集合")


def drop_collection(client: MilvusClient, collection_name: str = None):
    """删除集合"""
    print(f"\n=== 删除集合 ===")
    if collection_name:
        client.collection_name = collection_name
    client.delete_collection()


def clear_collection(client: MilvusClient, collection_name: str = None):
    """清空集合"""
    print(f"\n=== 清空集合 ===")
    if collection_name:
        client.collection_name = collection_name
    client.clear_collection()


def show_stats(client: MilvusClient):
    """显示集合统计信息"""
    print("\n=== 集合统计信息 ===")
    stats = client.get_collection_stats()
    print(f"集合名称: {stats['collection_name']}")
    print(f"文档数量: {stats['num_entities']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Milvus文档相似性搜索MVP")
    parser.add_argument("--host", default="localhost", help="Milvus服务器地址")
    parser.add_argument("--port", type=int, default=19530, help="Milvus服务器端口")
    parser.add_argument("--action", 
                       choices=["insert", "search", "both", "delete", "update", "get", 
                               "list-collections", "drop-collection", "clear", "stats"],
                       default="both",
                       help="执行的操作")
    parser.add_argument("--query", default="什么是向量数据库？", help="搜索查询文本")
    parser.add_argument("--top-k", type=int, default=5, help="返回最相似的k个结果")
    parser.add_argument("--doc-id", type=int, help="文档ID（用于delete、update、get操作）")
    parser.add_argument("--doc-ids", help="文档ID列表，用逗号分隔（用于批量删除）")
    parser.add_argument("--text", help="文档文本（用于update操作）")
    parser.add_argument("--collection-name", help="集合名称")
    
    args = parser.parse_args()
    
    # 初始化
    print("=" * 50)
    print("Milvus文档相似性搜索MVP")
    print("=" * 50)
    
    client = MilvusClient(host=args.host, port=args.port)
    vectorizer = TextVectorizer()
    
    try:
        # 连接Milvus
        client.connect()
        
        # 根据操作类型决定是否需要创建集合
        need_collection = args.action not in ["list-collections", "drop-collection"]
        
        if need_collection:
            if args.collection_name:
                client.collection_name = args.collection_name
            # 创建集合（如果不存在）
            client.create_collection(dimension=vectorizer.get_dimension())
        
        # 执行操作
        if args.action in ["insert", "both"]:
            insert_documents(client, vectorizer)
        
        if args.action in ["search", "both"]:
            search_documents(client, vectorizer, args.query, args.top_k)
        
        if args.action == "delete":
            if args.doc_ids:
                # 批量删除
                doc_ids = [int(id.strip()) for id in args.doc_ids.split(",")]
                delete_documents_batch(client, doc_ids)
            elif args.doc_id:
                # 单个删除
                delete_document(client, args.doc_id)
            else:
                print("✗ 请提供 --doc-id 或 --doc-ids 参数")
        
        if args.action == "update":
            if not args.doc_id or not args.text:
                print("✗ 请提供 --doc-id 和 --text 参数")
            else:
                update_document(client, vectorizer, args.doc_id, args.text)
        
        if args.action == "get":
            if not args.doc_id:
                print("✗ 请提供 --doc-id 参数")
            else:
                get_document(client, args.doc_id)
        
        if args.action == "list-collections":
            list_collections(client)
        
        if args.action == "drop-collection":
            drop_collection(client, args.collection_name)
        
        if args.action == "clear":
            clear_collection(client, args.collection_name)
        
        if args.action == "stats":
            show_stats(client)
        
        print("\n" + "=" * 50)
        print("操作完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

