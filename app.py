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
        print(f"{i}. [相似度: {result['score']:.4f}, 距离: {result['distance']:.4f}]")
        print(f"   {result['text']}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Milvus文档相似性搜索MVP")
    parser.add_argument("--host", default="localhost", help="Milvus服务器地址")
    parser.add_argument("--port", type=int, default=19530, help="Milvus服务器端口")
    parser.add_argument("--action", choices=["insert", "search", "both"], default="both",
                       help="执行的操作: insert(插入), search(搜索), both(两者)")
    parser.add_argument("--query", default="什么是向量数据库？", help="搜索查询文本")
    parser.add_argument("--top-k", type=int, default=5, help="返回最相似的k个结果")
    
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
        
        # 创建集合（如果不存在）
        client.create_collection(dimension=vectorizer.get_dimension())
        
        # 执行操作
        if args.action in ["insert", "both"]:
            insert_documents(client, vectorizer)
        
        if args.action in ["search", "both"]:
            search_documents(client, vectorizer, args.query, args.top_k)
        
        print("\n" + "=" * 50)
        print("操作完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

