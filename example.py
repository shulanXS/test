"""
快速示例脚本
使用新的包结构 (milvus_mvp) 进行文档相似性搜索
"""
from milvus_mvp import MilvusClient, TextVectorizer, MilvusSettings


def main():
    print("=" * 60)
    print("Milvus文档相似性搜索 - 快速示例")
    print("=" * 60)
    
    # 初始化
    settings = MilvusSettings(host="localhost", port=19530)
    client = MilvusClient(settings=settings)
    vectorizer = TextVectorizer()
    
    try:
        # 连接Milvus
        print("\n1. 连接Milvus...")
        client.connect()
        
        # 创建集合
        print("\n2. 创建集合...")
        client.create_collection(dimension=vectorizer.get_dimension())
        
        # 准备文档
        print("\n3. 准备文档数据...")
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
        print(f"   准备插入 {len(documents)} 条文档")
        
        # 向量化
        print("\n4. 向量化文档...")
        embeddings = vectorizer.encode(documents)
        print(f"   向量维度: {len(embeddings[0])}")
        
        # 插入文档
        print("\n5. 插入文档到Milvus...")
        client.insert_documents(documents, embeddings)
        
        # 显示统计信息
        stats = client.get_collection_stats()
        print(f"\n   集合统计: {stats['num_entities']} 条文档")
        
        # 执行搜索
        print("\n6. 执行相似性搜索...")
        queries = [
            "什么是向量数据库？",
            "机器学习是什么？",
            "Python用于什么？"
        ]
        
        for query in queries:
            print(f"\n   查询: '{query}'")
            query_embedding = vectorizer.encode([query])[0]
            results = client.search(query_embedding, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. [相似度: {result['score']:.4f}] {result['text']}")
        
        print("\n" + "=" * 60)
        print("示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

