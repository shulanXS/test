# Milvus文档相似性搜索MVP

这是一个基于Milvus向量数据库的文档相似性搜索最小可行产品（MVP）。

## 功能特性

- 📚 文档向量化存储
- 🔍 相似性搜索
- ✏️ 文档更新和删除
- 📋 文档查询和管理
- 🗂️ 集合管理（创建、删除、清空、列表）
- 🚀 快速检索
- 💡 简单易用

## 技术栈

- **Milvus**: 向量数据库
- **sentence-transformers**: 文本向量化
- **Python 3.7+**

## 安装步骤

### 1. 安装Milvus

#### 使用Docker（推荐）

```bash
# 下载Milvus Docker Compose文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动Milvus
docker-compose up -d

# 检查状态
docker-compose ps
```

#### 使用pip安装（仅客户端）

```bash
pip install pymilvus
```

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

注意：首次运行时会自动下载sentence-transformers模型（约90MB）。

## 使用方法

### 基本使用

```bash
# 插入文档并搜索（默认操作）
python app.py

# 只插入文档
python app.py --action insert

# 只搜索
python app.py --action search --query "你的查询文本"

# 自定义搜索参数
python app.py --action search --query "机器学习" --top-k 3

# 连接到远程Milvus服务器
python app.py --host 192.168.1.100 --port 19530
```

### CRUD操作

```bash
# 删除单个文档
python app.py --action delete --doc-id 1

# 批量删除文档
python app.py --action delete --doc-ids "1,2,3"

# 更新文档
python app.py --action update --doc-id 1 --text "新的文档内容"

# 查询单个文档
python app.py --action get --doc-id 1

# 查看集合统计信息
python app.py --action stats
```

### 集合管理

```bash
# 列出所有集合
python app.py --action list-collections

# 删除集合
python app.py --action drop-collection --collection-name "my_collection"

# 清空集合（保留集合结构，删除所有数据）
python app.py --action clear --collection-name "my_collection"
```

### 命令行参数

- `--host`: Milvus服务器地址（默认: localhost）
- `--port`: Milvus服务器端口（默认: 19530）
- `--action`: 执行的操作，可选值:
  - `insert`: 插入文档
  - `search`: 搜索文档
  - `both`: 插入并搜索（默认）
  - `delete`: 删除文档
  - `update`: 更新文档
  - `get`: 查询单个文档
  - `stats`: 显示统计信息
  - `list-collections`: 列出所有集合
  - `drop-collection`: 删除集合
  - `clear`: 清空集合
- `--query`: 搜索查询文本（默认: "什么是向量数据库？"）
- `--top-k`: 返回最相似的k个结果（默认: 5）
- `--doc-id`: 文档ID（用于delete、update、get操作）
- `--doc-ids`: 文档ID列表，用逗号分隔（用于批量删除）
- `--text`: 文档文本（用于update操作）
- `--collection-name`: 集合名称

## 项目结构

```
.
├── app.py                    # 入口脚本（转发到 milvus_mvp.cli）
├── example.py                # 快速示例
├── milvus_mvp/               # 包化后的核心代码
│   ├── __init__.py
│   ├── cli.py                # CLI 入口（argparse）
│   ├── client.py             # Milvus 客户端封装（CRUD/集合/搜索）
│   ├── config.py             # 配置（可用环境变量覆盖）
│   ├── log.py                # 日志配置
│   ├── vectorizer.py         # 文本向量化
│   └── services/             # 业务层：组合向量化 + Milvus 操作
│       ├── ingest.py         # 插入/更新/删除/查询
│       ├── search.py         # 搜索
│       └── __init__.py
├── requirements.txt          # Python依赖
└── README.md                 # 项目说明
```

## 代码示例

### 插入文档

```python
from milvus_mvp import MilvusClient, TextVectorizer, MilvusSettings

settings = MilvusSettings()
client = MilvusClient(settings=settings)
vectorizer = TextVectorizer()

client.connect()
client.create_collection(dimension=vectorizer.get_dimension())

documents = ["文档1", "文档2", "文档3"]
embeddings = vectorizer.encode(documents)
client.insert_documents(documents, embeddings)
```

### 搜索相似文档

```python
query = "你的查询文本"
query_embedding = vectorizer.encode([query])[0]
results = client.search(query_embedding, top_k=5)

for result in results:
    print(f"相似度: {result['score']:.4f}")
    print(f"文档: {result['text']}")
```

### 删除文档

```python
# 删除单个文档
client.delete_document(doc_id=1)

# 批量删除文档
client.delete_documents(doc_ids=[1, 2, 3])
```

### 更新文档

```python
new_text = "更新后的文档内容"
new_embedding = vectorizer.encode([new_text])[0]
client.update_document(doc_id=1, text=new_text, embedding=new_embedding)
```

### 查询文档

```python
# 查询单个文档
doc = client.get_document(doc_id=1)
if doc:
    print(f"ID: {doc['id']}")
    print(f"文本: {doc['text']}")

# 批量查询文档
docs = client.query_by_ids(doc_ids=[1, 2, 3])
```

### 集合管理

```python
# 列出所有集合
collections = client.list_collections()
print(collections)

# 删除集合
client.delete_collection("my_collection")

# 清空集合
client.clear_collection("my_collection")

# 获取集合统计信息
stats = client.get_collection_stats()
print(f"文档数量: {stats['num_entities']}")

# 检查连接状态
if client.is_connected():
    print("已连接到Milvus")
```

## 模型说明

默认使用 `all-MiniLM-L6-v2` 模型：
- 向量维度: 384
- 速度快，适合快速原型开发
- 支持中文和英文

如需更高精度，可在 `vectorizer.py` 中修改为 `all-mpnet-base-v2`（768维）。

## 常见问题

### Q: 如何连接到远程Milvus服务器？

A: 使用 `--host` 和 `--port` 参数：
```bash
python app.py --host your-server-ip --port 19530
```

### Q: 如何修改向量维度？

A: 在 `vectorizer.py` 中修改模型名称，或在 `milvus_client.py` 的 `create_collection` 方法中指定维度。

### Q: Milvus连接失败怎么办？

A: 
1. 确认Milvus服务正在运行：`docker-compose ps`
2. 检查端口是否正确（默认19530）
3. 确认防火墙设置

## 已实现功能

- ✅ 文档插入和向量化存储
- ✅ 相似性搜索
- ✅ 文档更新和删除（单个和批量）
- ✅ 文档查询（单个和批量）
- ✅ 集合管理（创建、删除、清空、列表）
- ✅ 连接状态检查
- ✅ 集合统计信息

## 下一步扩展

- [ ] 支持批量文档导入（从文件）
- [ ] 添加Web API接口
- [ ] 支持更多向量化模型
- [ ] 性能优化和索引调优
- [ ] 添加数据导出功能
- [ ] 支持条件查询和过滤

## 许可证

MIT License

## 参考资源

- [Milvus官方文档](https://milvus.io/docs)
- [sentence-transformers文档](https://www.sbert.net/)
- [PyMilvus文档](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)

