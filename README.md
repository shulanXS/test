# Milvus文档相似性搜索MVP

这是一个基于Milvus向量数据库的文档相似性搜索最小可行产品（MVP）。

## 功能特性

- 📚 文档向量化存储
- 🔍 相似性搜索
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

### 命令行参数

- `--host`: Milvus服务器地址（默认: localhost）
- `--port`: Milvus服务器端口（默认: 19530）
- `--action`: 执行的操作，可选值: `insert`, `search`, `both`（默认: both）
- `--query`: 搜索查询文本（默认: "什么是向量数据库？"）
- `--top-k`: 返回最相似的k个结果（默认: 5）

## 项目结构

```
.
├── app.py              # 主应用脚本
├── milvus_client.py    # Milvus客户端封装
├── vectorizer.py       # 文本向量化模块
├── requirements.txt    # Python依赖
└── README.md          # 项目说明
```

## 代码示例

### 插入文档

```python
from milvus_client import MilvusClient
from vectorizer import TextVectorizer

client = MilvusClient()
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

## 下一步扩展

- [ ] 支持批量文档导入（从文件）
- [ ] 添加Web API接口
- [ ] 支持更多向量化模型
- [ ] 添加文档更新和删除功能
- [ ] 性能优化和索引调优

## 许可证

MIT License

## 参考资源

- [Milvus官方文档](https://milvus.io/docs)
- [sentence-transformers文档](https://www.sbert.net/)
- [PyMilvus文档](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)

