# Milvus MVP 项目架构与流程详解

本文档详细讲解 Milvus MVP 项目的整体架构、模块职责、执行流程和设计思路，帮助开发者快速理解项目结构和运行逻辑。

---

## 一、项目概述

### 1.1 项目定位

这是一个**最小可行产品（MVP）**，专注于实现向量数据库 Milvus 的**最基础 CRUD 操作**和**相似性搜索**功能。项目采用**企业级最佳实践**的代码结构，但功能范围保持最小化，便于学习和扩展。

### 1.2 核心功能

- **连接管理**：连接/断开 Milvus 服务器
- **集合管理**：创建、删除、清空、列出集合
- **文档插入**：将文本向量化后存入 Milvus
- **文档查询**：按 ID 查询单个或批量文档
- **文档更新**：更新文档内容和向量
- **文档删除**：单个或批量删除文档
- **相似性搜索**：基于向量相似度检索相关文档
- **统计信息**：查看集合中的文档数量

---

## 二、项目结构

### 2.1 目录布局

```
Milvus-MVP/
├── app.py                    # 应用入口（转发到 milvus_mvp.cli）
├── example.py                # 快速示例脚本
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖列表
│
└── milvus_mvp/              # 主包目录
    ├── __init__.py          # 包初始化，导出核心类
    ├── config.py            # 配置管理（MilvusSettings）
    ├── log.py               # 日志工具（get_logger）
    ├── vectorizer.py        # 文本向量化（TextVectorizer）
    ├── client.py            # Milvus 客户端封装（MilvusClient）
    ├── cli.py               # 命令行入口（参数解析、调度）
    │
    └── services/            # 业务服务层
        ├── __init__.py
        ├── ingest.py        # 文档插入/更新/删除服务
        └── search.py        # 搜索服务
```

### 2.2 架构分层

项目采用**分层架构**，从下到上分为：

1. **配置层**（`config.py`）：集中管理所有配置参数
2. **基础设施层**（`log.py`）：提供日志功能
3. **领域层**（`client.py`, `vectorizer.py`）：封装 Milvus 和向量化的核心能力
4. **服务层**（`services/`）：组合领域层能力，实现业务用例
5. **接口层**（`cli.py`）：命令行接口，解析参数并调用服务层

---

## 三、核心模块详解

### 3.1 配置管理（config.py）

**职责**：统一管理所有配置参数，支持环境变量覆盖。

**设计思路**：
- 使用 `dataclass` 定义配置类 `MilvusSettings`
- 每个字段都有默认值，可通过环境变量覆盖
- 提供便捷方法（如 `index_params()`, `search_params()`）生成 Milvus 所需的参数字典

**关键配置项**：
- **连接配置**：`host`（服务器地址）、`port`（端口）
- **集合配置**：`collection_name`（集合名）、`dimension`（向量维度）、`max_length`（文本最大长度）
- **索引配置**：`metric_type`（距离度量方式）、`index_type`（索引类型）、`index_nlist`（索引参数）
- **搜索配置**：`search_nprobe`（搜索参数）、`top_k_default`（默认返回数量）

**优先级**：命令行参数 > 环境变量 > 默认值

### 3.2 日志管理（log.py）

**职责**：提供统一的日志记录功能。

**设计思路**：
- 使用 Python 标准库 `logging`
- 日志级别可通过环境变量 `LOG_LEVEL` 控制（默认 INFO）
- 统一日志格式：`时间戳 [级别] 模块名 - 消息内容`
- 每个模块通过 `get_logger(__name__)` 获取自己的 logger

**使用场景**：
- 记录连接状态、操作结果、错误信息
- 便于调试和问题排查

### 3.3 向量化器（vectorizer.py）

**职责**：将文本转换为向量（embedding）。

**设计思路**：
- 封装 `sentence-transformers` 库
- 默认使用 `all-MiniLM-L6-v2` 模型（384 维，速度快）
- 支持批量编码，提高效率
- 提供 `get_dimension()` 方法获取向量维度

**工作流程**：
1. 初始化时加载模型（首次运行会下载，约 90MB）
2. `encode()` 方法接收文本列表，返回向量列表
3. 向量维度固定（如 384），必须与 Milvus 集合的维度一致

**注意事项**：
- 模型加载较慢（首次约 5-10 秒），后续调用很快
- 向量维度必须与 Milvus 集合的 `dimension` 配置一致

### 3.4 Milvus 客户端（client.py）

**职责**：封装所有 Milvus 数据库操作。

**核心类**：`MilvusClient`

**主要方法**：

#### 连接管理
- `connect()`：连接到 Milvus 服务器（幂等操作）
- `disconnect()`：断开连接
- `is_connected()`：检查连接状态

#### 集合管理
- `create_collection()`：创建集合（如果已存在则返回现有集合）
- `get_collection()`：获取集合对象
- `list_collections()`：列出所有集合
- `delete_collection()`：删除集合
- `clear_collection()`：清空集合数据（保留结构）
- `get_collection_stats()`：获取集合统计信息

#### CRUD 操作
- `insert_documents()`：插入文档（文本 + 向量）
- `get_document()`：按 ID 查询单个文档
- `query_by_ids()`：批量查询文档
- `update_document()`：更新文档（先删除再插入，注意会生成新 ID）
- `delete_document()`：删除单个文档
- `delete_documents()`：批量删除文档

#### 搜索操作
- `search()`：向量相似性搜索，返回最相似的 k 个结果

**设计特点**：
- 所有方法都通过 `get_collection()` 获取集合对象，确保集合存在
- 搜索前自动调用 `collection.load()`，确保数据加载到内存
- 插入/删除后自动调用 `collection.flush()`，确保数据持久化
- 使用配置对象（`MilvusSettings`）统一管理参数

### 3.5 服务层（services/）

**职责**：组合领域层能力，实现完整的业务用例。

#### ingest.py（文档管理服务）

提供高级文档操作，内部会调用向量化器和客户端：

- `insert_documents()`：接收文本列表，自动向量化后插入
- `update_document()`：接收文本，自动向量化后更新
- `delete_documents()`：批量删除文档

**优势**：调用方无需关心向量化细节，只需提供文本即可。

#### search.py（搜索服务）

提供搜索功能，内部会调用向量化器和客户端：

- `search_documents()`：接收查询文本，自动向量化后搜索

**优势**：调用方只需提供查询文本，无需手动向量化。

### 3.6 命令行接口（cli.py）

**职责**：解析命令行参数，组装各个组件，调用服务层完成操作。

**核心函数**：

#### `parse_args()`
- 定义所有命令行参数
- 使用 `argparse` 解析参数
- 返回 `argparse.Namespace` 对象

#### `_build_settings()`
- 根据命令行参数构建 `MilvusSettings`
- 优先级：命令行参数 > 环境变量 > 默认值

#### `run_action()`
- 根据 `action` 参数执行相应操作
- 按需创建 `MilvusClient` 和 `TextVectorizer`
- 调用 `services` 层完成业务逻辑
- 处理异常和资源清理

**设计特点**：
- CLI 层不直接操作 Milvus，只做调度
- 向量化器按需加载（只在需要时创建，避免不必要的模型加载）
- 统一的错误处理和日志记录

---

## 四、执行流程详解

### 4.1 整体流程

```
用户执行命令
    ↓
app.py（入口）
    ↓
cli.py（解析参数）
    ↓
构建配置（MilvusSettings）
    ↓
创建客户端（MilvusClient）
    ↓
连接 Milvus（connect）
    ↓
创建/获取集合（create_collection）
    ↓
按需创建向量化器（TextVectorizer）
    ↓
调用服务层（services.ingest / services.search）
    ↓
服务层调用客户端和向量化器
    ↓
执行具体操作（插入/搜索/删除等）
    ↓
返回结果，记录日志
    ↓
断开连接（disconnect）
```

### 4.2 插入文档流程（action=insert）

1. **解析参数**：`cli.py` 解析 `--action insert`
2. **构建配置**：创建 `MilvusSettings`，读取 host/port/collection 等配置
3. **创建客户端**：实例化 `MilvusClient`，传入配置
4. **连接数据库**：调用 `client.connect()`，建立与 Milvus 的连接
5. **创建向量化器**：实例化 `TextVectorizer`，加载模型（首次较慢）
6. **创建集合**：调用 `client.create_collection()`，传入向量维度
   - 如果集合已存在，直接返回现有集合
   - 如果不存在，创建新集合并建立索引
7. **准备数据**：准备示例文档列表（硬编码在 `cli.py` 中）
8. **向量化**：调用 `vectorizer.encode()`，将文本转换为向量
9. **插入数据**：调用 `services.ingest.insert_documents()`
   - 内部调用 `client.insert_documents()`，传入文本和向量
   - 客户端执行 `collection.insert()` 和 `collection.flush()`
10. **显示统计**：调用 `client.get_collection_stats()`，显示文档数量
11. **断开连接**：调用 `client.disconnect()`

### 4.3 搜索流程（action=search）

1. **解析参数**：解析 `--action search --query "查询文本" --top-k 3`
2. **构建配置和客户端**：同插入流程
3. **连接数据库**：同插入流程
4. **创建向量化器**：同插入流程（如果尚未创建）
5. **获取集合**：调用 `client.get_collection()`，确保集合存在
6. **向量化查询**：调用 `vectorizer.encode([query])`，将查询文本转换为向量
7. **执行搜索**：调用 `services.search.search_documents()`
   - 内部调用 `client.search()`，传入查询向量和 top_k
   - 客户端执行 `collection.load()`（确保数据加载）
   - 调用 `collection.search()`，使用配置的搜索参数
   - 返回相似文档列表（包含 id、text、distance、score）
8. **格式化结果**：将搜索结果格式化为易读格式
9. **显示结果**：输出到控制台
10. **断开连接**：同插入流程

### 4.4 删除文档流程（action=delete）

1. **解析参数**：解析 `--action delete --doc-id 1` 或 `--doc-ids "1,2,3"`
2. **构建配置和客户端**：同插入流程
3. **连接数据库**：同插入流程
4. **获取集合**：确保集合存在
5. **执行删除**：
   - 单个删除：调用 `client.delete_document(doc_id)`
   - 批量删除：解析 `--doc-ids`，调用 `client.delete_documents(doc_ids)`
6. **构建删除表达式**：如 `id == 1` 或 `id in [1,2,3]`
7. **执行删除**：调用 `collection.delete(expr)` 和 `collection.flush()`
8. **显示统计**：显示删除后的文档数量
9. **断开连接**：同插入流程

### 4.5 更新文档流程（action=update）

1. **解析参数**：解析 `--action update --doc-id 1 --text "新文本"`
2. **构建配置和客户端**：同插入流程
3. **连接数据库**：同插入流程
4. **创建向量化器**：需要向量化新文本
5. **获取集合**：确保集合存在
6. **检查文档存在**：调用 `client.get_document(doc_id)`，确认文档存在
7. **向量化新文本**：调用 `vectorizer.encode([new_text])`
8. **执行更新**：调用 `client.update_document()`
   - 先删除旧文档：`collection.delete(expr=f"id == {doc_id}")`
   - 再插入新文档：`collection.insert([new_text], [new_embedding])`
   - **注意**：由于使用 `auto_id=True`，新文档会有新的 ID
9. **刷新数据**：调用 `collection.flush()`
10. **断开连接**：同插入流程

### 4.6 集合管理流程

#### 列出集合（action=list-collections）

1. **解析参数**：解析 `--action list-collections`
2. **构建配置和客户端**：同插入流程
3. **连接数据库**：同插入流程
4. **执行列出**：调用 `client.list_collections()`
   - 内部调用 `utility.list_collections()`
5. **显示结果**：格式化输出集合列表
6. **断开连接**：同插入流程

#### 删除集合（action=drop-collection）

1. **解析参数**：解析 `--action drop-collection --collection-name "my_collection"`
2. **构建配置和客户端**：同插入流程
3. **连接数据库**：同插入流程
4. **执行删除**：调用 `client.delete_collection()`
   - 内部调用 `utility.drop_collection(name)`
5. **断开连接**：同插入流程

---

## 五、设计模式与最佳实践

### 5.1 分层架构

**优势**：
- **职责清晰**：每层只负责自己的事情
- **易于测试**：可以单独测试每一层
- **易于扩展**：新增功能只需在相应层添加代码

**各层职责**：
- **配置层**：管理参数
- **领域层**：封装核心能力
- **服务层**：组合领域层，实现业务用例
- **接口层**：对外提供接口

### 5.2 依赖注入

**实现方式**：
- `MilvusClient` 接收 `MilvusSettings` 作为参数
- `services` 层接收 `MilvusClient` 和 `TextVectorizer` 作为参数
- 便于测试和替换实现

### 5.3 配置管理

**三层优先级**：
1. 命令行参数（最高优先级）
2. 环境变量（中等优先级）
3. 默认值（最低优先级）

**优势**：
- 开发环境使用默认值
- 生产环境通过环境变量配置
- 临时测试通过命令行参数覆盖

### 5.4 幂等性设计

**关键操作**：
- `connect()`：多次调用不会重复创建连接
- `create_collection()`：集合已存在时直接返回，不报错
- `disconnect()`：已断开时不会报错

**优势**：
- 调用方无需检查状态
- 代码更简洁、更安全

### 5.5 资源管理

**实现方式**：
- 使用 `try-finally` 确保连接断开
- 向量化器按需创建（避免不必要的模型加载）
- 集合自动加载（搜索前自动 `load()`）

### 5.6 错误处理

**策略**：
- 领域层抛出异常，不捕获
- 接口层捕获异常，记录日志，返回错误码
- 使用标准异常类型（`ValueError`, `ConnectionError` 等）

---

## 六、数据流转

### 6.1 插入数据流

```
文本列表（List[str]）
    ↓
TextVectorizer.encode()
    ↓
向量列表（List[List[float]]）
    ↓
MilvusClient.insert_documents(texts, embeddings)
    ↓
Collection.insert(entities)
    ↓
Collection.flush()
    ↓
Milvus 数据库（持久化存储）
```

### 6.2 搜索数据流

```
查询文本（str）
    ↓
TextVectorizer.encode([query])
    ↓
查询向量（List[float]）
    ↓
MilvusClient.search(query_embedding, top_k)
    ↓
Collection.load()（加载数据到内存）
    ↓
Collection.search(data=[query_embedding], ...)
    ↓
搜索结果（List[dict]：id, text, distance, score）
    ↓
格式化输出（CLI 层）
```

### 6.3 删除数据流

```
文档 ID（int 或 List[int]）
    ↓
构建删除表达式（如 "id == 1" 或 "id in [1,2,3]"）
    ↓
Collection.delete(expr)
    ↓
Collection.flush()
    ↓
Milvus 数据库（数据被删除）
```

---

## 七、关键概念解释

### 7.1 集合（Collection）

**定义**：Milvus 中存储数据的逻辑单元，类似于关系数据库中的表。

**组成**：
- **字段（Field）**：定义数据结构
  - `id`：主键字段（INT64，自动生成）
  - `text`：文本字段（VARCHAR，最大长度 5000）
  - `embedding`：向量字段（FLOAT_VECTOR，维度 384）
- **Schema**：字段定义的集合
- **索引**：加速搜索的数据结构（IVF_FLAT）

### 7.2 向量（Vector/Embedding）

**定义**：文本经过向量化模型处理后得到的数值数组。

**特点**：
- 维度固定（如 384 维）
- 相似文本的向量在空间中距离较近
- 用于相似性搜索

### 7.3 索引（Index）

**定义**：加速向量搜索的数据结构。

**本项目使用**：
- **类型**：IVF_FLAT（倒排索引 + 扁平存储）
- **参数**：nlist=128（聚类中心数量）
- **度量方式**：L2（欧式距离）

### 7.4 搜索参数

**nprobe**：搜索时检查的聚类中心数量（默认 10）
- 值越大，精度越高，速度越慢
- 值越小，速度越快，精度可能降低

**top_k**：返回最相似的 k 个结果（默认 5）

### 7.5 距离与相似度

**距离（distance）**：向量之间的欧式距离，值越小越相似。

**相似度（score）**：通过公式 `1 / (1 + distance)` 转换得到，值越大越相似（0-1 之间）。

---

## 八、扩展指南

### 8.1 添加新操作

1. **在 `client.py` 添加底层方法**（如果需要新的 Milvus 操作）
2. **在 `services/` 添加服务方法**（如果需要组合多个操作）
3. **在 `cli.py` 添加命令行参数和 action**（如果需要 CLI 支持）

### 8.2 更换向量化模型

1. **修改 `vectorizer.py`**：更改 `model_name` 参数
2. **修改 `config.py`**：更新 `dimension` 默认值
3. **重新创建集合**：删除旧集合，创建新集合（维度必须匹配）

### 8.3 优化性能

1. **调整索引参数**：修改 `index_nlist` 和 `search_nprobe`
2. **批量操作**：使用批量插入/删除，减少网络往返
3. **连接池**：生产环境使用连接池管理连接

---

## 九、常见问题

### 9.1 为什么更新文档会生成新 ID？

**原因**：集合 Schema 使用 `auto_id=True`，插入时 Milvus 自动生成 ID。

**解决方案**：如果需要保持 ID 不变，修改 Schema 使用 `auto_id=False`，手动指定 ID。

### 9.2 为什么搜索前要调用 `load()`？

**原因**：Milvus 的数据默认存储在磁盘，`load()` 将数据加载到内存，加速搜索。

**注意**：`load()` 是幂等的，多次调用不会重复加载。

### 9.3 如何提高搜索精度？

**方法**：
1. 增加 `search_nprobe` 参数（检查更多聚类中心）
2. 使用更高质量的向量化模型（如 `all-mpnet-base-v2`）
3. 调整索引类型（如使用 HNSW）

### 9.4 如何批量导入大量文档？

**方法**：
1. 准备文本列表
2. 批量向量化（`vectorizer.encode(texts)`）
3. 批量插入（`client.insert_documents(texts, embeddings)`）
4. 注意：单次插入建议不超过 10 万条

---

## 十、总结

本项目采用**分层架构**和**企业级最佳实践**，实现了 Milvus 向量数据库的基础 CRUD 和搜索功能。核心设计理念是：

1. **职责分离**：每层只负责自己的事情
2. **配置集中**：统一管理所有配置参数
3. **按需加载**：避免不必要的资源消耗
4. **幂等设计**：操作可重复执行，不会出错
5. **错误处理**：统一的异常处理和日志记录

通过清晰的模块划分和规范的代码结构，项目既保持了 MVP 的简洁性，又具备了良好的可扩展性和可维护性。

