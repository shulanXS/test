"""
命令行入口模块（CLI）

作用可以理解为“总控台 / 调度中心”，负责：

- 解析命令行参数（`--action`, `--query`, `--doc-id` 等）
- 根据参数构造 Milvus 配置（`MilvusSettings`）
- 按需创建 Milvus 客户端（`MilvusClient`）和向量化器（`TextVectorizer`）
- 调用 service 层（`services.ingest` / `services.search`）完成真正业务逻辑

本模块**不直接操作 Milvus**，也不写复杂业务逻辑，只做“组装、调度、错误处理和日志”：

调用链大致是：

    命令行参数  →  parse_args() → run_action()
                 →  MilvusSettings / MilvusClient / TextVectorizer
                 →  services.ingest / services.search
                 →  client.MilvusClient（真正访问 Milvus）
"""

import argparse  # 解析命令行参数的标准库
import sys       # 用于退出程序并返回状态码
from typing import Optional, List

# 领域层：直接封装 Milvus 的 CRUD / 集合管理 / 搜索
from .client import MilvusClient
# 配置层：集中管理 host/port/collection/index/search 等参数，可由环境变量覆盖
from .config import MilvusSettings
# 日志：统一格式和日志级别
from .log import get_logger
# 向量化层：将文本转换为向量
from .vectorizer import TextVectorizer
# service 层：更高层级的“业务用例”（组合向量化 + Milvus 操作）
from .services import ingest, search as search_service

# 为当前模块（milvus_mvp.cli）创建一个 logger 实例，用于日志输出和记录，
# 方便在终端中看到清晰的、带模块名和级别的日志信息。
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    只负责定义参数和 help 文案，不做任何业务逻辑。
    返回 argparse.Namespace，后续交给 run_action 处理。
    """
    # `ArgumentParser` 会自动帮我们生成 `--help` 帮助文档
    parser = argparse.ArgumentParser(description="Milvus 文档相似性搜索 MVP（命令行工具）")

    # === 基础连接参数（如果不传，则使用 MilvusSettings 中的默认值 / 环境变量） ===
    parser.add_argument("--host", default=None, help="Milvus 服务器地址（可选，默认从环境变量/配置中读取）")
    parser.add_argument("--port", type=int, default=None, help="Milvus 服务器端口（可选）")
    parser.add_argument("--collection-name", help="集合名称（可选，覆盖默认集合名）")

    # === 选择要执行的动作：覆盖最基础的 CRUD + 搜索 + 集合管理 ===
    parser.add_argument(
        "--action",
        choices=[
            "insert",          # 插入示例文档
            "search",          # 仅搜索
            "both",            # 先插入示例文档，再搜索（demo 用）
            "delete",          # 删除文档（单个 / 批量）
            "update",          # 更新文档（文本 + 向量）
            "get",             # 按 ID 获取文档
            "stats",           # 查看当前集合统计信息
            "list-collections",# 列出所有集合
            "drop-collection", # 删除集合
            "clear",           # 清空集合数据
        ],
        default="both",
        help="执行的操作",
    )

    # === 搜索相关参数（只在 action=search/both 时有意义） ===
    parser.add_argument("--query", default="什么是向量数据库？", help="搜索查询文本")
    parser.add_argument("--top-k", type=int, default=None, help="返回最相似的 k 个结果（默认使用配置中的值）")

    # === 文档 ID / 文本相关参数（CRUD 操作中会用到） ===
    parser.add_argument("--doc-id", type=int, help="文档 ID（用于 delete/update/get）")
    parser.add_argument("--doc-ids", help="文档 ID 列表，逗号分隔（用于批量删除）")
    parser.add_argument("--text", help="文档文本（用于 update）")

    return parser.parse_args()


def _build_settings(args: argparse.Namespace) -> MilvusSettings:
    """
    根据命令行参数构建 MilvusSettings。

    逻辑：
    - 先用 MilvusSettings() 读默认值 / 环境变量
    - 再用命令行参数（如果提供）覆盖默认配置
    """
    # 先从环境变量 / 默认值构造一个 settings
    settings = MilvusSettings()

    # 命令行优先级高于环境变量 / 默认值
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.collection_name:
        settings.collection_name = args.collection_name

    return settings


def _maybe_vectorizer(actions: List[str]) -> Optional[TextVectorizer]:
    """
    根据动作是否需要向量化，懒加载 TextVectorizer。

    这样做的原因：
    - 向量化模型较大，加载开销高
    - 对于 list-collections / drop-collection 等操作，不需要加载模型
    """
    if any(act in actions for act in ["insert", "search", "both", "update"]):
        return TextVectorizer()
    return None


def run_action(args: argparse.Namespace):
    """
    根据解析好的参数执行对应操作。

    职责：
    - 初始化配置 (MilvusSettings)、客户端 (MilvusClient)、向量化器 (TextVectorizer)
    - 决定是否需要创建 / 加载集合
    - 调用 service 层（ingest / search_service）和 client 的基础 CRUD/统计/集合操作
    - 统一异常捕获和日志输出
    """
    # 1. 构建配置对象（合并环境变量和命令行参数）
    #    之后所有 MilvusClient 的行为（host/port/collection/index/search）都受这个 settings 控制
    settings = _build_settings(args)
    client = MilvusClient(settings=settings)

    # 2. 根据 action 决定是否预先加载向量化模型
    #    注意：有些操作（list-collections / drop-collection 等）完全不需要模型
    vectorizer = _maybe_vectorizer([args.action])

    try:
        # 3. 连接 Milvus（如果已经连接过，PyMilvus 会做幂等处理）
        client.connect()

        # 4. 除了极少数集合管理操作以外（比如列集合、删除集合），绝大多数操作都要依赖目标集合存在，所以下面要按需创建集合
        # need_collection 为 True 代表当前 action 不是 ["list-collections", "drop-collection"] 之一，即“需要集合”
        need_collection = args.action not in ["list-collections", "drop-collection"]
        if need_collection:
            # 这里有个细致的判断：
            #   - 如果已经有 vectorizer（即向量化器已加载），则优先选用该模型的输出向量维度
            #     保证集合和后续入库/检索的向量匹配（比如模型 all-mpnet-base-v2 输出是 768维，要创建 768维向量字段）。
            #   - 如果没有加载 vectorizer（实际上在 insert/search/both/update 时会加载，兜底策略），
            #     则退回用 settings.dimension（一般是配置文件或环境变量的默认维度，比如 384）。
            # 这样保证了：集合字段与后面所有插/查/向量维度都严格一致，不易出错。
            dim = vectorizer.get_dimension() if vectorizer else settings.dimension
            # create_collection 是幂等操作，如果集合已存在不会重复创建
            # 这里会按 dim 参数创建 embedding 向量字段，索引类型/主键/分区已封装在 MilvusClient 里
            client.create_collection(dimension=dim)

        # 5. insert / both：插入内置示例文档
        if args.action in ["insert", "both"]:
            # 示例文档（MVP 里直接写在代码里，真实项目里可以改为从文件/DB 加载）
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
            # 理论上这里一定需要向量化器，做一次兜底判断
            if not vectorizer:
                vectorizer = TextVectorizer()
            count = ingest.insert_texts(client, vectorizer, documents)
            logger.info("插入完成，数量: %s", count)

        # 6. search / both：执行向量搜索
        if args.action in ["search", "both"]:
            if not vectorizer:
                vectorizer = TextVectorizer()
            top_k = args.top_k or settings.top_k_default
            results = search_service.search_texts(client, vectorizer, args.query, top_k=top_k)
            logger.info("搜索结果数量: %d", len(results))
            for idx, res in enumerate(results, 1):
                logger.info(
                    "[%d] ID: %s, score: %.4f, distance: %.4f, text: %s",
                    idx,
                    res["id"],
                    res["score"],
                    res["distance"],
                    res["text"],
                )

        # 7. delete：单个或批量删除
        if args.action == "delete":
            if args.doc_ids:
                # 支持 --doc-ids "1,2,3"
                ids = [int(i.strip()) for i in args.doc_ids.split(",") if i.strip()]
                ingest.delete_by_ids(client, ids)
            elif args.doc_id is not None:
                ingest.delete_by_ids(client, [args.doc_id])
            else:
                raise ValueError("delete 操作需要 --doc-id 或 --doc-ids")

        # 8. update：更新指定 ID 的文档
        if args.action == "update":
            if args.doc_id is None or not args.text:
                raise ValueError("update 操作需要同时提供 --doc-id 和 --text")
            if not vectorizer:
                vectorizer = TextVectorizer()
            ingest.update_text(client, vectorizer, args.doc_id, args.text)

        # 9. get：按 ID 获取文档
        if args.action == "get":
            if args.doc_id is None:
                raise ValueError("get 操作需要 --doc-id")
            doc = ingest.get_by_id(client, args.doc_id)
            if doc:
                logger.info("文档: %s", doc)
            else:
                logger.info("未找到文档 ID: %s", args.doc_id)

        # 10. stats：集合统计信息
        if args.action == "stats":
            stats = client.get_collection_stats()
            logger.info("集合统计: %s", stats)

        # 11. list-collections：列出所有集合
        if args.action == "list-collections":
            names = client.list_collections()
            logger.info("集合列表 (%d): %s", len(names), names)

        # 12. drop-collection：删除集合
        if args.action == "drop-collection":
            client.drop_collection(args.collection_name)

        # 13. clear：清空集合
        if args.action == "clear":
            client.clear_collection(args.collection_name)

    except Exception as exc:
        # 统一异常处理，打印堆栈并以非 0 状态码退出
        logger.exception("操作失败: %s", exc)
        sys.exit(1)
    finally:
        # 确保连接被正常关闭
        client.disconnect()


def main():
    """
    CLI 程序入口函数。

    设计上保持非常薄：
    - 只负责：解析参数 + 调用 run_action
    - 方便被其他 Python 代码复用：`from milvus_mvp.cli import main`
    """
    args = parse_args()
    run_action(args)


if __name__ == "__main__":
    # 允许直接 `python -m milvus_mvp.cli` 或脚本方式执行时，作为入口使用
    main()

