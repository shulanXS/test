from dataclasses import dataclass, field
import os
from typing import Dict

def _get_env(key: str, default: str) -> str:
    """
    获取环境变量的值，如果未设置则返回默认值。
    :param key: 环境变量的名称
    :param default: 默认值
    :return: 环境变量的值或默认值
    """
    return os.getenv(key, default)

@dataclass
class MilvusSettings:
    """
    Milvus 基础配置类。
    该配置支持通过读取环境变量覆盖默认参数，方便灵活部署和测试。

    属性说明:
        host: Milvus 服务器主机名，默认为 localhost，可通过环境变量 MILVUS_HOST 配置
        port: Milvus 服务器端口，默认为 19530，可通过 MILVUS_PORT 环境变量配置
        collection_name: Milvus 集合名，默认为 document_collection，可通过 MILVUS_COLLECTION 配置
        dimension: 向量维度，默认为 384，可通过 MILVUS_DIMENSION 配置
        max_length: 单条数据最大长度，默认为 5000，可通过 MILVUS_MAX_LENGTH 配置

        metric_type: 相似度度量方式，默认为 L2（欧式距离），可通过 MILVUS_METRIC 配置
        index_type: 索引类型，默认为 IVF_FLAT，可通过 MILVUS_INDEX_TYPE 配置
        index_nlist: 索引 nlist 参数，默认为 128，可通过 MILVUS_INDEX_NLIST 配置
        search_nprobe: 搜索 nprobe 参数，默认为 10，可通过 MILVUS_SEARCH_NPROBE 配置

        top_k_default: 默认检索 topK 数量，默认为 5，可通过 MILVUS_TOPK 配置
        auto_id: 是否使用 Milvus 自动生成主键，默认为 true，可通过 MILVUS_AUTO_ID 配置

        anns_field: 用于存储 embedding 的字段名
        text_field: 用于存储文本的字段名
        id_field: 用于存储主键的字段名
    """

    # Milvus 服务器主机名，通过环境变量覆盖
    host: str = field(default_factory=lambda: _get_env("MILVUS_HOST", "localhost"))
    # Milvus 服务器端口，通过环境变量覆盖
    port: int = field(default_factory=lambda: int(_get_env("MILVUS_PORT", "19530")))
    # 集合名称，通过环境变量覆盖
    collection_name: str = field(default_factory=lambda: _get_env("MILVUS_COLLECTION", "document_collection"))
    # 向量维度，通过环境变量覆盖
    dimension: int = field(default_factory=lambda: int(_get_env("MILVUS_DIMENSION", "384")))
    # 最大长度，通过环境变量覆盖
    max_length: int = field(default_factory=lambda: int(_get_env("MILVUS_MAX_LENGTH", "5000")))

    # 相似度度量类型，通过环境变量覆盖
    metric_type: str = field(default_factory=lambda: _get_env("MILVUS_METRIC", "L2"))
    # 索引类型，通过环境变量覆盖
    index_type: str = field(default_factory=lambda: _get_env("MILVUS_INDEX_TYPE", "IVF_FLAT"))
    # nlist 参数，通过环境变量覆盖
    index_nlist: int = field(default_factory=lambda: int(_get_env("MILVUS_INDEX_NLIST", "128")))
    # nprobe 参数，通过环境变量覆盖
    search_nprobe: int = field(default_factory=lambda: int(_get_env("MILVUS_SEARCH_NPROBE", "10")))

    # top_k 默认值，通过环境变量覆盖
    top_k_default: int = field(default_factory=lambda: int(_get_env("MILVUS_TOPK", "5")))
    # 是否自动生成 ID，通过环境变量覆盖 ("true"/"false")
    auto_id: bool = field(default_factory=lambda: _get_env("MILVUS_AUTO_ID", "true").lower() == "true")

    # 字段名设定，通常无需更改
    anns_field: str = "embedding"
    text_field: str = "text"
    id_field: str = "id"

    def index_params(self) -> Dict:
        """
        构建用于创建索引的参数字典
        :return: 包含 metric_type, index_type, 以及 nlist 的参数字典
        """
        return {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"nlist": self.index_nlist},
        }

    def search_params(self) -> Dict:
        """
        构建用于向量检索的参数字典
        :return: 包含 metric_type 以及 nprobe 的参数字典
        """
        return {"metric_type": self.metric_type, "params": {"nprobe": self.search_nprobe}}

