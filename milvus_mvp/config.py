from dataclasses import dataclass, field
import os
from typing import Dict


def _get_env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class MilvusSettings:
    """Milvus 基础配置，支持通过环境变量覆盖默认值。"""

    host: str = field(default_factory=lambda: _get_env("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_get_env("MILVUS_PORT", "19530")))
    collection_name: str = field(default_factory=lambda: _get_env("MILVUS_COLLECTION", "document_collection"))
    dimension: int = field(default_factory=lambda: int(_get_env("MILVUS_DIMENSION", "384")))
    max_length: int = field(default_factory=lambda: int(_get_env("MILVUS_MAX_LENGTH", "5000")))

    metric_type: str = field(default_factory=lambda: _get_env("MILVUS_METRIC", "L2"))
    index_type: str = field(default_factory=lambda: _get_env("MILVUS_INDEX_TYPE", "IVF_FLAT"))
    index_nlist: int = field(default_factory=lambda: int(_get_env("MILVUS_INDEX_NLIST", "128")))
    search_nprobe: int = field(default_factory=lambda: int(_get_env("MILVUS_SEARCH_NPROBE", "10")))

    top_k_default: int = field(default_factory=lambda: int(_get_env("MILVUS_TOPK", "5")))
    auto_id: bool = field(default_factory=lambda: _get_env("MILVUS_AUTO_ID", "true").lower() == "true")

    anns_field: str = "embedding"
    text_field: str = "text"
    id_field: str = "id"

    def index_params(self) -> Dict:
        return {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"nlist": self.index_nlist},
        }

    def search_params(self) -> Dict:
        return {"metric_type": self.metric_type, "params": {"nprobe": self.search_nprobe}}

