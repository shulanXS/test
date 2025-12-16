"""
文本向量化模块
使用 sentence-transformers 将文本转换为向量
"""

from typing import List
from sentence_transformers import SentenceTransformer

from .log import get_logger

logger = get_logger(__name__)


class TextVectorizer:
    """文本向量化器"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: sentence-transformers 模型名称
                - all-MiniLM-L6-v2: 快速，384维
                - all-mpnet-base-v2: 更准确，768维
        """
        logger.info("加载向量化模型: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("模型加载完成，向量维度: %s", self.dimension)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension

