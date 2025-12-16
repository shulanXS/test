"""
向量化模块
使用sentence-transformers将文本转换为向量
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class TextVectorizer:
    """文本向量化器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量化器
        
        Args:
            model_name: sentence-transformers模型名称
                - all-MiniLM-L6-v2: 快速，384维
                - all-mpnet-base-v2: 更准确，768维
        """
        print(f"正在加载模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ 模型加载完成，向量维度: {self.dimension}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量（嵌入）列表

        Args:
            texts: 需要进行向量化的文本列表（List[str]）。
                   也支持直接传入单个字符串，将自动转换为列表处理。

        Returns:
            List[List[float]]: 每个输入文本对应的向量（以浮点数列表表示）。
        
        详细流程说明:
        1. 兼容参数类型。如果输入为单个字符串，封装成单元素列表，实现统一处理。
        2. 使用SentenceTransformer模型进行向量化（self.model.encode）。
           - convert_to_numpy=True: 输出为numpy array，便于后续数值计算和高效存储。
        3. 将numpy array通过tolist()方法转为普通Python列表，便于序列化或与数据库/框架交互。
        """
        # 如果传入的是单个字符串，封装为单元素列表以统一处理
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用句子Transformer模型编码，得到文本的嵌入表示（NumPy数组格式）
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # 将NumPy数组转换为普通Python列表（List[List[float]]），以便兼容性和易于存储
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension

