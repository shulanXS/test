import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取带基础配置的 logger。
    日志级别可通过环境变量 LOG_LEVEL 控制，默认 INFO。
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger(name or "milvus_mvp")

