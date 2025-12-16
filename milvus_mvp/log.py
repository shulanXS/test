import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取配置好的 logger 实例，根据 LOG_LEVEL 环境变量决定日志详细程度。

    主要流程：
    1. 读取环境变量 LOG_LEVEL，决定日志输出级别，默认 INFO。
    2. 配置日志输出格式和最低级别（只会全局生效一次）。
    3. 返回指定名称的 logger，如果不传 name，则返回名为 "milvus_mvp" 的 logger。

    Args:
        name (Optional[str]): logger的名称（通常为 __name__）

    Returns:
        logging.Logger: 按需配置的 logger 实例
    """
    # step1: 获取日志级别（如 "INFO", "DEBUG" 等等，不区分大小写）
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # step2: 配置日志格式，只会在第一次调用时真正生效
    # （有 handler 时，basicConfig 会忽略重复设置）
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # step3: 返回 logger 对象，默认名称 "milvus_mvp"（方便全局日志分组）
    if name:
        logger_name = name
    else:
        logger_name = "milvus_mvp"

    return logging.getLogger(logger_name)

