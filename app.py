"""
应用入口：转发至 milvus_mvp.cli，保持单一入口便于部署和调用。
"""

from milvus_mvp.cli import main


if __name__ == "__main__":
    main()

