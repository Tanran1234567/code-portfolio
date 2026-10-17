# diplomat/__init__.py
from .attitude import generate_attitude
from .strategy import generate_strategy

# 显式指定包的公共接口，仅包含需要对外暴露的函数
__all__ = ["generate_attitude", "generate_strategy"]
