# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    所有Agent的抽象基类。
    强制要求每个Agent都必须实现一个 run 方法。
    """
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        每个Agent执行其核心逻辑的入口点。
        """
        pass

