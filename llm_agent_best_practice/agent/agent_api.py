from typing import List

import inject
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from .memory import AgentMemory
from .tools import default_tool_kits

# 在类外部定义 _agents_dict
_agents_dict: dict[int, 'LLMAgent'] = {}


class LLMAgent:

    @inject.autoparams()
    def __new__(cls, agent_id: int, llm: LLM):
        if agent_id not in _agents_dict:
            # 仅传入 cls 创建新实例
            instance = super(LLMAgent, cls).__new__(cls)
            _agents_dict[agent_id] = instance
        return _agents_dict[agent_id]

    @inject.autoparams()
    def __init__(self, agent_id: int, llm: LLM):
        # 防止重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.react_agent = ReActAgent.from_tools(default_tool_kits(), llm=llm, verbose=True)
        self.memory = AgentMemory(agent_id)
        self._initialized = True  # 标记实例已初始化

    async def chat(self, message: str) -> str:
        long_term_memory = self.memory.long_term
        short_term_memory = self.memory.short_term
        chat_history: List[ChatMessage] = list()
        chat_history += long_term_memory.get()
        chat_history += short_term_memory.get()
        resp = self.react_agent.chat(message=message, chat_history=chat_history)
        return str(resp)
