import asyncio
import json
import os
from datetime import datetime

import dspy
import inject
import loguru
from dspy import LM
from litellm import completion

from .memory import AgentMemory
from .tools import default_tool_kits
from ..module.modules import BasicQA

# 在类外部定义 _agents_dict
_agents_dict: dict[int, "LLMAgent"] = {}


class LLMAgent:
    history = []

    @inject.autoparams()
    def __new__(cls, agent_id: int, lm: LM):
        if agent_id not in _agents_dict:
            # 仅传入 cls 创建新实例
            instance = super(LLMAgent, cls).__new__(cls)
            _agents_dict[agent_id] = instance
        return _agents_dict[agent_id]

    @inject.autoparams()
    def __init__(self, agent_id: int, lm: LM):
        # 防止重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.lm = lm
        self.agent_id = agent_id
        self.tools = default_tool_kits()
        self.memory = AgentMemory(agent_id=agent_id)
        self.history = self.memory.read(20)
        self._initialized = True

    async def chat(self, message: str) -> str:
        output = ""
        with dspy.context(lm=self.lm):

            chat_module = BasicQA()
            output = chat_module(history=self.history[-20:], question=message).output
        history = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message=message,
            response=output,
        )
        self.history.append(history)
        task = asyncio.create_task(self.memory.write(json.dumps(history)))
        loguru.logger.debug(
            "Agent: {}, Input: {}, Output: {}".format(self.agent_id, message, output)
        )
        return output

    async def stream_chat(self, message: str):
        response = completion(
            model=os.getenv("OPENAI_API_MODEL"),
            api_base=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=[{"content": message, "role": "user"}],
            cache={"no-cache": True, "no-store": True},
            stream=True
        )
        for part in response:
            print(part.choices[0].delta.content or "", end="")
        print()
