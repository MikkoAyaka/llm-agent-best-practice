import json
from datetime import datetime

import dspy
import inject
import loguru
from dspy import LM

from .memory import AgentMemory
from .tools import default_tool_kits
from ..signature.signatures import Intent, IntentType, AgentChat, PerformTask

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
        self.history = self.memory.read(4)
        self._initialized = True

    async def chat(self, message: str) -> str:
        intent_module = dspy.Predict(signature=Intent)
        intent: IntentType = intent_module(user_msg=message).intent
        output = ""
        with dspy.context(lm=self.lm):
            match intent:
                case "daily_chat":
                    chat_module = dspy.Predict(signature=AgentChat)
                    result = chat_module(
                        memory=self.memory.get_summary(),
                        history=self.history[-4:],
                        user_msg=message,
                    )
                    output = result.respond_msg
                case "perform_task":
                    chat_module = dspy.ReAct(signature=PerformTask, tools=self.tools)
                    result = chat_module(
                        command=message
                    )
                    output = result.respond_msg
                case "other":
                    print("Unsupported intent: other")
                    pass
        history = dict(
                time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                message=message,
                response=output,
            )
        self.history.append(history)
        self.memory.write(json.dumps(history))
        loguru.logger.debug(
            "Agent: {}, Input: {}, Intent: {}, Output: {}".format(
                self.agent_id, message, intent, output
            )
        )
        return output
