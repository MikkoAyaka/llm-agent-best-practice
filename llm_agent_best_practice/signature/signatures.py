from typing import Literal

import dspy


class AgentChat(dspy.Signature):
    """你与用户进行聊天"""
    memory = dspy.InputField(desc="你脑海里的模糊记忆")
    history = dspy.InputField(desc="你与用户最近的聊天记录")
    user_msg = dspy.InputField(desc="用户的消息")
    respond_msg = dspy.OutputField(desc="你的回应消息")


IntentType = Literal["daily_chat", "ask_question", "perform_task"]


class Intent(dspy.Signature):
    """分析用户本次对话的意图"""
    user_msg = dspy.InputField(desc="用户的消息")
    intent: IntentType = dspy.OutputField(desc="用户本次消息的意图")


class MemorySummarizer(dspy.Signature):
    """总结大量记忆内容，生成简短的、概括性的摘要"""
    raw_memory = dspy.InputField(desc="原始记忆内容")
    summarized_memory = dspy.OutputField(desc="总结概括后的记忆内容")
