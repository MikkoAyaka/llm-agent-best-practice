from typing import Literal

import dspy


class AgentChat(dspy.Signature):
    """与用户进行聊天对话"""

    memory = dspy.InputField(desc="脑海里的相关记忆")
    history = dspy.InputField(desc="与用户最近的聊天记录")
    user_msg = dspy.InputField(desc="用户的消息")
    respond_msg = dspy.OutputField(desc="回应消息")


class PerformTask(dspy.Signature):
    """你按照用户的指令调用工具执行操作"""
    command = dspy.InputField(desc="用户提出的指令")
    respond_msg = dspy.OutputField(desc="指令执行的结果(回应用户的消息)")


IntentType = Literal["daily_chat", "perform_task", "other"]


class Intent(dspy.Signature):
    """分析用户本次对话的意图"""

    user_msg = dspy.InputField(desc="用户的消息")
    intent: IntentType = dspy.OutputField(desc="用户本次消息的意图")


class MemorySummarizer(dspy.Signature):
    """总结大量记忆内容，生成简短的、概括性的摘要"""

    raw_memory = dspy.InputField(desc="原始记忆内容")
    summarized_memory = dspy.OutputField(desc="总结概括后的记忆内容")


class MemoryRecall(dspy.Signature):
    """根据关键信息进行记忆回忆，优先思考最新的记忆，生成关键性的摘要（保留关键信息）"""
    related_memory = dspy.InputField(desc="与关键信息相关的大量记忆内容")
    summarized_memory = dspy.OutputField(desc="最关键的记忆摘要信息")


class RelativeTime2AbsoluteTime(dspy.Signature):
    """把句子中的相对时间表述改为绝对时间表述（2024年5月7日下午14点31分）"""
    relative = dspy.InputField(desc="包含相对时间表述的句子(如昨天、今天、上个月、去年...)")
    absolute = dspy.OutputField(desc="采用绝对时间表述的句子(如 2024年5月7日下午14点31分...)")
