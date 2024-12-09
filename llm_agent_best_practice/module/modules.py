import dspy

from llm_agent_best_practice.agent.tools import realtime_tool_func
from llm_agent_best_practice.factory.factory import get_retriever
from llm_agent_best_practice.signature.signatures import (
    MemoryRecall,
    RelativeTime2AbsoluteTime,
    AgentChat,
    ModelBasicQA,
)


class MRAG(dspy.Module):
    def __init__(self, agent_id: int):
        super().__init__()
        # self.neo4j_retriever = get_retriever("neo4j", "memory_{}".format(agent_id))
        self.chroma_retriever = get_retriever("chroma", "memory_{}".format(agent_id))
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        # neo4j_context = self.neo4j_retriever(question)
        chroma_context = self.chroma_retriever(question)
        # combined_context = neo4j_context + chroma_context
        return self.respond(context=chroma_context, question=question)


class BasicQA(dspy.Module):

    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(signature=ModelBasicQA)

    def forward(self, history, question):
        return self.respond(history=history, msg=question)


class DeepQA(dspy.Module):

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.mrag_retriever = MRAG(agent.agent_id)
        self.time_sentence_converter = dspy.ReAct(
            signature=RelativeTime2AbsoluteTime, tools=[realtime_tool_func]
        )
        self.respond = dspy.ReAct(signature=AgentChat, tools=agent.tools)

    def forward(self, question):
        absolute_sentence = self.time_sentence_converter(relative=question).absolute
        related_memory = self.mrag_retriever(question=absolute_sentence).response
        summarized_memory = dspy.Predict(signature=MemoryRecall)(
            related_memory=related_memory
        )
        return self.respond(
            memory=summarized_memory,
            history=self.agent.history[-20:],
            user_msg=question,
        )
