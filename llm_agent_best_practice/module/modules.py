import dspy

from llm_agent_best_practice.factory.factory import get_retriever


class MRAG(dspy.Module):
    def __init__(self, agent_id: int):
        super().__init__()
        self.neo4j_retriever = get_retriever("neo4j", agent_id)
        self.chroma_retriever = get_retriever("chroma", agent_id)
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        neo4j_context = self.neo4j_retriever.forward(question)
        chroma_context = self.chroma_retriever.forward(question)
        combined_context = neo4j_context + chroma_context
        return self.respond(context=combined_context, question=question)
