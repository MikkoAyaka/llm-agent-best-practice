import inject
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import SimpleComposableMemory, VectorMemory, ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding

from llm_agent_best_practice.prompt.prompts import Prompts
from llm_agent_best_practice.repository.chroma_memory import ChromaMemoryRepository

class AgentMemory:

    def __init__(self, agent_id: int):
        self.short_term = _init_short_memory(agent_id)
        self.long_term = _init_long_memory(agent_id)


@inject.autoparams()
def _init_short_memory(agent_id: int, embed_model: OpenAIEmbedding, prompts: Prompts) -> SimpleComposableMemory:
    preset_memories = [
        ChatMessage.from_str(prompts.get('short-memory'), MessageRole.SYSTEM),
    ]

    vector_memory = VectorMemory.from_defaults(
        vector_store=None,
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 1},
    )

    chat_memory_buffer = ChatMemoryBuffer.from_defaults()
    chat_memory_buffer.put_messages(preset_memories)
    composable_memory = SimpleComposableMemory.from_defaults(
        primary_memory=chat_memory_buffer,
        secondary_memory_sources=[vector_memory],
    )
    return composable_memory


@inject.autoparams()
def _init_long_memory(agent_id: int, embed_model: OpenAIEmbedding, prompts: Prompts,
                      memory_repo: ChromaMemoryRepository) -> SimpleComposableMemory:
    preset_memories = [ChatMessage.from_str(prompts.get('long-memory'), MessageRole.SYSTEM)]

    vector_memory = VectorMemory.from_defaults(
        vector_store=memory_repo.get(agent_id),
        embed_model=embed_model,
        retriever_kwargs={"similarity_top_k": 1},
    )

    chat_memory_buffer = ChatMemoryBuffer.from_defaults()
    chat_memory_buffer.put_messages(preset_memories)

    composable_memory = SimpleComposableMemory.from_defaults(
        primary_memory=chat_memory_buffer,
        secondary_memory_sources=[vector_memory],
    )
    return composable_memory
