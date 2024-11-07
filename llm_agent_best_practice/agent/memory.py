import inject
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import SimpleComposableMemory, VectorMemory, ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding

from llm_agent_best_practice.database import chroma_store_memory

short_memory_prompt = ""
long_memory_prompt = ""


class AgentMemory:

    def __init__(self, agent_id: int):
        self.short_term = _init_short_memory(agent_id)
        self.long_term = _init_long_memory(agent_id)


@inject.autoparams()
def _init_short_memory(agent_id: int, embed_model: OpenAIEmbedding) -> SimpleComposableMemory:
    preset_memories = [
        ChatMessage.from_str(short_memory_prompt, MessageRole.SYSTEM),
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
def _init_long_memory(agent_id: int, embed_model: OpenAIEmbedding) -> SimpleComposableMemory:
    preset_memories = [ChatMessage.from_str(long_memory_prompt, MessageRole.SYSTEM)]

    vector_memory = VectorMemory.from_defaults(
        vector_store=chroma_store_memory(agent_id),
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
