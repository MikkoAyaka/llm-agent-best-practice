import inject
from chromadb import ClientAPI
from llama_index.vector_stores.chroma import ChromaVectorStore


class ChromaMemoryRepository:
    def __init__(self, chroma_client: ClientAPI):
        self.chroma_client = chroma_client
        self.dict: dict[int, ChromaVectorStore] = {}

    def get(self, index: int) -> ChromaVectorStore:
        if index not in self.dict:
            self.dict[index] = ChromaVectorStore(
                chroma_collection=self.chroma_client.get_or_create_collection("memory_store_" + str(index)))
        return self.dict[index]
