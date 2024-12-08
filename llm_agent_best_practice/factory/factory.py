import chromadb
import inject
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from dspy import Retrieve
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.retrieve.neo4j_rm import Neo4jRM


def get_retriever(retriever_type: str, collection_name: str, *args) -> Retrieve:
    if retriever_type == "neo4j":
        return Neo4jRM(index_name=collection_name, *args)
    if retriever_type == "chroma":
        embedding_function = inject.instance(OpenAIEmbeddingFunction)
        return ChromadbRM(
            embedding_function=embedding_function,
            persist_directory="/chroma_dir",
            collection_name=collection_name,
            *args
        )
    raise TypeError("No retriever class found for argument {}".format(retriever_type))


def get_chroma_writer(collection_name: str) -> chromadb.Collection:
    return get_retriever("chroma", collection_name).__getattribute__(
        "_chromadb_collection"
    )
