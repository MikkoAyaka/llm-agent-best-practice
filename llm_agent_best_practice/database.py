import os
import sqlite3

import chromadb
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore

sql_client = sqlite3.connect('sqlite_nyxis.db')
chroma_client = chromadb.Client()

print(os.getenv('NEO4J_URI'))
neo4j_store = Neo4jGraphStore(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)


def chroma_store_memory(_id: int):
    return ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("memory_store_"+str(_id)))

