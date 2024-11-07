import os
import sqlite3

import chromadb
import inject
from llama_index.core import SQLDatabase, VectorStoreIndex, ServiceContext
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableSchema, SQLTableNodeMapping, ObjectIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from sqlalchemy import create_engine, MetaData

from llm_agent_best_practice.prompt.prompts import Prompts
from llm_agent_best_practice.repository.chroma_memory import ChromaMemoryRepository


def ioc_config_llm(binder):
    prompts = Prompts()
    binder.bind(Prompts, prompts)

    llm = OpenAI(api_base=os.getenv('OPENAI_LLM_API_BASE'), api_key=os.getenv('OPENAI_LLM_API_KEY'),
                 model=os.getenv('OPENAI_LLM_API_MODEL'))
    binder.bind(OpenAI, llm)

    embedding_model = OpenAIEmbedding(temperature=0, api_base=os.getenv('OPENAI_LLM_API_BASE'))
    binder.bind(OpenAIEmbedding, embedding_model)


def ioc_config_database(binder):
    sqlite_db_path = os.path.abspath('../sqlite_nyxis.db')
    sqlite_db_url = f"sqlite:///{sqlite_db_path}"

    sql_conn = sqlite3.connect(sqlite_db_path)
    binder.bind(sqlite3.Connection, sql_conn)

    chroma_client = chromadb.Client()
    binder.bind(chromadb.ClientAPI, chroma_client)

    chroma_memory_repo = ChromaMemoryRepository(chroma_client)
    binder.bind(ChromaMemoryRepository, chroma_memory_repo)

    chroma_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("nyxis"))
    binder.bind(ChromaVectorStore, chroma_store)

    neo4j_store = Neo4jGraphStore(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    binder.bind(Neo4jGraphStore, neo4j_store)

    def factory_sql_engine():
        base_engine = create_engine(sqlite_db_url, echo=False)
        metadata_obj = MetaData()
        metadata_obj.reflect(base_engine)
        sql_database = SQLDatabase(base_engine)
        table_schema_objs = []
        for table_name in metadata_obj.tables.keys():
            table_schema_objs.append(SQLTableSchema(table_name=table_name))
        table_node_mapping = SQLTableNodeMapping(sql_database)
        obj_index = ObjectIndex.from_objects(
            objects=table_schema_objs,
            object_mapping=table_node_mapping,
            index_cls=VectorStoreIndex
        )
        llm = inject.instance(OpenAI)
        service_context = ServiceContext.from_defaults(llm=llm)
        return SQLTableRetrieverQueryEngine(
            sql_database,
            obj_index.as_retriever(similarity_top_k=1),
            service_context=service_context,
        )

    binder.bind(SQLTableRetrieverQueryEngine, factory_sql_engine)


def ioc_config(binder):
    ioc_config_llm(binder)
    ioc_config_database(binder)


def ioc_init():
    inject.configure(ioc_config)
