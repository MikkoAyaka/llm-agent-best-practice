import os
import sqlite3

import chromadb
import inject
from llama_index.core import SQLDatabase, VectorStoreIndex, ServiceContext
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.objects import SQLTableSchema, SQLTableNodeMapping, ObjectIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from sqlalchemy import create_engine, MetaData

from llm_agent_best_practice.prompt.prompts import Prompts
from llm_agent_best_practice.repository.chroma_memory import ChromaMemoryRepository
from llm_agent_best_practice.util.utils import py_require, soft_import


def ioc_config_llm(binder):
    prompts = Prompts()
    binder.bind(Prompts, prompts)

    def import_dash_scope():
        DashScope = soft_import("llama_index.llms.dashscope", "DashScope")
        DashScopeGenerationModels = soft_import("llama_index.llms.dashscope", "DashScopeGenerationModels")
        llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=os.getenv("DASHSCOPE_API_KEY"))
        binder.bind(LLM, llm)

    if not py_require(import_dash_scope):
        llm = OpenAI(api_base=os.getenv('OPENAI_LLM_API_BASE'), api_key=os.getenv('OPENAI_LLM_API_KEY'),
                     model=os.getenv('OPENAI_LLM_API_MODEL'))
        binder.bind(LLM, llm)

    logger.success("LLM service connected.")

    embedding_model = OpenAIEmbedding(temperature=0, api_base=os.getenv('OPENAI_LLM_API_BASE'))
    binder.bind(OpenAIEmbedding, embedding_model)
    logger.success("Embedding service connected.")


def ioc_config_database(binder):
    sqlite_db_path = os.path.abspath('../sqlite_nyxis.db')
    sqlite_db_url = f"sqlite:///{sqlite_db_path}"

    sql_conn = sqlite3.connect(sqlite_db_path)
    binder.bind(sqlite3.Connection, sql_conn)
    logger.success("SQL service connected.")

    chroma_client = chromadb.Client()
    binder.bind(chromadb.ClientAPI, chroma_client)
    logger.success("Chroma service connected.")

    chroma_memory_repo = ChromaMemoryRepository(chroma_client)
    binder.bind(ChromaMemoryRepository, chroma_memory_repo)

    chroma_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("nyxis"))
    binder.bind(ChromaVectorStore, chroma_store)

    def init_neo4j_store():
        Neo4jGraphStore = soft_import("llama_index.graph_stores.neo4j", "Neo4jGraphStore")
        neo4j_store = Neo4jGraphStore(
            url=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        binder.bind(Neo4jGraphStore, neo4j_store)
        logger.success("Neo4j service connected.")

    py_require(init_neo4j_store)

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
    logger.success("SQLRetrieverQueryEngine has been initialized.")


def ioc_config(binder):
    ioc_config_llm(binder)
    logger.success("Beans of llm has been initialized.")
    ioc_config_database(binder)
    logger.success("Beans of database has been initialized.")


def ioc_init():
    logger.info("Initializing IOC...")
    inject.configure(ioc_config)
    logger.success("IOC initialized.")
