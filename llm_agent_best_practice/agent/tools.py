from datetime import datetime

import inject
from chromadb import ClientAPI
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from llm_agent_best_practice.prompt.prompts import Prompts


def realtime_tool_func(unit: str) -> int:
    """
    get the real time date,such as year 2024, month 12, day 15, day 2 of this week
    Args:
        unit(str): the time unit of real time date.
    Example argument inputs:
        'YEAR'
        'MONTH'
        'MINUTE'
        'HOUR'
        'DAY_OF_MONTH'
        'DAY_OF_WEEK'
    """
    now = datetime.now()
    if unit == 'MINUTE':
        return now.minute
    elif unit == 'YEAR':
        return now.year
    elif unit == 'MONTH':
        return now.month
    elif unit == 'HOUR':
        return now.hour
    elif unit == 'DAY_OF_MONTH':
        return now.day
    elif unit == 'DAY_OF_WEEK':
        return now.isoweekday()
    else:
        raise ValueError("Unsupported time unit")


@inject.autoparams()
def default_tool_kits(prompts: Prompts, neo4j_store: Neo4jGraphStore, chroma_store: ChromaVectorStore,
                      sql_query_engine: SQLTableRetrieverQueryEngine):
    tool_kits = []

    neo4j_tool = QueryEngineTool.from_defaults(
        neo4j_store,
        name="neo4j",
        description=prompts.get("neo4j-tool-usage"),
    )

    chroma_tool = QueryEngineTool.from_defaults(
        chroma_store,
        name="chroma",
        description=prompts.get("chroma-tool-usage"),
    )

    sql_tool = QueryEngineTool.from_defaults(
        sql_query_engine,
        name="sqlite",
        description=prompts.get("sql-tool-usage"),
    )
