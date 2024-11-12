import importlib
import os
from datetime import datetime

import inject
import loguru
from chromadb import ClientAPI
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore

from llm_agent_best_practice.prompt.prompts import Prompts
from llm_agent_best_practice.util.utils import py_require, soft_import


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
def default_tool_kits(prompts: Prompts, chroma_store: ChromaVectorStore,
                      sql_query_engine: SQLTableRetrieverQueryEngine):
    tool_kits = []

    def import_bing_search_tool():
        BingSearchToolSpec = soft_import("llama_index.tools.bing_search", "BingSearchToolSpec")
        search_tool_kits = BingSearchToolSpec(api_key=os.getenv('BING_API_KEY')).to_tool_list()
        tool_kits.extend(search_tool_kits)

    py_require(import_bing_search_tool, "Bing_search_tool for agent will not work.")

    def import_neo4j_tool():
        Neo4jGraphStore = soft_import("llama_index.graph_stores.neo4j", "Neo4jGraphStore")
        neo4j_store = inject.instance(Neo4jGraphStore)
        neo4j_tool = QueryEngineTool.from_defaults(
            neo4j_store,
            name="neo4j",
            description=prompts.get("neo4j-tool-usage"),
        )
        tool_kits.append(neo4j_tool)

    py_require(import_neo4j_tool, "Neo4j_tool for agent will not work.")

    chroma_tool = QueryEngineTool.from_defaults(
        chroma_store,
        name="chroma",
        description=prompts.get("chroma-tool-usage"),
    )
    tool_kits.append(chroma_tool)

    sql_tool = QueryEngineTool.from_defaults(
        sql_query_engine,
        name="sqlite",
        description=prompts.get("sql-tool-usage"),
    )
    tool_kits.append(sql_tool)

    return tool_kits
