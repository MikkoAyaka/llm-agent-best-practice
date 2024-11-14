import os

import dataset
import dspy
import inject
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from dataset import Database
from dspy import LM
from loguru import logger


def ioc_config_dspy(binder):
    def create_lm():
        return dspy.LM(model=os.getenv('OPENAI_API_MODEL'))

    binder.bind_to_provider(LM, create_lm)

    default_lm = create_lm()
    dspy.configure(lm=default_lm)

    binder.bind(OpenAIEmbeddingFunction, OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('OPENAI_LLM_API_BASE')
    ))

    logger.success("LLM service connected.")


def ioc_config_database(binder):
    sqlite_db_path = os.path.abspath('../sqlite_nyxis.db')
    sqlite_db_url = f"sqlite:///{sqlite_db_path}"

    sql_db = dataset.connect(sqlite_db_url)

    binder.bind(Database, sql_db)
    logger.success("SQL service connected.")


def ioc_config(binder):
    ioc_config_dspy(binder)
    logger.success("Beans of dspy has been initialized.")
    ioc_config_database(binder)
    logger.success("Beans of database has been initialized.")


def ioc_init():
    logger.info("Initializing IOC...")
    inject.configure(ioc_config)
    logger.success("IOC initialized.")
