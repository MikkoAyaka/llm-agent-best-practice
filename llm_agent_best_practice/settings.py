import os

import inject
from dotenv import load_dotenv, dotenv_values
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI


def load_env():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '.env.development')

    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        raise FileNotFoundError(f"Environment file {env_path} not found")


def ioc_config(binder):
    llm = LlamaIndexOpenAI(api_base=os.getenv('OPENAI_LLM_API_BASE'), api_key=os.getenv('OPENAI_LLM_API_KEY'),
                           model=os.getenv('OPENAI_LLM_API_MODEL'))
    binder.bind(LlamaIndexOpenAI, llm)
    embedding_model = OpenAIEmbedding(temperature=0, api_base=os.getenv('OPENAI_LLM_API_BASE'))
    binder.bind(OpenAIEmbedding, embedding_model)


def init_ioc():
    inject.configure(ioc_config)


def global_init():
    load_env()
    init_ioc()
