import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from llm_agent_best_practice.config.ioc import ioc_init


def load_env():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '.env.development')

    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        raise FileNotFoundError(f"Environment file {env_path} not found")


def global_init():
    logger.info("Global init start")
    load_env()
    ioc_init()
    logger.success("Global init done")


async def test():
    from llm_agent_best_practice.agent.agent import LLMAgent
    llm_agent = LLMAgent(agent_id=1)
    response2 = await llm_agent.chat("你好，我叫什么名字？")


if __name__ == '__main__':
    logger.info("LLM-Agent service starting...")
    global_init()
    logger.success("LLM-Agent service started.")
    asyncio.run(test())
