import asyncio
import os

from dotenv import load_dotenv

from llm_agent_best_practice.config.ioc import ioc_init


def load_env():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, '.env.development')

    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        raise FileNotFoundError(f"Environment file {env_path} not found")


def global_init():
    load_env()
    ioc_init()


async def test():
    from llm_agent_best_practice.agent.agent_api import LLMAgent
    llm_agent = LLMAgent(agent_id=1)
    response = await llm_agent.chat("你好")
    print(response)


if __name__ == '__main__':
    global_init()
    asyncio.run(test())
