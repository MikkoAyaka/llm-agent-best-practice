import asyncio

from llm_agent_best_practice.settings import global_init


async def test():
    from llm_agent_best_practice.agent.agent_api import LLMAgent
    llm_agent = LLMAgent(agent_id=1)
    response = await llm_agent.chat("你好")
    print(response)


if __name__ == '__main__':
    global_init()
    asyncio.run(test())



