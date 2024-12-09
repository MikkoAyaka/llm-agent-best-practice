import asyncio
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from dspy.datasets.gsm8k import gsm8k_metric, GSM8K
from dspy.teleprompt import BootstrapFewShot
from loguru import logger

from llm_agent_best_practice.config.ioc import ioc_init
from llm_agent_best_practice.module.modules import BasicQA


def load_env():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env.development")

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

    input = [
        "你好 请问我叫什么名字?",
        "你好,我叫什么",
        "Hi，我是谁",
        "我是谁你还记得嘛?",
        "你记得我是谁吗??",
        "你记得我的名字吗???",
        "你还记不记得我的名字?",
        "你知道我是谁吗?",
        "你知道 我的名字吗??",
        "你,记不记得我的名字叫什么??",
    ]
    llm_agent = LLMAgent(agent_id=1)
    total = 0
    repeat = 10
    for i in range(repeat):
        start_time = time.time()
        response2 = await llm_agent.chat(input[i])
        end_time = time.time()
        total += end_time - start_time

    print("Aver Time: " + str(total / repeat))


if __name__ == "__main__":
    logger.info("LLM-Agent service starting...")
    global_init()
    logger.success("LLM-Agent service started.")
    asyncio.run(test())
