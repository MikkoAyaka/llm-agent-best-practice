from datetime import datetime

import dspy
import inject
from dataset import Database

from llm_agent_best_practice.factory.factory import get_retriever
from llm_agent_best_practice.signature.signatures import MemorySummarizer


class AgentMemory:
    # 摘要记忆上次更新时间
    summary_last_update = 0

    @inject.autoparams()
    def __init__(self, agent_id: int, sql_db: Database):
        self.memory_retriever = get_retriever(retriever_type="chroma", agent_id=agent_id)
        self.sql_db = sql_db
        self.agent_id = agent_id

    def get_summary(self) -> str:
        table = self.sql_db['summary_memory']
        row = table.find_one(agent_id=self.agent_id)
        if row is None:
            return ""
        return row['content']

    def update_summary(self):
        nowtime = datetime.now().timestamp()
        # 每 10 分钟更新一次摘要
        if nowtime - self.summary_last_update > 60 * 10:
            self.summary_last_update = nowtime
            summary = self._summarize()
            table = self.sql_db['summary_memory']
            table.upsert(dict(content=summary, agent_id=self.agent_id))

    def _extract(self) -> list[str]:
        return [doc.long_text for doc in self.memory_retriever("", k=20)]

    def _summarize(self) -> str:
        module = dspy.Predict(signature=MemorySummarizer)
        result = module(raw_memory='\n'.join(self._extract()))
        return result
