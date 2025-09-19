from typing import List
from langchain_core.embeddings import Embeddings
import os

class ZhipuAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    def __init__(self):
        """
        实例化ZhipuAiClient为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有zai库，则将返回实例化的ZhipuAiClient类；否则将报错 'ModuleNotFoundError: No module named 'zai''.
        """
        from zai import ZhipuAiClient
        api_key= os.getenv("ZAI_API_KEY")
        self.client = ZhipuAiClient(api_key=api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """

        result = []
        for i in range(0, len(texts), 64):
            response = self.client.embeddings.create(
                model="embedding-3",
                input=texts[i:i+64]
            )
            result.extend([embedding.embedding for embedding in response.data])
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]