from abc import abstractmethod
from openai import OpenAI
import re
import os
# from dotenv import load_dotenv

# load_dotenv()
from loguru import logger


def get_client(model_type: str):
    logger.info(f"获取模型类型的客户端: '{model_type}'")
    
    if model_type.lower() == "local":
        logger.info("检测到本地模型，返回None作为客户端")
        return None
        
    if "openai" in model_type:
        return OpenAI(api_key=os.environ["openai_api_key"])
    if "ollama" in model_type:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    if "lmstudio" in model_type:
        return OpenAI(base_url="http://localhost:1234/v1", api_key="lmstudio")
    
    # 如果没有匹配任何已知类型，提供错误日志
    logger.error(f"未知的模型类型: '{model_type}'")
    return None

def call_llm(client: OpenAI, model: str, user_prompt: str) -> str:
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0,
    )
    return chat_completion.choices[0].message.content


def get_move_from_response(response: str) -> str:
    try:
        matches = re.findall(r"- ([\w ]+)", response)
        moves = ["".join(match) for match in matches]
        return moves
    except Exception:
        logger.info(response)
        return [""]


class LLM:
    @abstractmethod
    def get_actions(self, prompt: str) -> list[str]:
        pass


class LLamaLLM(LLM):
    def __init__(self, model_type: str, model_name: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.client = get_client(self.model_type)

    def get_actions(self, prompt: str) -> list[str]:
        response = call_llm(self.client, self.model_name, prompt)
        actions = get_move_from_response(response)
        return actions


if __name__ == "__main__":
    client = get_client()
    system_prompt = ""
    with open("prompts/1.txt", "r") as f:
        system_prompt = "".join(f.readlines())
    # print(system_prompt)
    message = call_llm(client, "mistral-nemo:12b", system_prompt)

    matches = re.findall(r"- ([\w ]+)", message)
    moves = ["".join(match) for match in matches]
    print(moves)
