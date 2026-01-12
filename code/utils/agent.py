import openai
import backoff
import time
import random
# from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError
from openai import (
    RateLimitError,
    APIError,
    APIConnectionError,
)
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, InternalServerError

try:
    from openai.error import (
        RateLimitError,
        APIError,
        ServiceUnavailableError,
        APIConnectionError,

    )
except ImportError:
    from openai import (
        RateLimitError,
        APIError,
        APIConnectionError,
        InternalServerError,
    )

    ServiceUnavailableError = APIError

from .openai_utils import OutOfQuotaException, AccessTerminatedException
from .openai_utils import num_tokens_from_string, model2max_context
import re

def sanitize_api_key(k: str) -> str:
    if k is None:
        raise ValueError("API key is None (did -k get parsed and passed correctly?).")
    k = str(k).strip()

    # 关键：HTTP Header 只接受 ASCII
    try:
        k.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"API key contains non-ASCII characters: {k!r}")

    # 可选：基本格式检查（不强制，但有助于早发现传错参数）
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", k):
        raise ValueError(f"API key contains unusual characters: {k!r}")

    return k

# support_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314']
support_models = [
    'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314',
    'deepseek-chat', 'deepseek-reasoner'
]


class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0,
                 api_key: str = None) -> None:
        """Create an agent

        Args:
            model_name(str): model name
            name (str): name of this agent
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            sleep_time (float): sleep because of rate limits
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.openai_api_key = api_key

        # @backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError), max_tries=20)

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APIConnectionError, InternalServerError),
        max_tries=20
    )
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        """make a query

        Args:
            messages (list[dict]): chat history in turbo format
            max_tokens (int): max token in api call
            api_key (str): openai api key
            temperature (float): sampling temperature

        Raises:
            OutOfQuotaException: the apikey has out of quota
            AccessTerminatedException: the apikey has been ban

        Returns:
            str: the return msg
        """
        time.sleep(self.sleep_time)
        if self.model_name not in support_models:
            print(f"Warning: {self.model_name} not in support_models. Proceeding anyway.")
        api_key = sanitize_api_key(api_key)

        # 定位用：确认运行时到底拿到什么 key（跑通后可删除这行）
        # print("DEBUG api_key repr:", repr(api_key))

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        try:
            # 新版 API 调用语法
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 修改返回值的获取方式 对象属性
            gen = response.choices[0].message.content
            return gen

        except RateLimitError as e:
            # 新版异常信息处理
            err_msg = str(e)
            if "You exceeded your current quota" in err_msg:
                raise OutOfQuotaException(api_key)
            elif "Your access was terminated" in err_msg:
                raise AccessTerminatedException(api_key)
            else:
                raise e
        except Exception as e:
            print(f"Unexpected Error: {e}")
            raise e

    def set_meta_prompt(self, meta_prompt: str):
        """Set the meta_prompt

        Args:
            meta_prompt (str): the meta prompt
        """
        # 设置系统提示词（System Message），定义智能体的角色（如“你是一个数学家”）
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        """Add an new event in the memory

        Args:
            event (str): string that describe the event.
        """
        # 将其他人的发言或新问题存入记忆
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
        """
        # 将智能体自己的回答存入记忆
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float = None):
        # 处理 Token
        # 注意：DeepSeek 的 token 计算可能与 GPT 不完全一致，这里沿用 tiktoken 做估算
        model_for_token = self.model_name
        if "deepseek" in self.model_name:
            model_for_token = "gpt-3.5-turbo"  # 用 gpt-3.5 代替计算长度

        num_context_token = sum([num_tokens_from_string(m["content"], model_for_token) for m in self.memory_lst])

        # 获取最大上下文限制，如果 deepseek 不在 model2max_context 里，给个默认值
        max_total_tokens = model2max_context.get(self.model_name, 8192)
        max_token = max_total_tokens - num_context_token

        # 确保不会出现负数
        if max_token <= 0:
            print("Warning: Context window exceeded.")
            max_token = 512

        # 注意：这里需要确保 DebatePlayer 传过来的 self.openai_api_key 存在
        return self.query(
            self.memory_lst,
            max_token,
            api_key=self.openai_api_key,  # 这里会调用子类 DebatePlayer 中的属性
            temperature=temperature if temperature else self.temperature
        )
