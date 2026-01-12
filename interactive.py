"""
文件概述
Debateplayer 代表辩论赛中的一个选手 它继承自 Agent 类（核心的 API 调用逻辑在 Agent 类里）存储了选手的名称、模型类型、随机随机性 和 API Key
Debate 类
流程
1. 读入辩题（用户输入）
2. 用 config4all.json 里的 prompt 模板初始化各个角色
3. 正反双方先各给出观点
4. 主持人总结/判断是否已有最终答案
5. 若没有最终答案，就进入后续回合继续辩论
6. 若一直没得出答案，就启用 “Judge（终局裁判）” 强行从候选中选一个
"""

import os
import json
import random
# random.seed(0)
from code.utils.agent import Agent
import ast
import re

openai_api_key = "sk-3792687bb4804a1d8f97f6c61cbb17e3"
# 角色 正方 反方 裁判
NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
]


class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        # model_name：要调用的模型名 name：角色名（正方/反方/主持人）
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key

def safe_parse_dict(text: str) -> dict:
    """
    安全解析模型输出为 dict：
    1) 优先按 JSON 解析（最规范）
    2) JSON 失败时，尝试抽取 {...} 再解析
    3) 仍失败则用 ast.literal_eval（比 eval 安全，不执行代码）
    解析失败会抛出 ValueError
    """
    if text is None:
        raise ValueError("Model output is None")

    s = text.strip()

    # 有些模型会用 ```json ... ``` 包起来，先去掉 code fence
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    # 先尝试直接 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 尝试从文本中抽取第一个 {...}（避免前后多了自然语言）
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        candidate = m.group(0).strip()
        # 再试 JSON
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 再试 literal_eval
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 最后尝试对整段做 literal_eval
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(f"Cannot parse model output as dict. Raw output:\n{s}")

class Debate:
    def __init__(self,
                 model_name: str = 'deepseek-chat',
                 temperature: float = 0,
                 num_players: int = 3,  # num_players=3：玩家数，当前代码固定就是 3（正、反、主持）
                 openai_api_key: str = None,
                 config: dict = None,  # config=None：prompt 配置（来自 config4all.json）
                 max_round: int = 3,
                 sleep_time: float = 0
                 ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            openai_api_key (str): As the parameter name suggests
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time

        self.init_prompt()  # 对 config 里的 prompt 模板做替换

        # creat&init agents
        self.creat_agents() # 创建 3 个 DebatePlayer
        self.init_agents()  # 给每个 agent 设置 meta prompt，然后跑第一轮辩论

    def init_prompt(self):
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##debate_topic##", self.config["debate_topic"])
            # config 是从 json 读来的字典，里面有很多 prompt 文本  替换成用户输入的辩题 self.config["debate_topic"]

        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        # creates players
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature,
                         openai_api_key=self.openai_api_key, sleep_time=self.sleep_time) for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt']) # You are a debater
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])

        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")
        self.affirmative.add_event(self.config['affirmative_prompt'])
        # 把某句话加入“当前对话事件列表/上下文”
        self.aff_ans = self.affirmative.ask()
        self.affirmative.add_memory(self.aff_ans)
        # 把回答存入长期记忆
        self.config['base_answer'] = self.aff_ans
        # 把正方第一轮的回答作为“基础答案”保存到 config，最后输出用

        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))  # 用来把正方观点塞进去让反方反驳
        self.neg_ans = self.negative.ask()
        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(
            self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##',
                                                                                         self.neg_ans).replace(
                '##round##', 'first'))
        self.mod_ans = self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)
        # self.mod_ans = eval(self.mod_ans)
        # self.mod_ans = json.loads(self.mod_ans)
        self.mod_ans = safe_parse_dict(self.mod_ans)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
            9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config["debate_topic"])
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Debate Reason -----")
        print(self.config["Reason"])

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str): # speaker 自己不接收这条消息，其他人接收
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer): # 某个玩家生成回答后把回答存起来并告诉其他人
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)

    def run(self):

        for round in range(self.max_round - 1):

            if self.mod_ans["debate_answer"] != '':
                break
            else:
                print(f"===== Debate Round-{round + 2} =====\n")
                self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                self.aff_ans = self.affirmative.ask()
                self.affirmative.add_memory(self.aff_ans)

                self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
                self.neg_ans = self.negative.ask()
                self.negative.add_memory(self.neg_ans)

                self.moderator.add_event(
                    self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##',
                                                                                                 self.neg_ans).replace(
                        '##round##', self.round_dct(round + 2)))
                self.mod_ans = self.moderator.ask()
                self.moderator.add_memory(self.mod_ans)
                # self.mod_ans = eval(self.mod_ans)
                self.mod_ans = safe_parse_dict(self.mod_ans)

        if self.mod_ans["debate_answer"] != '':
            self.config.update(self.mod_ans)
            self.config['success'] = True

        # ultimate deadly technique.
        else:
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature,
                                        openai_api_key=self.openai_api_key, sleep_time=self.sleep_time)
            aff_ans = self.affirmative.memory_lst[2]['content']
            neg_ans = self.negative.memory_lst[2]['content']

            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

            # extract answer candidates
            judge_player.add_event(
                self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            ans = judge_player.ask()
            judge_player.add_memory(ans)

            # select one from the candidates
            judge_player.add_event(self.config['judge_prompt_last2'])
            ans = judge_player.ask()
            judge_player.add_memory(ans)

            # ans = eval(ans)
            ans = safe_parse_dict(ans)

            if ans["debate_answer"] != '':
                self.config['success'] = True
                # save file
            self.config.update(ans)
            self.players.append(judge_player)

        self.print_answer()


if __name__ == "__main__":

    current_script_path = os.path.abspath(__file__)
    # MAD_path = current_script_path.rsplit("/", 1)[0]
    MAD_path = os.path.dirname(current_script_path)

    while True:
        debate_topic = ""
        while debate_topic == "":
            debate_topic = input(f"\nEnter your debate topic: ")

        # config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
        # config['debate_topic'] = debate_topic
        config_file_path = os.path.join(MAD_path, "code", "utils", "config4all.json")

        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"找不到配置文件，请检查路径: {config_file_path}")
            break

        config['debate_topic'] = debate_topic

        debate = Debate(num_players=3, openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=0)
        debate.run()
