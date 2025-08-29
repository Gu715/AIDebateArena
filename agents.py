# agents.py
from dataclasses import dataclass
from typing import Dict
from llm import OpenAICompatLLM, ChatMessage
import config

# ===== System Prompts =====
SYS_PRO = (
    "你是正方（Pro）。\n"
    "立场：支持该命题。\n"
    "任务：阐述并捍卫你的立场，同时回应和反驳反方的观点。\n"
    "你不必用括号说明你的语气或目的。你的阐述不必过于书面化，可以保持一定的口语化，可以适当与对方进行互动，比如引用对方观点，或者反问对方。"
)
SYS_CON = (
    "你是反方（Con）。\n"
    "立场：反对该命题。\n"
    "任务：质疑和削弱正方的立场，同时提出替代性的解释或反例。\n"
    "你不必用括号说明你的语气或目的。你的阐述不必过于书面化，可以保持一定的口语化，可以适当与对方进行互动，比如引用对方观点，或者反问对方。"
)
SYS_JUDGE_FINAL = (
    '你是裁判（Judge）。现在是终局阶段，请保持中立态度，仔细分析比较正方与反方的辩词，'
    '然后输出按照如下格式的判决（满分10分），不要用“```json```”代码块包裹，最外层不要有双引号。'
    '{"scores":{"Pro":{"argument":int,"evidence":int,"rebuttal":int,"clarity":int},'
    '"Con":{"argument":int,"evidence":int,"rebuttal":int,"clarity":int}},'
    '"verdict":"Pro|Con","rationale":"简要文字总结"}'
)

# ===== Agent Class =====
@dataclass
class DebateAgent:
    name: str
    llm: OpenAICompatLLM
    system_prompt: str
    extra_context: str = ""

    def speak(
        self,
        topic: str,
        round_idx: int,
        max_rounds: int,
        transcript_text: str,
        remaining_turns: int = None,
        is_final_turn: bool = None,
    ) -> str:
        guidance = ""
        if remaining_turns is not None and is_final_turn is not None:
            guidance = (
                f"[回合信息]\n"
                f"- 总轮数: {max_rounds}\n"
                f"- 已完成轮数: {round_idx}\n"
                f"- 你本方剩余发言次数(含本次): {remaining_turns}\n"
                f"- 是否是你本方最后一次发言: {'是' if is_final_turn else '否'}\n"
            )

        closing_rules = (
            "\n[最后一次发言策略]\n"
            "1) 明确立场，给出≤3条最强要点\n"
            "2) 逐条点名反驳对手关键主张\n"
            "3) 用1-2句完成收束，不引入全新复杂证据\n"
        ) if is_final_turn else ""

        non_final_rules = (
            "\n[非最后一次发言策略]\n"
            "1) 推进一个核心论据并配简要证据\n"
            "2) 精准反驳对方上轮1-2个关键点\n"
            "3) 留出后续空间，勿一次抛太多新线索\n"
        ) if (is_final_turn is not None and not is_final_turn) else ""

        user_prompt = (
            f"命题：{topic}\n"
            f"{guidance}"
            + (f"\n[你的私有上下文]\n{self.extra_context}\n" if self.extra_context else "")
            + "\n[完整历史]\n" + transcript_text
            + "\n\n请按照你的角色定位发言。"
            + non_final_rules
            + closing_rules
        )

        return self.llm.invoke([
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ])


# ===== Factory =====
def build_agents(
    pro_key: str,
    pro_url: str,
    pro_model: str,
    pro_temp: float,
    con_key: str,
    con_url: str,
    con_model: str,
    con_temp: float,
    judge_key: str,
    judge_url: str,
    judge_model: str,
    judge_temp: float,
) -> Dict[str, DebateAgent]:
    """
    构建 Pro / Con / Judge 三个 Agent。
    """
    return {
        "Pro": DebateAgent(
            name="Pro",
            llm=OpenAICompatLLM(pro_key, pro_url, pro_model, pro_temp),
            system_prompt=SYS_PRO,
        ),
        "Con": DebateAgent(
            name="Con",
            llm=OpenAICompatLLM(con_key, con_url, con_model, con_temp),
            system_prompt=SYS_CON,
        ),
        "Judge": DebateAgent(
            name="Judge",
            llm=OpenAICompatLLM(judge_key, judge_url, judge_model, judge_temp),
            system_prompt=SYS_JUDGE_FINAL,
        ),
    }
