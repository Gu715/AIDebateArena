# graph.py
import os, json, re
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from agents import build_agents
from datetime import datetime
from config import RUNS_DIR

DebateState = Dict[str, Any]

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def transcript_text(state: DebateState) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in state["transcript"])

def save_state(state: DebateState):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{state['run_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ----------------- Robust JSON parsing for Judge output -----------------

_CODE_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*(.*?)\s*```$", re.DOTALL)

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    m = _CODE_FENCE_RE.match(s)
    if m:
        return m.group(1).strip()
    # 兼容形如 ```json\n...\n``` 或 ```\n...\n``` 的手动拆分
    if s.startswith("```"):
        s2 = s[3:]
        # 去掉可能的语言标签
        if s2.lower().startswith("json"):
            s2 = s2[4:]
        # 去掉首行，直到第一个换行
        if "\n" in s2:
            s2 = s2.split("\n", 1)[1]
        # 去掉末尾 ```
        if s2.rstrip().endswith("```"):
            s2 = s2.rstrip()[:-3]
        return s2.strip()
    return s

def _parse_json_loose(text: str) -> Dict[str, Any]:
    """
    尝试把裁判输出解析成 dict：
    1) 去围栏 -> 直接 json.loads
       - 如果结果是 str，则再 loads 一次（处理字符串包裹的 JSON）
    2) 抽取最可能的 {...} 子串，多次收缩尝试
    3) 反转义再解析（处理 {\"k\":\"v\"}）
    4) 全失败 -> {"raw": 原文}
    """
    original = text
    s = _strip_code_fence(text)

    # 1) 直接解析
    try:
        obj = json.loads(s)
        if isinstance(obj, str):
            # 字符串里可能又是 JSON
            try:
                obj2 = json.loads(obj)
                if isinstance(obj2, dict):
                    return obj2
                return {"raw": original, "parsed_string": obj}
            except Exception:
                return {"raw": original, "parsed_string": obj}
        if isinstance(obj, dict):
            return obj
        # 非 dict（例如 list）也视为成功，但包一层给上层用
        return {"parsed": obj}
    except Exception:
        pass

    # 2) 在文本中抽取最可能的 JSON 对象子串
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        segment = s[first:last+1]
        # 直接尝试
        try:
            obj = json.loads(segment)
            if isinstance(obj, dict):
                return obj
            return {"parsed": obj}
        except Exception:
            # 末尾多余字符/括号：从右向左收缩尝试
            for end in range(last, first, -1):
                try:
                    obj = json.loads(s[first:end+1])
                    if isinstance(obj, dict):
                        return obj
                    return {"parsed": obj}
                except Exception:
                    continue

    # 3) 尝试反转义（处理 {\"a\":1} 这类）
    try:
        unescaped = s.encode("utf-8").decode("unicode_escape")
        obj = json.loads(unescaped)
        if isinstance(obj, dict):
            return obj
        return {"parsed": obj}
    except Exception:
        pass

    # 4) 彻底失败
    return {"raw": original}

# -------- build_app --------
def build_app(agents_factory=None):
    AGENTS = build_agents() if agents_factory is None else agents_factory()

    def pro_node(state: DebateState) -> DebateState:
        remaining = state["max_rounds"] - state["round"]
        msg = AGENTS["Pro"].speak(
            state["topic"], state["round"], state["max_rounds"],
            transcript_text(state), remaining_turns=remaining, is_final_turn=(remaining == 1)
        )
        state["transcript"].append({"ts": now_iso(), "role": "Pro", "content": msg})
        save_state(state)
        return state

    def con_node(state: DebateState) -> DebateState:
        remaining = state["max_rounds"] - state["round"]
        msg = AGENTS["Con"].speak(
            state["topic"], state["round"], state["max_rounds"],
            transcript_text(state), remaining_turns=remaining, is_final_turn=(remaining == 1)
        )
        state["transcript"].append({"ts": now_iso(), "role": "Con", "content": msg})
        state["round"] += 1
        save_state(state)
        return state

    def judge_final_node(state: DebateState) -> DebateState:
        final_view = transcript_text(state)
        # 让 Judge 感知“终局”
        msg = AGENTS["Judge"].speak(
            state["topic"], state["round"], state["max_rounds"], final_view,
            remaining_turns=0, is_final_turn=True
        )
        state["transcript"].append({"ts": now_iso(), "role": "Judge", "content": msg})
        state["final_verdict"] = _parse_json_loose(msg)
        save_state(state)
        return state

    def route_after_pro(_: DebateState) -> str:
        return "con"

    def route_after_con(state: DebateState) -> str:
        return "judge_final" if state["round"] >= state["max_rounds"] else "pro"

    g = StateGraph(DebateState)
    g.add_node("pro", pro_node)
    g.add_node("con", con_node)
    g.add_node("judge_final", judge_final_node)
    g.set_entry_point("pro")
    g.add_conditional_edges("pro", route_after_pro, {"con": "con"})
    g.add_conditional_edges("con", route_after_con, {"pro": "pro", "judge_final": "judge_final"})
    g.add_edge("judge_final", END)
    return g.compile()
