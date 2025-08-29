# ui_app.py
"""
Streamlit UI for DebateAgentTeam
- 增量渲染：Host 先显示；每个 Agent 说完一次就即时渲染；等待时提示“正在回答/正在裁决”
- Pro/Con/Judge/Host 颜色区分（绿/红/蓝/灰）
- 判决展示：胜者高亮 + 双方分数卡片 + 交互式雷达图（argument/evidence/rebuttal/clarity）+ 理由
- “检查设置”推荐但不强制；仅检查通过才保存到 settings.json
- 历史对局：打开 runs/*.json，查看并下载
- 持久化 LLM 配置到 settings.json；启动时自动读取并预填

运行：
    pip install streamlit plotly
    streamlit run ui_app.py
需要同目录：config.py, agents.py, graph.py, llm.py, runs/
"""

from __future__ import annotations
import uuid
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import streamlit as st
import plotly.graph_objects as go

from graph import build_app, now_iso, save_state  # type: ignore
from agents import build_agents  # type: ignore
from config import RUNS_DIR  # type: ignore
from llm import ChatMessage  # 健康检查最小请求

# -------- Paths --------
RUNS_PATH = Path(RUNS_DIR)
RUNS_PATH.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = Path("settings.json")  # 持久化 LLM 配置

# -------- Session helpers --------
def _reset_settings_ok():
    st.session_state["settings_ok"] = False
    st.session_state["settings_fingerprint"] = None

# 渲染去重：维护一个已渲染键的列表（用于新对局时清空）
if "rendered_keys" not in st.session_state:
    st.session_state["rendered_keys"] = []

# 保留最近一次“运行得到的最终状态”
if "last_run_state" not in st.session_state:
    st.session_state["last_run_state"] = None

# 保留“从文件打开的状态”
if "opened_run_data" not in st.session_state:
    st.session_state["opened_run_data"] = None

if "settings_ok" not in st.session_state:
    _reset_settings_ok()

# -------- Settings persistence --------
def default_settings() -> Dict[str, Any]:
    return {
        # Pro
        "pro_key": "your-api-key",
        "pro_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "pro_model": "qwen2.5-7b-instruct",
        "pro_temp": 0.7,
        # Con
        "con_key": "your-api-key",
        "con_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "con_model": "qwen2.5-7b-instruct",
        "con_temp": 0.7,
        # Judge
        "judge_key": "your-api-key",
        "judge_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "judge_model": "qwen2.5-14b-instruct",
        "judge_temp": 0.2,
    }

def load_settings() -> Dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            base = default_settings()
            base.update({k: data.get(k, base[k]) for k in base.keys()})
            return base
        except Exception:
            return default_settings()
    return default_settings()

def save_settings(data: Dict[str, Any]):
    allowed = default_settings().keys()
    payload = {k: data.get(k) for k in allowed}  # 只保存白名单字段
    SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# -------- Utils --------
def load_run_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def list_runs() -> Dict[str, Path]:
    files = list(RUNS_PATH.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {p.stem: p for p in files}

ROLE_COLORS = {
    "Host": "#9ca3af",   # 灰
    "Pro":  "#22c55e",   # 绿
    "Con":  "#ef4444",   # 红
    "Judge":"#3b82f6",   # 蓝
}

def _msg_key(m: Dict[str, Any]) -> str:
    # 以 (ts, role, content_hash) 作为唯一键，避免重复渲染
    ts = m.get("ts", "")
    role = m.get("role", "")
    content = m.get("content", "")
    return f"{ts}|{role}|{hash(content)}"

def render_one_message(msg: Dict[str, Any]):
    role = msg.get("role", "?")
    ts = msg.get("ts", "")
    content = msg.get("content", "")
    color = ROLE_COLORS.get(role, "#dddddd")

    with st.chat_message(role):
        # 时间条（HTML）
        st.markdown(
            f"<div style='border-left:6px solid {color};padding-left:12px;"
            f"color:#6b7280;font-size:0.85rem'>{ts}</div>",
            unsafe_allow_html=True,
        )
        # 正文（Markdown 解析，支持 **加粗**）
        st.markdown(content)

def render_transcript(transcript: List[Dict[str, Any]], start_idx: int = 0):
    # 去重渲染：只有未渲染过的消息才显示
    rendered = set(st.session_state.get("rendered_keys", []))
    for i in range(start_idx, len(transcript)):
        m = transcript[i]
        k = _msg_key(m)
        if k in rendered:
            continue
        render_one_message(m)
        rendered.add(k)
    st.session_state["rendered_keys"] = list(rendered)

def winner_badge(text: str):
    st.markdown(
        f"<div style='padding:10px 14px;background:#0ea5e9;color:white;display:inline-block;"
        f"border-radius:8px;font-weight:600'>Winner: {text}</div>",
        unsafe_allow_html=True
    )

def scores_two_columns(pro_score: int, con_score: int):
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Pro", pro_score)
    with c2:
        st.metric("Con", con_score)

def radar_chart_interactive(scores_pro: Dict[str,int], scores_con: Dict[str,int], show_pro: bool = True, show_con: bool = True):
    labels = ["argument", "evidence", "rebuttal", "clarity"]
    pro_vals = [scores_pro.get(k, 0) for k in labels]
    con_vals = [scores_con.get(k, 0) for k in labels]

    # 闭合首尾点
    labels_loop = labels + [labels[0]]
    pro_loop = pro_vals + [pro_vals[0]]
    con_loop = con_vals + [con_vals[0]]

    fig = go.Figure()
    if show_pro:
        fig.add_trace(go.Scatterpolar(
            r=pro_loop,
            theta=labels_loop,
            fill='toself',
            name='Pro',
            line=dict(color='#22c55e'),
            fillcolor='rgba(34,197,94,0.2)',
            hovertemplate='Pro · %{theta}: %{r}<extra></extra>'
        ))
    if show_con:
        fig.add_trace(go.Scatterpolar(
            r=con_loop,
            theta=labels_loop,
            fill='toself',
            name='Con',
            line=dict(color='#ef4444'),
            fillcolor='rgba(239,68,68,0.2)',
            hovertemplate='Con · %{theta}: %{r}<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 10], tickvals=[2,4,6,8,10])
        ),
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        height=320  # 紧凑
    )

    st.plotly_chart(fig, use_container_width=True)

def render_verdict(verdict: Dict[str, Any]):
    st.subheader("Final Verdict")
    if not verdict:
        st.info("尚无裁决（final_verdict 为空）。")
        return

    scores = verdict.get("scores", {})
    scores_pro = scores.get("Pro", {})
    scores_con = scores.get("Con", {})
    verdict_winner = verdict.get("verdict", "?")
    rationale = verdict.get("rationale", "")

    winner_badge(verdict_winner)

    def sum_scores(d: Dict[str,int]) -> int:
        return int(d.get("argument",0)+d.get("evidence",0)+d.get("rebuttal",0)+d.get("clarity",0))
    total_pro = sum_scores(scores_pro)
    total_con = sum_scores(scores_con)
    scores_two_columns(total_pro, total_con)

    # 交互式雷达图：可选显示 Pro/Con（用固定 key，交互只影响 Plotly，不清空会话）
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Score Radar**")
    with cols[1]:
        show_pro = st.checkbox("Show Pro", value=True, key="radar_show_pro")
    with cols[2]:
        show_con = st.checkbox("Show Con", value=True, key="radar_show_con")
    radar_chart_interactive(scores_pro, scores_con, show_pro=show_pro, show_con=show_con)

    if rationale:
        st.markdown("**Rationale**")
        st.write(rationale)

def fingerprint_settings(values: Tuple[Any, ...]) -> str:
    return "|".join(map(lambda x: "" if x is None else str(x), values))

def quick_check_agent(agent) -> Tuple[bool, str]:
    try:
        resp = agent.llm.invoke([
            ChatMessage(role="system", content="健康检查器"),
            ChatMessage(role="user", content="仅回复 OK"),
        ])
        if isinstance(resp, str) and len(resp.strip()) > 0:
            return True, resp.strip()
        return False, "空响应"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# -------- Page --------
st.set_page_config(page_title="AI Debate Arena", layout="wide")
st.title("AI Debate Arena")
st.caption("运行多轮 Pro vs Con 辩论，并由 Judge 终局裁决。支持增量渲染、配置持久化与可视化评分。")

# 读取 settings.json 并预填
_init_settings = load_settings()

# 侧边栏：新对局 + 设置 + 维护 + 历史
def _invalidate_check():
    # 只有调整 Agent 配置才使检查失效
    _reset_settings_ok()

with st.sidebar:
    st.header("New Debate")
    topic = st.text_area("Topic", value="是否应当在所有中学强制开设编程必修课？", height=80)
    max_rounds = st.number_input("Rounds (each = Pro+Con)", min_value=1, max_value=10, value=3, step=1)

    st.divider()
    st.header("Agent Settings (required)")

    # Pro
    with st.expander("Pro Agent", expanded=True):
        pro_key = st.text_input("Pro API Key", value=_init_settings.get("pro_key",""), type="password", key="pro_key", on_change=_invalidate_check)
        pro_url = st.text_input("Pro Base URL", value=_init_settings.get("pro_url",""), key="pro_url", on_change=_invalidate_check)
        pro_model = st.text_input("Pro Model", value=_init_settings.get("pro_model","qwen2.5-7b-instruct"), key="pro_model", on_change=_invalidate_check)
        pro_temp = st.number_input("Pro Temperature", min_value=0.0, max_value=2.0, value=float(_init_settings.get("pro_temp",0.7)), step=0.1, key="pro_temp", on_change=_invalidate_check)

    # Con
    with st.expander("Con Agent", expanded=True):
        con_key = st.text_input("Con API Key", value=_init_settings.get("con_key",""), type="password", key="con_key", on_change=_invalidate_check)
        con_url = st.text_input("Con Base URL", value=_init_settings.get("con_url",""), key="con_url", on_change=_invalidate_check)
        con_model = st.text_input("Con Model", value=_init_settings.get("con_model","qwen2.5-7b-instruct"), key="con_model", on_change=_invalidate_check)
        con_temp = st.number_input("Con Temperature", min_value=0.0, max_value=2.0, value=float(_init_settings.get("con_temp",0.7)), step=0.1, key="con_temp", on_change=_invalidate_check)

    # Judge
    with st.expander("Judge Agent", expanded=True):
        judge_key = st.text_input("Judge API Key", value=_init_settings.get("judge_key",""), type="password", key="judge_key", on_change=_invalidate_check)
        judge_url = st.text_input("Judge Base URL", value=_init_settings.get("judge_url",""), key="judge_url", on_change=_invalidate_check)
        judge_model = st.text_input("Judge Model", value=_init_settings.get("judge_model","qwen2.5-14b-instruct"), key="judge_model", on_change=_invalidate_check)
        judge_temp = st.number_input("Judge Temperature", min_value=0.0, max_value=2.0, value=float(_init_settings.get("judge_temp",0.2)), step=0.1, key="judge_temp", on_change=_invalidate_check)

    # 指纹只包含 Agent 配置（不含 Topic/Rounds）
    current_fp = fingerprint_settings((
        pro_key, pro_url, pro_model, pro_temp,
        con_key, con_url, con_model, con_temp,
        judge_key, judge_url, judge_model, judge_temp,
    ))

    st.divider()
    st.header("Settings Check")
    autosave = st.checkbox("检查通过后自动保存到 settings.json", value=True)

    if st.button("Check Settings", use_container_width=True):
        # 必填校验
        missing = []
        if not pro_key: missing.append("Pro API Key")
        if not pro_url: missing.append("Pro Base URL")
        if not con_key: missing.append("Con API Key")
        if not con_url: missing.append("Con Base URL")
        if not judge_key: missing.append("Judge API Key")
        if not judge_url: missing.append("Judge Base URL")
        if missing:
            st.error("以下字段必填： " + "、".join(missing))
            _reset_settings_ok()
        else:
            try:
                agents = build_agents(
                    pro_key, pro_url, pro_model, float(pro_temp),
                    con_key, con_url, con_model, float(con_temp),
                    judge_key, judge_url, judge_model, float(judge_temp),
                )
                ok_all = True
                for name in ("Pro", "Con", "Judge"):
                    ok, info = quick_check_agent(agents[name])
                    if ok:
                        st.success(f"{name} OK")
                    else:
                        ok_all = False
                        st.error(f"{name} FAILED: {info}")
                if ok_all:
                    st.session_state["settings_ok"] = True
                    st.session_state["settings_fingerprint"] = current_fp
                    st.success("All agents passed.")
                    if autosave:
                        save_settings({
                            "pro_key": pro_key, "pro_url": pro_url, "pro_model": pro_model, "pro_temp": float(pro_temp),
                            "con_key": con_key, "con_url": con_url, "con_model": con_model, "con_temp": float(con_temp),
                            "judge_key": judge_key, "judge_url": judge_url, "judge_model": judge_model, "judge_temp": float(judge_temp),
                        })
                        st.info(f"Settings saved to {SETTINGS_PATH}")
                else:
                    _reset_settings_ok()
            except Exception as e:
                _reset_settings_ok()
                st.error(f"构建或检查失败：{type(e).__name__}: {e}")

    # Run Debate 不强制检查
    run_button = st.button("Run Debate", use_container_width=True)
    st.caption("建议：修改 Agent 配置后可点击 “Check Settings” 做一次快速检查并保存设置；修改 Topic/Rounds 不影响检查状态。")

    st.divider()
    st.header("Maintenance")
    if st.button("Clear UI State", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.button("Delete all runs/*", use_container_width=True):
        shutil.rmtree(RUNS_PATH, ignore_errors=True)
        RUNS_PATH.mkdir(parents=True, exist_ok=True)
        st.success("All runs cleared.")

    st.divider()
    st.header("Previous Runs")
    # 刷新按钮
    if st.button("Refresh list", use_container_width=True):
        st.rerun()

    runs = list_runs()
    options = list(runs.keys())  # 不再包含空白项
    if options:
        sel = st.selectbox("Open a saved run", options=options, index=0, key="open_selectbox")
        if st.button("Open Selected", use_container_width=True):
            path = runs[sel]
            st.session_state["opened_run_data"] = load_run_json(path)
            st.session_state["last_run_state"] = None   # 打开文件时覆盖“最近运行”
            st.session_state["rendered_keys"] = []      # 重置去重
            st.success(f"Opened run: {sel}  ({path})")
            st.rerun()
    else:
        st.caption("No saved runs yet.")

# 主区域占位（提示在最上，状态在消息列表下）
info_ph = st.empty()
transcript_ph = st.container()
status_ph = st.empty()
verdict_ph = st.container()

# 如果用户点了“Open Selected”后，或曾经运行过一次，这里负责重绘 UI
def _render_from_state(state: Dict[str, Any]):
    with transcript_ph:
        st.session_state["rendered_keys"] = []  # 重绘前清空去重
        for m in state.get("transcript", []):
            render_one_message(m)
    with verdict_ph:
        render_verdict(state.get("final_verdict", {}))

# 打开历史对局（通过 session_state 持久化）
if st.session_state.get("opened_run_data") is not None and not run_button:
    data = st.session_state["opened_run_data"]
    _render_from_state(data)

# 启动新对局（增量渲染）
elif run_button:
    info_ph.empty()

    # 最小必填校验（key/url）
    missing = []
    if not pro_key: missing.append("Pro API Key")
    if not pro_url: missing.append("Pro Base URL")
    if not con_key: missing.append("Con API Key")
    if not con_url: missing.append("Con Base URL")
    if not judge_key: missing.append("Judge API Key")
    if not judge_url: missing.append("Judge Base URL")
    if missing:
        st.error("以下字段必填： " + "、".join(missing))
        st.stop()

    # 新对局开始：清空去重缓存、清掉已打开文件
    st.session_state["rendered_keys"] = []
    st.session_state["opened_run_data"] = None

    run_id = f"{topic}-{str(uuid.uuid4())[:6]}"
    init_state = {
        "topic": topic,
        "round": 0,
        "max_rounds": int(max_rounds),
        "transcript": [
            {"ts": now_iso(), "role": "Host", "content": f"欢迎观看AI辩论赛。流程：Pro <-> Con 若干轮后，由 Judge 终局裁决。本次辩论主题：{topic}"}
        ],
        "final_verdict": {},
        "run_id": run_id,
    }
    save_state(init_state)

    def agents_factory():
        return build_agents(
            pro_key, pro_url, pro_model, float(pro_temp),
            con_key, con_url, con_model, float(con_temp),
            judge_key, judge_url, judge_model, float(judge_temp),
        )

    app = build_app(agents_factory=agents_factory)

    # 先显示 Host，并立即提示“正方正在回答…”
    with transcript_ph:
        render_one_message(init_state["transcript"][0])
    status_ph.info("正方正在回答...")

    last_len = 1
    for update in app.stream(init_state, stream_mode="values"):  # type: ignore
        tr = update.get("transcript", [])
        if len(tr) > last_len:
            # 先渲染新增消息（带去重）
            with transcript_ph:
                render_transcript(tr, start_idx=last_len)
            last_role = tr[-1]["role"] if tr else ""

            # 计算下一位
            next_role = None
            if last_role == "Pro":
                next_role = "Con"
            elif last_role == "Con":
                # con 节点里已递增 round；满回合则到 Judge
                if update.get("round", 0) >= update.get("max_rounds", 0):
                    next_role = "Judge"
                else:
                    next_role = "Pro"
            elif last_role == "Judge":
                next_role = None  # 已结束

            # 更新状态提示（在消息下面）
            msg_map = {
                "Pro": "正方正在回答...",
                "Con": "反方正在回答...",
                "Judge": "裁判正在评价...",
            }
            if next_role:
                status_ph.info(msg_map[next_role])
            else:
                status_ph.empty()

            last_len = len(tr)

    status_ph.empty()

    # 最后一帧即最终状态
    final_state = update
    out_path = RUNS_PATH / f"{run_id}.json"
    out_path.write_text(json.dumps(final_state, ensure_ascii=False, indent=2), encoding="utf-8")

    # 把最终状态放进 session，后续任何交互（比如勾选 Show Pro/Con）都能重绘
    st.session_state["last_run_state"] = final_state

    with verdict_ph:
        render_verdict(final_state.get("final_verdict", {}))

    st.success(f"Saved to {out_path}")
    st.download_button(
        label="Download run JSON",
        data=out_path.read_bytes(),
        file_name=out_path.name,
        mime="application/json",
    )

# 没有“Open Selected”且没有刚运行，但存在“最近一次运行”的状态 —— 也要重绘，保证交互不清空
elif st.session_state.get("last_run_state") is not None:
    _render_from_state(st.session_state["last_run_state"])

else:
    info_ph.info("在左侧填写 Topic / Rounds 与三位 Agent 配置。建议先点击 “Check Settings” 做一次快速检查并保存设置；也可直接运行。")
