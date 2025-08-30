# ui_app.py
"""
Streamlit UI for DebateAgentTeam
- 增量渲染：Host 先显示；每个 Agent 说完一次就即时渲染；等待时提示“正在回答/正在裁决”
- Pro/Con/Judge/Host 颜色区分（绿/红/蓝/灰）
- 判决展示：胜者高亮 + 双方分数卡片 + 交互式雷达图（argument/evidence/rebuttal/clarity）+ 理由
- “检查设置”推荐但不强制；仅检查通过才保存到 settings.json
- 历史对局：打开 runs/*.json，查看
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

# 初始化运行/检查状态与缓存
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "is_checking" not in st.session_state:
    st.session_state["is_checking"] = False
if "check_feedback" not in st.session_state:
    st.session_state["check_feedback"] = []  # 最近一次检查的消息列表

# 其它状态
if "rendered_keys" not in st.session_state:
    st.session_state["rendered_keys"] = []
if "last_run_state" not in st.session_state:
    st.session_state["last_run_state"] = None
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
    payload = {k: data.get(k) for k in allowed}
    SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# -------- Utils --------
def load_run_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def list_runs() -> Dict[str, Path]:
    files = list(RUNS_PATH.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {p.stem: p for p in files}

ROLE_COLORS = {
    "Host": "#9ca3af",
    "Pro":  "#22c55e",
    "Con":  "#ef4444",
    "Judge":"#3b82f6",
}

def _msg_key(m: Dict[str, Any]) -> str:
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
        st.markdown(
            f"<div style='border-left:6px solid {color};padding-left:12px;"
            f"color:#6b7280;font-size:0.85rem'>{ts}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(content)

def render_transcript(transcript: List[Dict[str, Any]], start_idx: int = 0):
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
    labels_loop = labels + [labels[0]]
    pro_loop = pro_vals + [pro_vals[0]]
    con_loop = con_vals + [con_vals[0]]

    fig = go.Figure()
    if show_pro:
        fig.add_trace(go.Scatterpolar(
            r=pro_loop, theta=labels_loop, fill='toself', name='Pro',
            line=dict(color='#22c55e'), fillcolor='rgba(34,197,94,0.2)',
            hovertemplate='Pro · %{theta}: %{r}<extra></extra>'
        ))
    if show_con:
        fig.add_trace(go.Scatterpolar(
            r=con_loop, theta=labels_loop, fill='toself', name='Con',
            line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.2)',
            hovertemplate='Con · %{theta}: %{r}<extra></extra>'
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10], tickvals=[2,4,6,8,10])),
        showlegend=True, margin=dict(l=10, r=10, t=10, b=10), height=320
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

_init_settings = load_settings()

# ---------------- Sidebar（表单 + 运行/检查期禁用） ----------------
disabled_ui = st.session_state.get("is_running", False) or st.session_state.get("is_checking", False)

with st.sidebar:
    st.header("New Debate")
    with st.form("new_debate_form", clear_on_submit=False):
        topic = st.text_area(
            "Topic",
            value="是否应当在所有中学强制开设编程必修课？",
            height=80,
            key="topic_input",
            disabled=disabled_ui,
        )
        max_rounds = st.number_input(
            "Rounds (each = Pro+Con)",
            min_value=1, max_value=10, value=3, step=1,
            key="rounds_input",
            disabled=disabled_ui,
        )
        run_submit = st.form_submit_button("Run Debate", use_container_width=True, disabled=disabled_ui)

    st.divider()
    st.header("Agent Settings (required)")
    with st.form("agent_settings_form", clear_on_submit=False):
        with st.expander("Pro Agent", expanded=True):
            pro_key = st.text_input("Pro API Key", value=_init_settings.get("pro_key",""), type="password", key="pro_key", disabled=disabled_ui)
            pro_url = st.text_input("Pro Base URL", value=_init_settings.get("pro_url",""), key="pro_url", disabled=disabled_ui)
            pro_model = st.text_input("Pro Model", value=_init_settings.get("pro_model","qwen2.5-7b-instruct"), key="pro_model", disabled=disabled_ui)
            pro_temp = st.number_input("Pro Temperature", min_value=0.0, max_value=2.0, value=float(_init_settings.get("pro_temp",0.7)), step=0.1, key="pro_temp", disabled=disabled_ui)
        with st.expander("Con Agent", expanded=True):
            con_key = st.text_input("Con API Key", value=_init_settings.get("con_key",""), type="password", key="con_key", disabled=disabled_ui)
            con_url = st.text_input("Con Base URL", value=_init_settings.get("con_url",""), key="con_url", disabled=disabled_ui)
            con_model = st.text_input("Con Model", value=_init_settings.get("con_model","qwen2.5-7b-instruct"), key="con_model", disabled=disabled_ui)
            con_temp = st.number_input("Con Temperature", min_value=0.0, max_value=2.0, value=float(_init_settings.get("con_temp",0.7)), step=0.1, key="con_temp", disabled=disabled_ui)
        with st.expander("Judge Agent", expanded=True):
            judge_key = st.text_input("Judge API Key", value=_init_settings.get("judge_key",""), type="password", key="judge_key", disabled=disabled_ui)
            judge_url = st.text_input("Judge Base URL", value=_init_settings.get("judge_url",""), key="judge_url", disabled=disabled_ui)
            judge_model = st.text_input("Judge Model", value=_init_settings.get("judge_model","qwen2.5-14b-instruct"), key="judge_model", disabled=disabled_ui)
            judge_temp = st.number_input("Judge Temperature", min_value=0.0, max_value=2.0, value=float(_init_settings.get("judge_temp",0.2)), step=0.1, key="judge_temp", disabled=disabled_ui)

        current_fp = fingerprint_settings((
            pro_key, pro_url, pro_model, pro_temp,
            con_key, con_url, con_model, con_temp,
            judge_key, judge_url, judge_model, judge_temp,
        ))
        autosave = st.checkbox("检查通过后自动保存到 settings.json", value=True, key="autosave_checkbox", disabled=disabled_ui)
        check_submit = st.form_submit_button("Check Settings", use_container_width=True, disabled=disabled_ui)

    # 检查进度提示：就在“Check Settings”按钮正下方
    check_progress_box = st.empty()
    if st.session_state.get("is_checking", False):
        check_progress_box.info("正在检查 Agent 设置...")

    # 检查结果显示在 Agent 设置表单下方（一次性 Markdown）
    check_feedback_box = st.empty()
    if st.session_state.get("check_feedback") and not st.session_state.get("is_checking", False):
        emoji = {"success": "✅", "error": "❌", "info": "ℹ️"}
        lines = [f"{emoji.get(kind, '•')} {text}" for kind, text in st.session_state["check_feedback"]]
        check_feedback_box.markdown("\n\n".join(lines))

    # 点击“Check Settings”：进入检查态
    if check_submit and not disabled_ui:
        st.session_state["_pending_check"] = {
            "pro_key": st.session_state.get("pro_key"),
            "pro_url": st.session_state.get("pro_url"),
            "pro_model": st.session_state.get("pro_model"),
            "pro_temp": st.session_state.get("pro_temp"),
            "con_key": st.session_state.get("con_key"),
            "con_url": st.session_state.get("con_url"),
            "con_model": st.session_state.get("con_model"),
            "con_temp": st.session_state.get("con_temp"),
            "judge_key": st.session_state.get("judge_key"),
            "judge_url": st.session_state.get("judge_url"),
            "judge_model": st.session_state.get("judge_model"),
            "judge_temp": st.session_state.get("judge_temp"),
            "autosave": st.session_state.get("autosave_checkbox", True),
            "fp": current_fp,
        }
        st.session_state["is_checking"] = True
        st.session_state["check_feedback"] = []  # 清空旧反馈
        st.rerun()

    st.divider()
    st.header("Maintenance")
    clear_btn = st.button("Clear UI State", use_container_width=True, disabled=disabled_ui)
    if clear_btn:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    del_btn = st.button("Delete all runs/*", use_container_width=True, disabled=disabled_ui)
    if del_btn:
        shutil.rmtree(RUNS_PATH, ignore_errors=True)
        RUNS_PATH.mkdir(parents=True, exist_ok=True)
        st.success("All runs cleared.")

    st.divider()
    st.header("Previous Runs")
    refresh_btn = st.button("Refresh list", use_container_width=True, disabled=disabled_ui)
    if refresh_btn:
        st.rerun()

    runs = list_runs()
    options = list(runs.keys())
    if options:
        with st.form("open_run_form", clear_on_submit=False):
            sel = st.selectbox("Open a saved run", options=options, index=0, key="open_selectbox", disabled=disabled_ui)
            open_submit = st.form_submit_button("Open Selected", use_container_width=True, disabled=disabled_ui)
        if open_submit and not disabled_ui:
            path = runs[sel]
            st.session_state["opened_run_data"] = load_run_json(path)
            st.session_state["last_run_state"] = None
            st.session_state["rendered_keys"] = []
            st.rerun()
    else:
        st.caption("No saved runs yet.")

# 统一处理：Run 提交 → 进入运行态
if 'run_submit' in locals() and run_submit and not st.session_state.get("is_running", False) and not st.session_state.get("is_checking", False):
    st.session_state["_pending_run"] = {
        "topic": st.session_state.get("topic_input"),
        "max_rounds": st.session_state.get("rounds_input"),
        "pro": {
            "key": st.session_state.get("pro_key"), "url": st.session_state.get("pro_url"),
            "model": st.session_state.get("pro_model"), "temp": st.session_state.get("pro_temp"),
        },
        "con": {
            "key": st.session_state.get("con_key"), "url": st.session_state.get("con_url"),
            "model": st.session_state.get("con_model"), "temp": st.session_state.get("con_temp"),
        },
        "judge": {
            "key": st.session_state.get("judge_key"), "url": st.session_state.get("judge_url"),
            "model": st.session_state.get("judge_model"), "temp": st.session_state.get("judge_temp"),
        },
    }
    st.session_state["is_running"] = True
    st.rerun()

# ---------------- 主区域占位 ----------------
info_ph = st.empty()
transcript_ph = st.empty()
status_ph = st.empty()
verdict_ph = st.empty()

def _render_from_state(state: Dict[str, Any]):
    transcript_ph.empty()
    verdict_ph.empty()
    t_box = transcript_ph.container()
    with t_box:
        st.session_state["rendered_keys"] = []
        for m in state.get("transcript", []):
            render_one_message(m)
    v_box = verdict_ph.container()
    with v_box:
        render_verdict(state.get("final_verdict", {}))

# ---------------- 主逻辑分支（检查 > 运行 > 打开历史 > 最近状态 > 欢迎） ----------------

# A) 有挂起的“检查设置”任务 —— 检查期间保持主区域内容不变
if st.session_state.get("is_checking", False) and st.session_state.get("_pending_check"):
    payload = st.session_state["_pending_check"]
    autosave = bool(payload.get("autosave", True))
    current_fp = payload.get("fp", "")

    # 保持主区域可见：优先显示打开的历史，其次显示最近一次运行
    if st.session_state.get("opened_run_data") is not None:
        _render_from_state(st.session_state["opened_run_data"])
    elif st.session_state.get("last_run_state") is not None:
        _render_from_state(st.session_state["last_run_state"])

    feedback: List[Tuple[str, str]] = []
    try:
        missing = []
        if not payload.get("pro_key"): missing.append("Pro API Key")
        if not payload.get("pro_url"): missing.append("Pro Base URL")
        if not payload.get("con_key"): missing.append("Con API Key")
        if not payload.get("con_url"): missing.append("Con Base URL")
        if not payload.get("judge_key"): missing.append("Judge API Key")
        if not payload.get("judge_url"): missing.append("Judge Base URL")
        if missing:
            feedback.append(("error", "以下字段必填： " + "、".join(missing)))
            _reset_settings_ok()
        else:
            agents = build_agents(
                payload["pro_key"], payload["pro_url"], payload["pro_model"], float(payload["pro_temp"]),
                payload["con_key"], payload["con_url"], payload["con_model"], float(payload["con_temp"]),
                payload["judge_key"], payload["judge_url"], payload["judge_model"], float(payload["judge_temp"]),
            )
            ok_all = True
            for name in ("Pro", "Con", "Judge"):
                ok, info = quick_check_agent(agents[name])
                if ok:
                    feedback.append(("success", f"{name} OK"))
                else:
                    ok_all = False
                    feedback.append(("error", f"{name} FAILED: {info}"))
            if ok_all:
                st.session_state["settings_ok"] = True
                st.session_state["settings_fingerprint"] = current_fp
                feedback.append(("success", "All agents passed."))
                if autosave:
                    save_settings({
                        "pro_key": payload["pro_key"], "pro_url": payload["pro_url"], "pro_model": payload["pro_model"], "pro_temp": float(payload["pro_temp"]),
                        "con_key": payload["con_key"], "con_url": payload["con_url"], "con_model": payload["con_model"], "con_temp": float(payload["con_temp"]),
                        "judge_key": payload["judge_key"], "judge_url": payload["judge_url"], "judge_model": payload["judge_model"], "judge_temp": float(payload["judge_temp"]),
                    })
                    feedback.append(("info", f"Settings saved to {SETTINGS_PATH}"))
            else:
                _reset_settings_ok()
    except Exception as e:
        _reset_settings_ok()
        feedback.append(("error", f"构建或检查失败：{type(e).__name__}: {e}"))
    finally:
        st.session_state["check_feedback"] = feedback
        st.session_state["is_checking"] = False
        st.session_state["_pending_check"] = None
        st.rerun()

# B) 有挂起的“运行”任务
elif st.session_state.get("is_running", False) and st.session_state.get("_pending_run"):
    payload = st.session_state["_pending_run"]
    topic = payload["topic"]
    max_rounds = int(payload["max_rounds"])
    pro_key, pro_url, pro_model, pro_temp = payload["pro"]["key"], payload["pro"]["url"], payload["pro"]["model"], float(payload["pro"]["temp"])
    con_key, con_url, con_model, con_temp = payload["con"]["key"], payload["con"]["url"], payload["con"]["model"], float(payload["con"]["temp"])
    judge_key, judge_url, judge_model, judge_temp = payload["judge"]["key"], payload["judge"]["url"], payload["judge"]["model"], float(payload["judge"]["temp"])

    # 清空显示区（运行时才清空）
    info_ph.empty(); status_ph.empty(); transcript_ph.empty(); verdict_ph.empty()

    missing = []
    if not pro_key: missing.append("Pro API Key")
    if not pro_url: missing.append("Pro Base URL")
    if not con_key: missing.append("Con API Key")
    if not con_url: missing.append("Con Base URL")
    if not judge_key: missing.append("Judge API Key")
    if not judge_url: missing.append("Judge Base URL")
    if missing:
        st.error("以下字段必填： " + "、".join(missing))
        st.session_state["is_running"] = False
        st.session_state["_pending_run"] = None
        st.rerun()

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
            judge_key, judge_url, payload["judge"]["model"], float(payload["judge"]["temp"]),
        )

    app = build_app(agents_factory=agents_factory)

    transcript_box = transcript_ph.container()
    verdict_box = verdict_ph.container()

    with transcript_box:
        render_one_message(init_state["transcript"][0])
    status_ph.info("正方正在回答...")

    last_len = 1
    update = init_state

    try:
        for update in app.stream(init_state, stream_mode="values"):  # type: ignore
            tr = update.get("transcript", [])
            if len(tr) > last_len:
                with transcript_box:
                    render_transcript(tr, start_idx=last_len)

                last_role = tr[-1]["role"] if tr else ""
                next_role = None
                if last_role == "Pro":
                    next_role = "Con"
                elif last_role == "Con":
                    next_role = "Judge" if update.get("round", 0) >= update.get("max_rounds", 0) else "Pro"
                elif last_role == "Judge":
                    next_role = None

                msg_map = {"Pro": "正方正在回答...", "Con": "反方正在回答...", "Judge": "裁判正在评价..."}
                if next_role:
                    status_ph.info(msg_map[next_role])
                else:
                    status_ph.empty()

                last_len = len(tr)
    finally:
        final_state = update
        out_path = RUNS_PATH / f"{run_id}.json"
        out_path.write_text(json.dumps(final_state, ensure_ascii=False, indent=2), encoding="utf-8")
        st.session_state["last_run_state"] = final_state
        st.session_state["is_running"] = False
        st.session_state["_pending_run"] = None
        st.rerun()

# C) 打开历史对局（非运行/检查态）
elif st.session_state.get("opened_run_data") is not None and not st.session_state.get("is_running", False) and not st.session_state.get("is_checking", False):
    data = st.session_state["opened_run_data"]
    _render_from_state(data)

# D) 最近一次运行状态（非运行/检查态）
elif st.session_state.get("last_run_state") is not None and not st.session_state.get("is_running", False) and not st.session_state.get("is_checking", False):
    _render_from_state(st.session_state["last_run_state"])

# E) 欢迎提示
else:
    info_ph.info("在左侧填写 Topic / Rounds 与三位 Agent 配置。点击 “Run Debate” 或 “Check Settings” 后左侧将暂时锁定；检查期间主区域不会清空；完成后自动恢复，检查结果显示在 Agent 设置下方。")
