# run.py
import argparse
import config
from graph import build_app, now_iso, save_state
from agents import build_agents
import uuid
import json

def main():
    parser = argparse.ArgumentParser(description="Run an AI Debate")
    parser.add_argument("--topic", type=str, required=True, help="辩论论题")
    parser.add_argument("--rounds", type=int, default=3, help="总轮数")
    args = parser.parse_args()

    agents_factory = lambda: build_agents(
        config.PRO_KEY, config.PRO_URL, config.PRO_MODEL, config.PRO_TEMP,
        config.CON_KEY, config.CON_URL, config.CON_MODEL, config.CON_TEMP,
        config.JUDGE_KEY, config.JUDGE_URL, config.JUDGE_MODEL, config.JUDGE_TEMP,
    )

    app = build_app(agents_factory=agents_factory)

    run_id = f"{args.topic}-{str(uuid.uuid4())[:6]}"
    intro = f"欢迎观看AI辩论赛。流程：Pro <-> Con 若干轮后，由 Judge 终局裁决。本次辩论主题：{args.topic}"

    print(intro)

    init_state = {
        "topic": args.topic,
        "round": 0,
        "max_rounds": args.rounds,
        "transcript": [
            {"ts": now_iso(), "role": "Host", 
             "content": intro}
        ],
        "final_verdict": {},
        "run_id": run_id,
    }

    final_state = app.invoke(init_state)

    print("\n=== 辩论过程 ===")
    for msg in final_state["transcript"]:
        if msg['role'] in ['Pro', 'Con']:
            print(f"[{msg['role']}] {msg['content']}\n")

    print("=== 最终裁决 ===")
    print(json.dumps(final_state["final_verdict"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
