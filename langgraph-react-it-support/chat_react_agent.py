from __future__ import annotations

from pathlib import Path

from src.langgraph_react_agent.agent import build_react_agent, run_query


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    agent, config = build_react_agent(base_dir)

    print("LangGraph ReAct IT Support Agent")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        answer = run_query(agent, config, user_input)
        print("Agent:", answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
