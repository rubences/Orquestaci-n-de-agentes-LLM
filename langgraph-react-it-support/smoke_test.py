from __future__ import annotations

from pathlib import Path

from src.langgraph_react_agent.agent import build_react_agent, run_query


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    agent, config = build_react_agent(base_dir)

    print("Running LangGraph ReAct smoke test...")

    q1 = "Create a ticket for email outage in marketing with high urgency"
    a1 = run_query(agent, config, q1)
    print("\nQ1:", q1)
    print("A1:", a1)

    q2 = "List all current tickets"
    a2 = run_query(agent, config, q2)
    print("\nQ2:", q2)
    print("A2:", a2)

    print("\nSmoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
