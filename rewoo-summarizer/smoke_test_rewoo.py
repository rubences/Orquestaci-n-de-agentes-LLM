from __future__ import annotations

from rewoo_pipeline import ReWOOPipeline, load_config


def main() -> int:
    config = load_config()
    pipeline_obj = ReWOOPipeline(config)

    task = "Summarize the novella The Metamorphosis"
    print("Running ReWOO smoke test with task:", task)
    summary = pipeline_obj.solver(task)

    print("\n--- Summary Preview ---\n")
    print(summary[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
