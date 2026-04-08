from __future__ import annotations

import argparse

from rewoo_pipeline import ReWOOPipeline, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for ReWOO pipeline")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run with reduced loops/tokens for fast validation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()

    if args.quick_test:
        config.max_expert_loops = 1
        config.max_summary_loops = 1
        config.expert_max_new_tokens = min(config.expert_max_new_tokens, 48)
        config.summary_max_new_tokens = min(config.summary_max_new_tokens, 48)
        config.summary_do_sample = False

    pipeline_obj = ReWOOPipeline(config)

    task = "Summarize The Metamorphosis in 3 bullet points"
    print("Running ReWOO smoke test with task:", task)
    summary = pipeline_obj.solver(task)

    print("\n--- Summary Preview ---\n")
    print(summary[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
