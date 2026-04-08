from __future__ import annotations

from pathlib import Path

from src.customer_service_analyzer.crew import CustomerServiceAnalysisCrew


def load_transcript(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    transcript_path = base_dir / "data" / "transcript.txt"

    transcript = load_transcript(transcript_path)
    runner = CustomerServiceAnalysisCrew(base_dir=base_dir)
    result = runner.run(transcript=transcript)

    print("\n===== RESULTADO FINAL =====\n")
    print(result)
    print("\nReporte guardado en:", base_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
