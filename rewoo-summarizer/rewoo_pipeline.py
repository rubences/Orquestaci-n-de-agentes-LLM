from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class ReWOOConfig:
    model_id: str
    serper_api_key: str
    num_results: int
    max_expert_loops: int
    max_summary_loops: int


def load_config() -> ReWOOConfig:
    load_dotenv()

    model_id = os.getenv("REWOO_MODEL_ID", "ibm-granite/granite-3.2-2b-instruct")
    serper_api_key = os.getenv("SERPER_API_KEY", "")
    num_results = int(os.getenv("REWOO_NUM_RESULTS", "3"))
    max_expert_loops = int(os.getenv("REWOO_MAX_EXPERT_LOOPS", "5"))
    max_summary_loops = int(os.getenv("REWOO_MAX_SUMMARY_LOOPS", "5"))

    if not serper_api_key:
        raise ValueError("Missing SERPER_API_KEY. Set it in environment or .env file.")

    return ReWOOConfig(
        model_id=model_id,
        serper_api_key=serper_api_key,
        num_results=num_results,
        max_expert_loops=max_expert_loops,
        max_summary_loops=max_summary_loops,
    )


class ReWOOPipeline:
    def __init__(self, config: ReWOOConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def query_serper(self, question: str, num_results: int | None = None) -> str:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.config.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": question, "num": num_results or self.config.num_results}

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        snippets = [item.get("snippet", "") for item in data.get("organic", [])]
        snippets = [s.strip() for s in snippets if s and s.strip()]
        return "\n".join(snippets) if snippets else "No relevant snippets found."

    def expert(self, question: str) -> str:
        context = self.query_serper(question)
        prompt = f"""You are a knowledgeable expert. Based ONLY on the context below, answer the question clearly and concisely in your own words.

Do NOT mention any sources or references.

Context:
{context}

Question: {question}

Answer:"""

        input_prompt = prompt
        generated_text = ""
        last_generated = ""

        for _ in range(self.config.max_expert_loops):
            outputs = self.generator(
                input_prompt,
                max_new_tokens=120,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            text = outputs[0]["generated_text"]
            new_text = text[len(input_prompt):].strip()

            if new_text == last_generated:
                break

            generated_text += new_text + " "
            input_prompt = prompt + generated_text
            last_generated = new_text

            if new_text.endswith((".", "!", "?")) and len(generated_text.split()) > 50:
                break

        return generated_text.strip()

    @staticmethod
    def planner(task: str) -> list[str]:
        topic = task.replace("Summarize", "").replace("the novella", "").strip()
        return [
            f"What is the main plot related to {topic}?",
            f"Who are the key characters in {topic}?",
            f"What themes are explored in {topic}?",
        ]

    def final_summarizer(self, task: str, sub_answers: dict[str, str]) -> str:
        insights = "\n".join(sub_answers.values())
        base_prompt = f"""You are an expert summarizer. Based on the following insights, write a fresh, concise summary of the text. The summary must be newly written and must end in a complete sentence with proper punctuation.

Task:
{task}

Insights:
{insights}

Summary:"""

        summary = ""
        current_prompt = base_prompt
        max_total_tokens = 400
        total_tokens_used = 0

        for _ in range(self.config.max_summary_loops):
            response = self.generator(
                current_prompt,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            chunk = response[0]["generated_text"][len(current_prompt):].strip()
            summary = (summary + " " + chunk).strip()
            total_tokens_used += len(chunk.split())

            if summary.endswith((".", "!", "?")) or total_tokens_used >= max_total_tokens:
                break

            current_prompt = base_prompt + summary

        return summary.strip()

    def solver(self, task: str) -> str:
        print(f"Planner: Breaking down '{task}' into sub-questions...\n")
        subquestions = self.planner(task)
        answers: dict[str, str] = {}

        for q in subquestions:
            print(f"Expert answering: {q}")
            ans = self.expert(q)
            print(f"Answer: {ans}\n")
            answers[q] = ans

        print("=== Final Summary ===\n")
        final_summary = self.final_summarizer(task, answers)
        print(final_summary)
        return final_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReWOO summarization pipeline with IBM Granite + Serper")
    parser.add_argument(
        "--task",
        default="Summarize the novella The Metamorphosis",
        help="High-level task to solve with ReWOO",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config()
    pipeline_obj = ReWOOPipeline(config)
    pipeline_obj.solver(args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
