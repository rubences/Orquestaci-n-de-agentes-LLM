from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

from .tools.custom_tool import EscalationRiskTool, KeywordExtractionTool, SentimentAnalysisTool


class CustomerServiceAnalysisCrew:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.config_dir = base_dir / "src" / "customer_service_analyzer" / "config"
        self.agents_config = self._load_yaml(self.config_dir / "agents.yaml")
        self.tasks_config = self._load_yaml(self.config_dir / "tasks.yaml")

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _llm_model_name() -> str:
        # crewAI/litellm can route this model string using env credentials.
        return os.getenv("CREWAI_MODEL", "watsonx/ibm/granite-3-8b-instruct")

    def _build_agents(self) -> dict[str, Agent]:
        tools_text = [SentimentAnalysisTool(), KeywordExtractionTool(), EscalationRiskTool()]
        search_tool = SerperDevTool()

        transcript_agent = Agent(
            role=self.agents_config["transcript_analyzer"]["role"],
            goal=self.agents_config["transcript_analyzer"]["goal"],
            backstory=self.agents_config["transcript_analyzer"]["backstory"],
            tools=tools_text,
            llm=self._llm_model_name(),
            verbose=True,
        )

        qa_agent = Agent(
            role=self.agents_config["quality_assurance_specialist"]["role"],
            goal=self.agents_config["quality_assurance_specialist"]["goal"],
            backstory=self.agents_config["quality_assurance_specialist"]["backstory"],
            tools=[search_tool],
            llm=self._llm_model_name(),
            verbose=True,
        )

        report_agent = Agent(
            role=self.agents_config["report_generator"]["role"],
            goal=self.agents_config["report_generator"]["goal"],
            backstory=self.agents_config["report_generator"]["backstory"],
            llm=self._llm_model_name(),
            verbose=True,
        )

        return {
            "transcript_analyzer": transcript_agent,
            "quality_assurance_specialist": qa_agent,
            "report_generator": report_agent,
        }

    def _build_tasks(self, agents: dict[str, Agent]) -> list[Task]:
        transcript_task = Task(
            description=self.tasks_config["transcript_analysis"]["description"],
            expected_output=self.tasks_config["transcript_analysis"]["expected_output"],
            agent=agents[self.tasks_config["transcript_analysis"]["agent"]],
        )

        quality_task = Task(
            description=self.tasks_config["quality_evaluation"]["description"],
            expected_output=self.tasks_config["quality_evaluation"]["expected_output"],
            agent=agents[self.tasks_config["quality_evaluation"]["agent"]],
            context=[transcript_task],
        )

        report_task = Task(
            description=self.tasks_config["report_generation"]["description"],
            expected_output=self.tasks_config["report_generation"]["expected_output"],
            agent=agents[self.tasks_config["report_generation"]["agent"]],
            context=[transcript_task, quality_task],
            output_file=str(self.base_dir / "report.md"),
        )

        return [transcript_task, quality_task, report_task]

    def run(self, transcript: str) -> str:
        agents = self._build_agents()
        tasks = self._build_tasks(agents)
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff(inputs={"transcript": transcript})
        return str(result)
