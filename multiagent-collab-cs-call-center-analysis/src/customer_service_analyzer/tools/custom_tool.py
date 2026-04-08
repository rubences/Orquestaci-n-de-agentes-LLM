from __future__ import annotations

from crewai.tools import BaseTool

from .tool_helper import Helper


class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = "Determines overall sentiment from a customer service transcript."

    def _run(self, transcript: str) -> str:
        return Helper.analyze_sentiment(transcript)


class KeywordExtractionTool(BaseTool):
    name: str = "Keyword Extraction Tool"
    description: str = "Extracts key words and repeated themes from transcript text."

    def _run(self, transcript: str) -> str:
        return Helper.extract_keywords(transcript)


class EscalationRiskTool(BaseTool):
    name: str = "Escalation Risk Tool"
    description: str = "Flags escalation risk indicators from transcript language."

    def _run(self, transcript: str) -> str:
        return Helper.detect_escalation_risk(transcript)
