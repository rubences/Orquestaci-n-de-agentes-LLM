from __future__ import annotations

import re
from collections import Counter


class Helper:
    POSITIVE_WORDS = {
        "help",
        "replacement",
        "refund",
        "thanks",
        "resolved",
        "sorry",
    }
    NEGATIVE_WORDS = {
        "frustrating",
        "dramatic",
        "blaming",
        "unbelievable",
        "angry",
        "fault",
        "chill",
        "sucks",
    }
    ESCALATION_PATTERNS = {
        "review",
        "supervisor",
        "not good enough",
        "contacting",
        "refund",
    }

    @staticmethod
    def analyze_sentiment(transcript: str) -> str:
        text = transcript.lower()
        pos = sum(1 for w in Helper.POSITIVE_WORDS if w in text)
        neg = sum(1 for w in Helper.NEGATIVE_WORDS if w in text)

        if neg > pos:
            return "Negative"
        if pos > neg:
            return "Positive"
        return "Neutral"

    @staticmethod
    def extract_keywords(transcript: str, top_n: int = 12) -> str:
        tokens = re.findall(r"[a-zA-Z']+", transcript.lower())
        stop = {
            "the",
            "and",
            "a",
            "to",
            "of",
            "it",
            "is",
            "i",
            "you",
            "we",
            "that",
            "this",
            "in",
            "for",
            "on",
            "be",
            "with",
            "my",
            "your",
            "but",
            "not",
        }
        filtered = [t for t in tokens if len(t) > 2 and t not in stop]
        common = Counter(filtered).most_common(top_n)
        return ", ".join([k for k, _ in common])

    @staticmethod
    def detect_escalation_risk(transcript: str) -> str:
        text = transcript.lower()
        hits = [p for p in Helper.ESCALATION_PATTERNS if p in text]
        if len(hits) >= 2:
            return f"High escalation risk. Indicators: {', '.join(hits)}"
        if len(hits) == 1:
            return f"Medium escalation risk. Indicator: {hits[0]}"
        return "Low escalation risk."
