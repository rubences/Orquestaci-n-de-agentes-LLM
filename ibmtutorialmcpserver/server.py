"""Simple MCP server that exposes a single tool to search IBM tutorials.

How to run:
1) Install dependencies: pip install -r requirements.txt
2) Start the server from this folder: fastmcp run server.py
"""

from __future__ import annotations

from typing import Any

import requests
from fastmcp import FastMCP

DOCS_INDEX_URL = (
    "https://raw.githubusercontent.com/IBM/ibmdotcom-tutorials/refs/heads/main/docs_index.json"
)

mcp = FastMCP("IBM Tutorials")


@mcp.tool
def search_ibmtutorials(query: str) -> str:
    """Search IBM tutorials by title or URL in a public JSON index."""
    try:
        response = requests.get(DOCS_INDEX_URL, timeout=10)
        response.raise_for_status()
        tutorials = response.json()

        if not isinstance(tutorials, list):
            return "Unexpected tutorials payload format."

        query_lower = query.lower().strip()
        relevant_tutorials: list[dict[str, Any]] = []

        for tutorial in tutorials:
            if not isinstance(tutorial, dict):
                continue
            title = str(tutorial.get("title", "")).lower()
            url_path = str(tutorial.get("url", "")).lower()
            if query_lower in title or query_lower in url_path:
                relevant_tutorials.append(tutorial)

        if not relevant_tutorials:
            return f"No IBM tutorials found matching '{query}'."

        result_lines = [
            f"Found {len(relevant_tutorials)} tutorial(s) matching '{query}':",
            "",
        ]

        for i, tutorial in enumerate(relevant_tutorials, 1):
            title = tutorial.get("title", "No title")
            tutorial_url = tutorial.get("url", "No URL")
            date = tutorial.get("date", "No date")
            author = tutorial.get("author", "")

            result_lines.append(f"{i}. **{title}**")
            result_lines.append(f"   URL: {tutorial_url}")
            result_lines.append(f"   Date: {date}")
            if author:
                result_lines.append(f"   Author: {author}")
            result_lines.append("")

        return "\n".join(result_lines)

    except requests.exceptions.RequestException as exc:
        return f"Error fetching tutorials from GitHub: {exc}"
    except ValueError as exc:
        return f"Error parsing JSON data: {exc}"
    except Exception as exc:
        return f"Error searching IBM tutorials: {exc}"


if __name__ == "__main__":
    mcp.run()
