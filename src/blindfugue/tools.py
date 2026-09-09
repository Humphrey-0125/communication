from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests


def _string_list(value: Any, *, field: str, limit: int = 5) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"{field} must be a string or an array of strings")
    result = [str(item).strip() for item in items if str(item).strip()]
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result[:limit]


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError


class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for evidence. Put complementary queries in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One to five web search queries.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "", endpoint: str = "https://google.serper.dev/search") -> None:
        self.api_key = api_key.strip()
        self.endpoint = endpoint

    def execute(self, query: Any) -> str:
        if not self.api_key:
            return "[search] unavailable: set SERPER_API_KEY in .env"
        try:
            queries = _string_list(query, field="query")
            blocks = [self._search_one(item) for item in queries]
            return "\n\n=======\n\n".join(blocks)
        except Exception as exc:
            return f"[search] error: {type(exc).__name__}: {exc}"

    def _search_one(self, query: str) -> str:
        response = requests.post(
            self.endpoint,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json={"q": query, "page": 1},
            timeout=(5, 30),
        )
        response.raise_for_status()
        rows = response.json().get("organic") or []
        lines = [f"Web results for: {query}"]
        for index, row in enumerate(rows[:8], start=1):
            title = str(row.get("title") or "Untitled")
            url = str(row.get("link") or "")
            snippet = str(row.get("snippet") or "")[:800]
            lines.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")
        if len(lines) == 1:
            lines.append("No results found.")
        return "\n\n".join(lines)


class VisitTool(BaseTool):
    name = "visit"
    description = "Read webpages returned by search and return their text as evidence."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One to three HTTP(S) URLs.",
            },
            "goal": {"type": "string", "description": "Evidence to find on these pages."},
        },
        "required": ["url", "goal"],
    }

    def __init__(
        self,
        api_key: str = "",
        reader_prefix: str = "https://r.jina.ai/",
        max_content_chars: int = 20_000,
    ) -> None:
        self.api_key = api_key.strip()
        self.reader_prefix = reader_prefix.rstrip("/") + "/"
        self.max_content_chars = max(1_000, int(max_content_chars))

    def execute(self, url: Any, goal: str) -> str:
        try:
            urls = _string_list(url, field="url", limit=3)
            return "\n\n=======\n\n".join(self._visit_one(item, goal) for item in urls)
        except Exception as exc:
            return f"[visit] error: {type(exc).__name__}: {exc}"

    def _visit_one(self, url: str, goal: str) -> str:
        if not url.startswith(("http://", "https://")):
            return f"[visit] invalid URL: {url}"
        headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.get(
            self.reader_prefix + url,
            headers=headers,
            timeout=(10, 60),
        )
        response.raise_for_status()
        content = response.text.strip()
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars] + "\n[content truncated]"
        return f"URL: {url}\nGoal: {goal}\n\n{content or '[empty page]'}"


class GoogleScholarTool(BaseTool):
    name = "google_scholar"
    description = "Search Google Scholar for academic publications and citations."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One to five scholarly search queries.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "", endpoint: str = "https://google.serper.dev/scholar") -> None:
        self.api_key = api_key.strip()
        self.endpoint = endpoint

    def execute(self, query: Any) -> str:
        if not self.api_key:
            return "[google_scholar] unavailable: set SERPER_API_KEY in .env"
        try:
            queries = _string_list(query, field="query")
            return "\n\n=======\n\n".join(self._search_one(item) for item in queries)
        except Exception as exc:
            return f"[google_scholar] error: {type(exc).__name__}: {exc}"

    def _search_one(self, query: str) -> str:
        response = requests.post(
            self.endpoint,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json={"q": query, "page": 1},
            timeout=(5, 30),
        )
        response.raise_for_status()
        rows = response.json().get("organic") or []
        lines = [f"Scholar results for: {query}"]
        for index, row in enumerate(rows[:8], start=1):
            title = str(row.get("title") or "Untitled")
            url = str(row.get("pdfUrl") or row.get("link") or "")
            publication = row.get("publicationInfo") or ""
            authors = str(publication.get("summary") or "") if isinstance(publication, dict) else str(publication)
            snippet = str(row.get("snippet") or "")[:800]
            lines.append(f"{index}. {title}\nURL: {url}\nPublication: {authors}\nSnippet: {snippet}")
        if len(lines) == 1:
            lines.append("No results found.")
        return "\n\n".join(lines)


_SAFE_IMPORTS = {"collections", "decimal", "fractions", "itertools", "json", "math", "re", "statistics"}
_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


def _validate_code(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            if any(name not in _SAFE_IMPORTS for name in names):
                raise ValueError("only standard calculation modules may be imported")
        if isinstance(node, ast.ImportFrom):
            names = [str(node.module or "").split(".")[0]]
            if any(name not in _SAFE_IMPORTS for name in names):
                raise ValueError("only standard calculation modules may be imported")
        if isinstance(node, ast.Name) and node.id in _BLOCKED_CALLS:
            raise ValueError(f"blocked operation: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is blocked")


class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = "Run short Python calculations. Network and file access are not allowed."
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python calculation code."}},
        "required": ["code"],
    }

    def __init__(self, timeout: int = 20, max_output_chars: int = 12_000) -> None:
        self.timeout = max(1, int(timeout))
        self.max_output_chars = max(1_000, int(max_output_chars))

    def execute(self, code: str) -> str:
        try:
            _validate_code(code)
            with tempfile.TemporaryDirectory(prefix="blindfugue-python-") as directory:
                child_env = {
                    key: value
                    for key in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP")
                    if (value := os.environ.get(key))
                }
                child_env["PYTHONIOENCODING"] = "utf-8"
                result = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    cwd=Path(directory),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    env=child_env,
                    check=False,
                )
            output = (result.stdout + result.stderr).strip() or "Finished execution."
            if len(output) > self.max_output_chars:
                output = output[: self.max_output_chars] + "\n[output truncated]"
            return output
        except subprocess.TimeoutExpired:
            return f"[code_interpreter] error: timed out after {self.timeout} seconds"
        except Exception as exc:
            return f"[code_interpreter] error: {type(exc).__name__}: {exc}"


def build_default_tools() -> list[BaseTool]:
    """Build four independent tools without importing Cabeza."""
    serper_key = os.environ.get("SERPER_API_KEY", "")
    jina_key = os.environ.get("JINA_API_KEY", "")
    return [
        SearchTool(api_key=serper_key),
        VisitTool(api_key=jina_key),
        GoogleScholarTool(api_key=serper_key),
        CodeInterpreterTool(),
    ]
