from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from blindfugue.core import ChatModel, _as_text, extract_json_object
from blindfugue.tools import BaseTool


META_SYSTEM_PROMPT = (
    "You coordinate parallel search agents: first decompose the task, then "
    "synthesize their evidence into a final answer. Be concise, faithful to "
    "evidence, and follow any requested output format exactly."
)

SEARCH_SYSTEM_PROMPT = (
    "Search intensity is set to high. Please conduct thorough, multi-source "
    "research and provide comprehensive, well-cited results."
)


@dataclass
class Subtask:
    id: int
    title: str
    task: str


@dataclass
class SubagentResult:
    subagent_id: int
    name: str
    assigned_task: Subtask
    report: str
    error: str = ""


@dataclass
class NaiveOutcome:
    question: str
    answer: str
    confidence: float | None
    explanation: str
    decomposition: list[Subtask]
    subagent_results: list[SubagentResult]
    meta_decomposition_raw: str
    meta_synthesis_raw: str
    team_mode: str = "naive"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json_array(text: str) -> list[Any]:
    source = _as_text(text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(source):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


def _normalize_tasks(payload: Any, *, n: int, question: str) -> list[Subtask]:
    if isinstance(payload, dict):
        payload = next(
            (payload[key] for key in ("tasks", "subtasks", "agents") if isinstance(payload.get(key), list)),
            [],
        )
    tasks: list[Subtask] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                title = _as_text(item.get("title") or item.get("name"))
                task = _as_text(item.get("task") or item.get("description") or item.get("question"))
            else:
                title, task = "", _as_text(item)
            if task:
                index = len(tasks)
                tasks.append(Subtask(index, title or f"Subtask {index + 1}", task))
            if len(tasks) == n:
                break

    fallbacks = [
        "Find primary evidence and authoritative sources relevant to the whole question.",
        "Search for complementary sources, edge cases, and conflicting evidence.",
        "Verify the likely answer and the attributes required by the output format.",
    ]
    while len(tasks) < n:
        index = len(tasks)
        tasks.append(
            Subtask(
                index,
                f"Fallback Subtask {index + 1}",
                f"{fallbacks[index % len(fallbacks)]}\n\nOriginal question:\n{question}",
            )
        )
    return tasks


def _labeled_value(text: str, label: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else ""


def _parse_confidence(value: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    return min(100.0, max(0.0, float(match.group())))


class NaiveTeam:
    """Cabeza-style meta decomposition, parallel subagents, and meta synthesis."""

    def __init__(
        self,
        model: ChatModel,
        *,
        subagent_count: int = 2,
        tools: list[BaseTool] | None = None,
    ) -> None:
        if subagent_count < 1:
            raise ValueError("NaiveTeam requires at least one subagent.")
        self.model = model
        self.subagent_count = int(subagent_count)
        self.tools = list(tools or [])

    def _decompose(self, question: str) -> tuple[list[Subtask], str]:
        prompt = f"""
You are the meta-agent for a parallel research team. This is only the planning
phase, not the final-answer phase.
Split the question into exactly {self.subagent_count} complementary subproblems
for independent subagents.

Output contract:
- Return a raw JSON array only. Do not use Markdown fences.
- Each item must have exactly two keys: "title" and "task".
- Each task must be self-contained and preserve the relevant original constraints.
- Divide the problem into different useful pieces; do not merely paraphrase it.
- Give each subagent a clear deliverable that the final meta-agent can combine.

Original question:
{question}
""".strip()
        raw = self.model.complete(system=META_SYSTEM_PROMPT, user=prompt)
        payload = _extract_json_array(raw)
        if not payload:
            obj = extract_json_object(raw)
            payload = obj
        return _normalize_tasks(
            payload,
            n=self.subagent_count,
            question=question,
        ), raw

    def _run_subagent(self, question: str, subtask: Subtask) -> SubagentResult:
        prompt = f"""
You are SearchAgent-{subtask.id + 1} in a {self.subagent_count}-agent parallel
research baseline. Complete only your assigned investigation, not the whole
question by yourself.

Original question:
{question}

Assigned investigation:
Title: {subtask.title}
Task: {subtask.task}

Return a concise evidence-backed report containing your conclusion or partial
answer, useful sources or URLs, relevant facts or contradictions, and explicit
uncertainty for the final meta-agent.
""".strip()
        try:
            report = self.model.complete(
                system=SEARCH_SYSTEM_PROMPT,
                user=prompt,
                tools=self.tools,
            )
            error = ""
        except Exception as exc:
            report = ""
            error = f"{type(exc).__name__}: {exc}"
        return SubagentResult(
            subagent_id=subtask.id,
            name=f"SearchAgent-{subtask.id + 1}",
            assigned_task=subtask,
            report=report,
            error=error,
        )

    def _synthesize(
        self,
        question: str,
        results: list[SubagentResult],
    ) -> tuple[str, str, float | None, str]:
        reports = "\n\n".join(
            f"## {result.name}\n"
            f"Assigned task: {result.assigned_task.title}\n"
            f"{result.assigned_task.task}\n\n"
            f"Report:\n{result.report or 'No report.'}\n"
            f"Error: {result.error or 'none'}"
            for result in results
        )
        prompt = f"""
You are the meta-agent in the final synthesis phase. Combine the complementary
subagent reports into the final answer. Use tools only to fill a critical gap,
resolve a conflict, or verify decisive evidence.

Original question:
{question}

Subagent reports:
{reports}

Treat each report as evidence for its assigned subproblem. Align their
constraints and sources, resolve disagreements using the strongest evidence,
and do not invent unsupported facts. Follow the answer format requested in the
original question exactly.
""".strip()
        raw = self.model.complete(
            system=META_SYSTEM_PROMPT,
            user=prompt,
            tools=self.tools,
        )
        data = extract_json_object(raw)
        answer = _as_text(data.get("answer") or data.get("exact_answer"))
        if not answer:
            answer = _labeled_value(raw, "Exact Answer") or raw
        explanation = _as_text(data.get("explanation")) or _labeled_value(raw, "Explanation")
        confidence_value = data.get("confidence")
        if confidence_value is None:
            confidence_value = _labeled_value(raw, "Confidence")
        return answer, explanation, _parse_confidence(confidence_value), raw

    def run(self, question: str) -> NaiveOutcome:
        question = _as_text(question)
        if not question:
            raise ValueError("question must not be empty")

        decomposition, decomposition_raw = self._decompose(question)
        with ThreadPoolExecutor(max_workers=self.subagent_count) as executor:
            results = list(
                executor.map(
                    lambda task: self._run_subagent(question, task),
                    decomposition,
                )
            )
        results.sort(key=lambda result: result.subagent_id)
        answer, explanation, confidence, synthesis_raw = self._synthesize(question, results)
        return NaiveOutcome(
            question=question,
            answer=answer,
            confidence=confidence,
            explanation=explanation,
            decomposition=decomposition,
            subagent_results=results,
            meta_decomposition_raw=decomposition_raw,
            meta_synthesis_raw=synthesis_raw,
        )
