from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from blindfugue.core import BlindFugue, EpisodeStore, OpenAIChatModel, extract_json_object
from blindfugue.dataset import load_jsonl, run_dataset
from blindfugue.tools import (
    CodeInterpreterTool,
    GoogleScholarTool,
    SearchTool,
    VisitTool,
    build_default_tools,
)


class FakeModel:
    def __init__(self) -> None:
        self.received_tools: list[list[str]] = []

    def complete(self, *, system: str, user: str, tools=None) -> str:
        self.received_tools.append([tool.name for tool in (tools or [])])
        if "independent reasoning peer" in system:
            if "Peer-0" in user:
                return json.dumps(
                    {
                        "answer": "Attention Is All You Need; first author unknown.",
                        "evidence": ["The Transformer paper appeared in 2017."],
                        "uncertainty": "The first author is not verified.",
                        "private_note": "Identified the paper title and year.",
                        "need": {
                            "subgoal": "Verify authorship.",
                            "known_facts": ["Title is Attention Is All You Need."],
                            "missing_information": "Who is the first author?",
                            "evidence_type": "bibliographic evidence",
                            "failed_attempts": [],
                        },
                    }
                )
            return json.dumps(
                {
                    "answer": "The paper is Attention Is All You Need by Vaswani et al.",
                    "evidence": ["Ashish Vaswani is listed first."],
                    "uncertainty": "The exact venue is not checked.",
                    "private_note": "Contains bibliographic authorship evidence for the Transformer paper.",
                    "need": {
                        "subgoal": "Verify the title.",
                        "known_facts": ["Ashish Vaswani is the first author."],
                        "missing_information": "Confirm the exact paper title.",
                        "evidence_type": "bibliographic evidence",
                        "failed_attempts": [],
                    },
                }
            )
        if "hidden communication router" in system:
            index = json.loads(user.split("Hidden episode index:\n", 1)[1])
            return json.dumps(
                {"episode_id": index[0]["episode_id"], "reason": "It addresses the gap."}
            )
        if "delta reader" in system:
            if "Who is the first author?" in user:
                fact = "Ashish Vaswani is the first author."
            else:
                fact = "The exact title is Attention Is All You Need."
            return json.dumps({"useful": True, "delta": fact, "evidence": [fact]})
        if "revising your own answer" in system:
            return json.dumps(
                {"answer": "Attention Is All You Need; its first author is Ashish Vaswani."}
            )
        if "final aggregator" in system:
            return json.dumps(
                {"answer": "Attention Is All You Need; its first author is Ashish Vaswani.", "reason": "Both peers agree."}
            )
        raise AssertionError("Unexpected prompt")


class CoreTests(unittest.TestCase):
    def test_extract_json_object(self) -> None:
        self.assertEqual(extract_json_object("prefix {\"x\": 1} suffix"), {"x": 1})

    def test_store_hides_requester_episode(self) -> None:
        store = EpisodeStore()
        store.add(owner="Peer-0", note="a", raw_content="a")
        store.add(owner="Peer-1", note="b", raw_content="b")
        candidates = store.candidates_for("Peer-0")
        self.assertEqual([episode.owner for episode in candidates], ["Peer-1"])

    def test_end_to_end_without_network(self) -> None:
        model = FakeModel()
        result = BlindFugue(
            model,
            peer_count=2,
            tools=[CodeInterpreterTool()],
        ).run(
            "Identify the 2017 Transformer paper and its first author."
        )
        self.assertIn("Ashish Vaswani", result.answer)
        self.assertEqual(len(result.episodes), 2)
        self.assertEqual(len(result.peers), 2)
        for peer in result.peers:
            self.assertNotEqual(peer.peer_id, peer.route.source_owner)
            self.assertTrue(peer.delta.useful)
        self.assertEqual(model.received_tools.count(["code_interpreter"]), 2)
        self.assertTrue(all(not names for names in model.received_tools[2:]))

    def test_default_tool_names_and_code_execution(self) -> None:
        self.assertEqual(
            [tool.name for tool in build_default_tools()],
            ["search", "visit", "google_scholar", "code_interpreter"],
        )
        tool = CodeInterpreterTool()
        self.assertEqual(tool.execute("print(6 * 7)"), "42")
        self.assertIn("error", tool.execute("import os\nprint(os.getcwd())"))

    def test_web_tools_with_mocked_http(self) -> None:
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "organic": [
                    {
                        "title": "Evidence title",
                        "link": "https://example.com/evidence",
                        "snippet": "Evidence snippet",
                        "publicationInfo": {"summary": "A. Author, 2026"},
                    }
                ]
            },
            text="Full evidence page",
        )
        with patch("blindfugue.tools.requests.post", return_value=response):
            search_result = SearchTool(api_key="test").execute(["query"])
            scholar_result = GoogleScholarTool(api_key="test").execute(["paper"])
        with patch("blindfugue.tools.requests.get", return_value=response):
            visit_result = VisitTool().execute(
                ["https://example.com/evidence"],
                "Find the relevant fact.",
            )

        self.assertIn("https://example.com/evidence", search_result)
        self.assertIn("A. Author, 2026", scholar_result)
        self.assertIn("Full evidence page", visit_result)

    def test_openai_model_executes_tool_then_returns_text(self) -> None:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="code_interpreter",
                arguments=json.dumps({"code": "print(6 * 7)"}),
            ),
        )
        responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="", tool_calls=[tool_call]),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"answer": "42"}', tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            ),
        ]

        class FakeCompletions:
            def __init__(self) -> None:
                self.requests = []

            def create(self, **kwargs):
                self.requests.append(kwargs)
                return responses.pop(0)

        completions = FakeCompletions()
        model = OpenAIChatModel(
            model="fake",
            base_url="http://127.0.0.1:1/v1",
            max_tool_rounds=1,
        )
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        reply = model.complete(
            system="Return JSON.",
            user="Calculate 6 * 7.",
            tools=[CodeInterpreterTool()],
        )

        self.assertEqual(reply, '{"answer": "42"}')
        self.assertIn("tools", completions.requests[0])
        self.assertNotIn("tools", completions.requests[1])
        self.assertIn(
            "Tool budget reached",
            completions.requests[1]["messages"][-1]["content"],
        )
        tool_messages = [
            message
            for message in completions.requests[1]["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(tool_messages[0]["content"], "42")

    def test_dataset_slice_and_resume_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "eval.jsonl"
            output_path = root / "results.jsonl"
            rows = [
                {"id": "0", "question": "First question?", "golden_answers": "A"},
                {"id": "1", "question": "Second question?", "golden_answers": "B"},
                {"id": "2", "question": "Third question?", "golden_answers": "C"},
            ]
            dataset_path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            self.assertEqual(len(load_jsonl(dataset_path, limit=2)), 2)
            team = BlindFugue(FakeModel(), peer_count=2)
            first = run_dataset(
                team,
                dataset_path=dataset_path,
                output_path=output_path,
                limit=2,
            )
            second = run_dataset(
                team,
                dataset_path=dataset_path,
                output_path=output_path,
                limit=2,
            )
            saved = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(first["succeeded"], 2)
            self.assertEqual(second["skipped"], 2)
            self.assertEqual(len(saved), 2)


if __name__ == "__main__":
    unittest.main()
