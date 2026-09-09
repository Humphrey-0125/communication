from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from blindfugue.core import BlindFugue, OpenAIChatModel
from blindfugue.dataset import run_dataset
from blindfugue.tools import build_default_tools


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit("BLINDFUGUE_ENABLE_THINKING must be true or false.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal BlindFugue prototype.")
    parser.add_argument("question", nargs="?", help="Question all peers solve independently.")
    parser.add_argument("--dataset", default="", help="JSONL dataset path.")
    parser.add_argument("--limit", type=int, default=3, help="Number of dataset samples to run.")
    parser.add_argument("--output", default="outputs/bc_results.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--peers", type=int, default=2)
    parser.add_argument("--tools", choices=["all", "none"], default="all")
    parser.add_argument("--max-tool-rounds", type=int, default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    _load_env(project_root / ".env")
    args = _parser().parse_args(argv)

    model_name = args.model or os.environ.get("BLINDFUGUE_MODEL", "")
    base_url = args.base_url or os.environ.get("BLINDFUGUE_BASE_URL", "")
    api_key = args.api_key or os.environ.get("BLINDFUGUE_API_KEY", "EMPTY")
    if not model_name or not base_url:
        raise SystemExit(
            "Missing API configuration. Fill BLINDFUGUE_MODEL and "
            "BLINDFUGUE_BASE_URL in project/.env."
        )

    chat_model = OpenAIChatModel(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=float(os.environ.get("BLINDFUGUE_TEMPERATURE", "0.3")),
        max_tokens=int(os.environ.get("BLINDFUGUE_MAX_TOKENS", "1000")),
        timeout=float(os.environ.get("BLINDFUGUE_TIMEOUT", "120")),
        enable_thinking=_optional_bool(os.environ.get("BLINDFUGUE_ENABLE_THINKING")),
        max_tool_rounds=(
            args.max_tool_rounds
            if args.max_tool_rounds is not None
            else int(os.environ.get("BLINDFUGUE_MAX_TOOL_ROUNDS", "4"))
        ),
    )
    tools = build_default_tools() if args.tools == "all" else []
    team = BlindFugue(chat_model, peer_count=args.peers, tools=tools)
    if tools:
        print("[blindfugue] peer tools: " + ", ".join(tool.name for tool in tools))
        if not os.environ.get("SERPER_API_KEY"):
            print("[blindfugue] warning: search and google_scholar need SERPER_API_KEY")
        if not os.environ.get("JINA_API_KEY"):
            print("[blindfugue] note: visit will use the rate-limited anonymous Jina endpoint")
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path

        def show_progress(index: int, total: int, message: str) -> None:
            print(f"[{index}/{total}] {message}", flush=True)

        summary = run_dataset(
            team,
            dataset_path=dataset_path,
            output_path=output_path,
            limit=args.limit,
            resume=not args.no_resume,
            progress=show_progress,
        )
        print("\n===== Dataset summary =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["failed"] == 0 else 1

    if not args.question:
        raise SystemExit("Provide a question or use --dataset PATH.")

    result = team.run(args.question)

    print("\n===== Final answer =====")
    print(result.answer)
    print("\n===== Communication trace =====")
    for peer in result.peers:
        source = peer.route.source_owner or "none"
        episode = peer.route.episode_id or "none"
        print(f"\n[{peer.peer_id}]")
        print(f"Need: {peer.need.missing_information}")
        print(f"Route: {episode} from {source}")
        print(f"Delta: {peer.delta.delta or '[no useful delta]'}")
        print(f"Revised answer: {peer.final_answer}")
    return 0
