from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Protocol


class Runnable(Protocol):
    def run(self, question: str) -> Any:
        ...


def load_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the first `limit` valid question records from a JSONL file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not str(value.get("question") or "").strip():
                raise ValueError(f"Invalid dataset record at line {line_number}")
            record = dict(value)
            record["id"] = str(record.get("id", line_number - 1))
            records.append(record)
            if limit is not None and limit > 0 and len(records) >= limit:
                break
    return records


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if str(record.get("prediction") or "").strip():
                completed.add(str(record.get("id", "")))
    return completed


def _browsecomp_prompt(question: str) -> str:
    return (
        question.strip()
        + "\n\nYour response should be in the following format:\n"
        "Explanation: {your explanation for your final answer}\n"
        "Exact Answer: {your succinct, final answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}"
    )


def run_dataset(
    team: Runnable,
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    limit: int | None = 3,
    resume: bool = True,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run a small dataset slice sequentially and append one JSON record per item."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    if workers <= 0:
        raise ValueError("workers must be greater than zero")

    items = load_jsonl(dataset_path, limit=limit)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output) if resume else set()
    mode = "a" if resume and output.exists() else "w"

    succeeded = 0
    failed = 0
    skipped = 0
    started = time.time()

    def process(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        item_started = time.time()
        record: dict[str, Any] = {
            "id": str(item["id"]),
            "question": item["question"],
            "golden_answers": item.get("golden_answers", item.get("answer", "")),
        }
        try:
            result = team.run(_browsecomp_prompt(str(item["question"])))
            if not str(result.answer or "").strip():
                raise RuntimeError("The team returned an empty final answer.")
            record.update(
                {
                    "prediction": result.answer,
                    "confidence": getattr(result, "confidence", None),
                    "elapsed_seconds": time.time() - item_started,
                    "team_result": result.to_dict(),
                }
            )
            return record, True
        except Exception as exc:  # keep later samples runnable after one API failure
            record.update(
                {
                    "prediction": "",
                    "elapsed_seconds": time.time() - item_started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return record, False

    pending = [item for item in items if str(item["id"]) not in completed]
    skipped = len(items) - len(pending)
    if progress:
        for index, item in enumerate(items, start=1):
            item_id = str(item["id"])
            if item_id in completed:
                progress(index, len(items), f"skip id={item_id}")

    with output.open(mode, encoding="utf-8") as file:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, item): item for item in pending}
            completed_now = 0
            for future in as_completed(futures):
                record, ok = future.result()
                completed_now += 1
                succeeded += int(ok)
                failed += int(not ok)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                if progress:
                    status = "ok" if ok else "failed"
                    progress(
                        skipped + completed_now,
                        len(items),
                        f"{status} id={record['id']}",
                    )

    return {
        "requested": len(items),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": time.time() - started,
        "output": str(output.resolve()),
    }
