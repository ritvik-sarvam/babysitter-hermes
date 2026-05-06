from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import RunSnapshot


@dataclass
class Passage:
    chunk_id: str
    parent_doc_id: str
    parent_title: str
    page_range: list[int] | None
    heading_path: str
    text: str

    def citation(self) -> str:
        if self.page_range:
            return f"{self.parent_doc_id} p.{self.page_range[0]}-{self.page_range[1]}"
        return self.parent_doc_id


DEFAULT_TOP_K_CHUNKS = 5


async def retrieve_kb_for_snapshot(
    snapshot: RunSnapshot,
    *,
    top_k: int = DEFAULT_TOP_K_CHUNKS,
    kb_roots: list[Path] | None = None,
    tracking_dir: Path | None = None,
) -> list[Passage]:
    passages = []
    for root in kb_roots or []:
        if not root.exists():
            continue
        # TODO: Replace this shallow first-N markdown scan with symptom-tagged
        # retrieval derived from the snapshot metrics/config so KB evidence is
        # relevant to the observed failure mode.
        markdown_files = sorted(root.rglob("*.md"))[:top_k]
        for path in markdown_files:
            text = path.read_text(errors="replace")[:6000]
            passages.append(
                Passage(
                    chunk_id=path.stem,
                    parent_doc_id=str(path),
                    parent_title=path.name,
                    page_range=None,
                    heading_path=path.name,
                    text=text,
                )
            )
    if tracking_dir:
        tracking_dir.mkdir(parents=True, exist_ok=True)
        (tracking_dir / "retrieved_evidence.txt").write_text(
            "\n\n".join(passage.text for passage in passages)
        )
    return passages
