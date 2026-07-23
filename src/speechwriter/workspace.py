"""Reading back what the agent produced: drafts, research notes, and voice profiles.

Deliberately UI-free. The agent *writes* through its virtual filesystem; something has to
read the results on real disk, and where those files live is a property of the project's
conventions (:mod:`speechwriter.config`) rather than of whichever front end is asking.

Two things this module refuses to re-derive, because a second copy would drift:

* the output sub-directories and the speaking pace, which come from
  :mod:`speechwriter.config` and are interpolated into the prompts that *instruct* the
  agent to use them; and
* the exhaustive Store walk, which comes from :func:`speechwriter.memory.all_items` and is
  the single place the "never truncate at the default page limit" invariant lives.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.store.base import BaseStore

from speechwriter.config import (
    RESEARCH_SUBDIR,
    SPEECHES_SUBDIR,
    WORDS_PER_MINUTE,
    Settings,
)
from speechwriter.memory import all_items

# The prompt asks for "a short header block" without dictating a format, and the agent
# reliably reaches for `---` fences. That block must be lifted off before the rest is
# rendered or counted: in CommonMark a `---` line directly after a paragraph makes that
# paragraph a setext H2, so an unparsed header renders as one run-on heading — and its
# words inflate the spoken-length estimate the header itself is quoting.
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(?P<block>.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)

# Delivery cues the speaker acts on but never says: `[pause]`, `[beat]`, `[slow down here]`.
# The delivery-and-cadence skill asks for a marked-up script, so these are expected, not
# stray — and counting them would print a word count that contradicts the one the draft
# states in its own header, on the same screen.
#
# The trailing `(?!\()` spares a Markdown link: in `[our report](url)` the `]` is followed
# by `(`, so the label is *not* stripped and its words still count (they are spoken). A cue
# like `[pause]` is not followed by `(`, so it is removed as intended.
_STAGE_DIRECTION = re.compile(r"\[[^\[\]]*\](?!\()")


@dataclass(frozen=True)
class Document:
    """One Markdown file the agent wrote, split into its header block and its prose."""

    path: Path
    text: str
    """The file exactly as written — what a download should hand back."""

    body: str
    """The prose, with any front-matter header removed."""

    front_matter: tuple[tuple[str, str], ...]
    """Header fields as ordered ``(label, value)`` pairs; ``label`` is empty for a bare line."""

    modified: datetime

    @property
    def slug(self) -> str:
        return self.path.stem

    @property
    def words(self) -> int:
        """Roughly how many words get *said*.

        Counts ``body`` rather than ``text``, and drops bracketed stage directions: the
        header block states its own word count, and a metric that disagreed with the header
        rendered directly above it would just look broken. Still an estimate — Markdown
        punctuation counts as a word — but one that lands within a few words of the draft's.
        """
        return len(_STAGE_DIRECTION.sub(" ", self.body).split())

    @property
    def minutes(self) -> float:
        """Approximate time to deliver aloud, at the pace the prompt tells the agent to use."""
        return self.words / WORDS_PER_MINUTE


@dataclass(frozen=True)
class MemoryEntry:
    """One persisted item from the Store — normally a speaker's voice profile."""

    key: str
    namespace: tuple[str, ...]
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Every Markdown file in ``directory``, newest first.

    A missing directory yields an empty list rather than raising: the workspace folders are
    created lazily by the agent's first write, so "no speeches yet" is the normal state of a
    fresh checkout, not an error worth surfacing.
    """
    if not directory.is_dir():
        return []

    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            modified = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            # The agent may be mid-write, or the file may have just been removed. One
            # unreadable draft should not blank out the whole listing.
            continue
        front_matter, body = _split_front_matter(text)
        documents.append(
            Document(
                path=path,
                text=text,
                body=body,
                front_matter=front_matter,
                modified=modified,
            )
        )

    return sorted(documents, key=lambda doc: doc.modified, reverse=True)


def speeches_dir(settings: Settings) -> Path:
    """Real directory the agent saves speech drafts under."""
    return settings.workspace_dir / SPEECHES_SUBDIR


def research_dir(settings: Settings) -> Path:
    """Real directory the agent saves research briefs under."""
    return settings.workspace_dir / RESEARCH_SUBDIR


def speeches(settings: Settings) -> list[Document]:
    """Saved speech drafts, newest first."""
    return load_documents(speeches_dir(settings))


def research_notes(settings: Settings) -> list[Document]:
    """Saved research briefs, newest first."""
    return load_documents(research_dir(settings))


def memories(store: BaseStore) -> list[MemoryEntry]:
    """Every persisted memory item, sorted by key.

    Read from the live Store rather than the JSON snapshot on disk, so a profile the agent
    learned this session shows up before anything has called ``persist()``.
    """
    return sorted(
        (
            MemoryEntry(
                key=item.key,
                namespace=tuple(item.namespace),
                text=_as_markdown(item.value),
            )
            for item in all_items(store)
        ),
        key=lambda entry: entry.key,
    )


def _split_front_matter(text: str) -> tuple[tuple[tuple[str, str], ...], str]:
    """Separate a leading ``---`` header block from the prose that follows.

    Parsed by hand rather than with PyYAML: that package reaches this project only as a
    transitive dependency of LangChain, so importing it here would make the package depend
    on something it never declared. The header is also not reliably valid YAML — values like
    ``~2:03 at 130 wpm`` carry stray colons — and a strict parser raising on a draft would
    be a far worse outcome than a lenient split on the first colon.
    """
    match = _FRONT_MATTER.match(text)
    if match is None:
        return (), text

    fields: list[tuple[str, str]] = []
    for line in match.group("block").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        label, separator, value = stripped.partition(":")
        if separator and label.strip():
            fields.append((label.strip(), value.strip()))
        else:
            fields.append(("", stripped))

    return tuple(fields), text[match.end() :].lstrip("\n")


def _as_markdown(value: object) -> str:
    """Best-effort render of a Store value as displayable Markdown.

    The Store holds whatever deepagents' ``StoreBackend`` chose to write, and its
    ``file_format`` is deliberately left at the default — so the payload shape is not ours
    to assume, and it can change under a dependency bump. A recognised ``{"content": ...}``
    document renders as the Markdown it is; anything else falls back to fenced JSON, which
    is ugly but honest, rather than an empty panel that reads as "no memory saved".
    """
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Filter rather than `all(isinstance(...))`: the comprehension gives the type
            # checker a genuine list[str], and an equal length still means "every line was
            # a string", so a mixed payload falls through to the JSON branch as intended.
            lines = [line for line in content if isinstance(line, str)]
            if len(lines) == len(content):
                return "\n".join(lines)
    if isinstance(value, str):
        return value

    rendered = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return f"```json\n{rendered}\n```"
