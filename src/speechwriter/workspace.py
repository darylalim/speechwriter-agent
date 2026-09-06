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
from typing import Any

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
        return _count_spoken(self.body)

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


def _spoken_text(body: str) -> str:
    """An already-header-stripped body reduced to what is actually said aloud.

    The single corpus definition. :func:`_count_spoken` counts it and
    :func:`measure_spoken_length` synthesises it, so the estimated and measured figures the
    browser prints side by side describe the *same* words — two numbers derived from two
    different strings would differ for a reason the reader could not see.
    """
    return _STAGE_DIRECTION.sub(" ", body)


def _count_spoken(body: str) -> int:
    """Words said aloud in an already-header-stripped body."""
    return len(_spoken_text(body).split())


def spoken_words(text: str) -> int:
    """Words said aloud in a draft, given the file's full text.

    The eval scorers need this figure for a speech that only ever existed in a message, with
    no file behind it, and a second implementation there would drift from the one the browser
    shows -- the same reason this module refuses to re-derive ``WORDS_PER_MINUTE``. Strips the
    ``---`` header block first, then the bracketed delivery cues, exactly as
    :attr:`Document.words` does.
    """
    _, body = _split_front_matter(text)
    return _count_spoken(body)


# Kokoro is small (82M), Apple-Silicon-native via MLX, and runs at roughly RTF 0.06 — a
# three-minute speech is synthesised in about nine seconds. Named here rather than in
# `config.py` because, unlike SPEECHES_SUBDIR or WORDS_PER_MINUTE, nothing else in the
# project consumes them: there is no second subsystem to drift from.
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"

# Loading the weights costs ~0.5s and building the phonemiser pipeline rather more, so the
# model is kept per process. A plain dict rather than `functools.lru_cache` because the
# value is an unhashable, lazily-imported object and this keeps the module's top-level
# imports exactly as light as they were.
# `Any`, explicitly rather than by omission. A Protocol declaring `generate` would be the
# precise alternative and it cannot work here: with the extra installed `load_model` returns
# `nn.Module`, which does not satisfy such a Protocol, while in CI the symbol is unresolvable
# and satisfies anything — so the annotation would fail in exactly one of the two
# environments. Same bind as the suppression comment below, one level up.
_TTS_MODELS: dict[str, Any] = {}


class AudioUnavailable(RuntimeError):
    """Raised when spoken-length measurement is requested without the ``audio`` extra.

    A distinct type rather than letting ``ImportError`` escape: the caller is a UI that must
    tell the reader *how to fix it*, and catching bare ``ImportError`` around a call this
    deep would also swallow a genuine broken install inside the TTS stack.
    """


@dataclass(frozen=True)
class SpokenLength:
    """A draft's *measured* delivery time, plus the audio it was measured from."""

    seconds: float
    wav: bytes
    """Complete RIFF/WAVE bytes — playable as-is, so the synthesis is not thrown away."""

    sample_rate: int

    @property
    def minutes(self) -> float:
        return self.seconds / 60


def _load_tts(model_id: str) -> Any:
    """Load (once per process) the MLX TTS model, or explain why it cannot be loaded."""
    cached = _TTS_MODELS.get(model_id)
    if cached is not None:
        return cached
    # Resolved by name at call time rather than imported at module scope: `mlx_audio` is an
    # optional extra pulling a torch/spacy stack, and a top-level import would charge every
    # `import speechwriter` for it — the same lazy-import discipline `__init__.py` applies to
    # the langchain stack.
    try:
        from importlib import import_module

        generate = import_module("mlx_audio.tts.generate")
        hub = import_module("mlx_audio.utils")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise AudioUnavailable(
            "Measuring spoken length needs the optional audio extra. "
            "Install it with: uv sync --extra audio"
        ) from exc

    # Resolved in two steps on purpose. `load_model` is annotated `model_path: Path` and means
    # it — handed a Hub id as a Path it looks for a literal directory, misses the download
    # branch, and raises FileNotFoundError. `get_model_path` is the half that takes a repo-id
    # *string*, downloads if needed, and returns the real snapshot directory.
    #
    # To reproduce the type error the collapsed form causes, you must check **with the extra
    # installed**: ty resolves `import_module` with a literal argument and types `load_model`
    # precisely. Run it in the CI state instead and mlx_audio is unresolvable, everything is
    # Any, and the error does not appear — which reads as though the rule were stale.
    #
    # Collapsing these into one `load_model(model_id)` call is the obvious-looking tidy-up and
    # it is a trap: it works at runtime but only type-checks with a suppression, and *that*
    # has no correct form. With the extra installed the suppression is required; in CI, which
    # installs no extras, the same comment is an unused-suppression warning and `ty` exits 1.
    # Two correctly-typed calls need no suppression, so both environments stay green.
    model = generate.load_model(hub.get_model_path(model_id))
    _TTS_MODELS[model_id] = model
    return model


def measure_spoken_length(
    text: str, *, model_id: str = TTS_MODEL, voice: str = TTS_VOICE
) -> SpokenLength:
    """Synthesise a draft and report how long it actually takes to say.

    ``WORDS_PER_MINUTE`` is a single constant standing in for pace, and it cannot know that
    one draft is dense with long words while another is short and punchy. This measures the
    real thing — at the cost of running a TTS model, which is why the caller decides when to
    pay it rather than it happening on every page render.

    Measured over the same corpus :attr:`Document.words` counts (header block and bracketed
    delivery cues removed), so the two figures are comparable. The consequence worth knowing:
    a ``[pause]`` contributes *no* silence here, so this is the time to say the words, not
    the time the performance runs.

    Raises :class:`AudioUnavailable` if the ``audio`` extra is not installed.
    """
    _, body = _split_front_matter(text)
    spoken = _spoken_text(body).strip()
    if not spoken:
        return SpokenLength(seconds=0.0, wav=b"", sample_rate=0)

    segments = list(_load_tts(model_id).generate(text=spoken, voice=voice))

    # Imported *after* the guard above, not at the top of the function. numpy reaches this
    # project only transitively — through streamlit — so it is neither a declared base
    # dependency nor part of the `audio` extra. Imported earlier, an install without it would
    # raise a bare ModuleNotFoundError past the one error type this API promises, and past the
    # empty-draft return that needs no audio stack at all. Here, `_load_tts` has already
    # succeeded, so mlx-audio is installed and numpy came with it.
    import io
    import wave

    import numpy as np

    if not segments:  # pragma: no cover - defensive; the model yields at least one segment
        return SpokenLength(seconds=0.0, wav=b"", sample_rate=0)

    sample_rate = int(segments[0].sample_rate)
    pcm = np.concatenate([np.asarray(segment.audio, dtype=np.float32) for segment in segments])

    # Kokoro emits float samples nominally in [-1, 1]; clip before the int16 cast so an
    # overshoot wraps to the opposite rail as a loud click instead of silently inverting.
    ints = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(ints.tobytes())

    return SpokenLength(
        seconds=len(pcm) / sample_rate,
        wav=buffer.getvalue(),
        sample_rate=sample_rate,
    )


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
