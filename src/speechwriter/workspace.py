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

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import wave
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


# Deepgram's Aura-2 text-to-speech, reached over its REST endpoint. Named here rather than in
# `config.py` because, unlike SPEECHES_SUBDIR or WORDS_PER_MINUTE, nothing else in the project
# consumes them: there is no second subsystem to drift from. Thalia is a clear, even-paced
# American English voice — a neutral reader, which is what a timing measurement wants.
TTS_MODEL = "aura-2-thalia-en"
TTS_ENDPOINT = "https://api.deepgram.com/v1/speak"
# Raw 16-bit little-endian mono PCM at this rate (`encoding=linear16&container=none`), so the
# duration is arithmetic on the byte count and the WAV header is written here, by `wave`, over
# the whole joined speech — rather than trusting a header Deepgram wrote for one chunk.
TTS_SAMPLE_RATE = 24_000
_BYTES_PER_SAMPLE = 2
# Deepgram's REST endpoint takes at most 2,000 characters of text per request, and a
# three-minute speech is roughly twice that — so a draft is sent in sentence-bounded pieces and
# the audio joined. Kept a little under the limit so a count that disagrees with Deepgram's by a
# character or two (it strips control characters first) never trips it.
TTS_CHUNK_CHARS = 1_900
# Per request. Synthesis is faster than real time, so a chunk's audio arrives in seconds; this
# only bounds a server that accepts the connection and then stalls.
TTS_TIMEOUT = 60.0

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class AudioUnavailable(RuntimeError):
    """Raised when spoken-length measurement is requested with no ``DEEPGRAM_API_KEY``.

    A distinct type rather than a generic failure: the caller is a UI that must tell the reader
    *how to fix it* — a missing key is a setup step, where a rejected request is an error.
    """


class SynthesisFailed(RuntimeError):
    """Raised when Deepgram answered a synthesis request with an error, or not at all."""


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


def tts_chunks(text: str, limit: int = TTS_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters, at sentence ends if possible.

    Sentence boundaries first, because a cut mid-sentence changes the prosody Deepgram gives it
    — a sentence read as two sounds like two, and the timing drifts with it. A sentence longer
    than the limit (a speech can build to one) falls back to word boundaries, and a single
    "word" longer than the limit — a pasted URL — is cut where it must be.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        line = ""
        for word in sentence.split():
            while len(word) > limit:
                if line:
                    pieces.append(line)
                    line = ""
                pieces.append(word[:limit])
                word = word[limit:]
            candidate = f"{line} {word}" if line else word
            if len(candidate) > limit:
                pieces.append(line)
                line = word
            else:
                line = candidate
        if line:
            pieces.append(line)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) > limit:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so the API key never leaves the host it was meant for.

    ``urllib`` copies every header except the body's length and type onto a redirected request,
    ``Authorization`` included — so a 3xx from anywhere on the path would hand the key to
    wherever it pointed. The endpoint is a fixed HTTPS URL with no reason to redirect, which
    makes refusing outright both the simplest guard and the correct one.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_OPENER = urllib.request.build_opener(_NoRedirects)


def _synthesise(chunk: str, *, api_key: str, model: str, endpoint: str, timeout: float) -> bytes:
    """One Deepgram request: ``chunk`` in, raw linear16 PCM out."""
    query = urllib.parse.urlencode(
        {
            "model": model,
            "encoding": "linear16",
            "container": "none",
            "sample_rate": TTS_SAMPLE_RATE,
        }
    )
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        data=json.dumps({"text": chunk}).encode("utf-8"),
        # `Token`, not `Bearer`: Deepgram reserves Bearer for short-lived JWTs, and an API key
        # sent that way is a 401 that reads like a bad key.
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        # Errors come back as JSON (`err_code`, `err_msg`); success is binary audio — so the
        # status decides which one this is, never an attempt to parse the body.
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise SynthesisFailed(f"Deepgram answered {exc.code}: {detail}") from exc
    except OSError as exc:  # URLError, timeouts, refused connections
        raise SynthesisFailed(f"Could not reach Deepgram: {exc}") from exc


def measure_spoken_length(
    text: str,
    *,
    api_key: str | None,
    model: str = TTS_MODEL,
    endpoint: str = TTS_ENDPOINT,
    timeout: float = TTS_TIMEOUT,
) -> SpokenLength:
    """Synthesise a draft with Deepgram and report how long it actually takes to say.

    ``WORDS_PER_MINUTE`` is a single constant standing in for pace, and it cannot know that
    one draft is dense with long words while another is short and punchy. This measures the
    real thing — at the cost of a billed API call per ~2,000 characters, which is why the caller
    decides when to pay it rather than it happening on every page render.

    Measured over the same corpus :attr:`Document.words` counts (header block and bracketed
    delivery cues removed), so the two figures are comparable. The consequence worth knowing:
    a ``[pause]`` contributes *no* silence here, so this is the time to say the words, not the
    time the performance runs. Each chunk also carries the short lead-in and tail silence of a
    separate utterance, so a long draft reads a little longer than one continuous read would.

    Raises :class:`AudioUnavailable` with no key, and :class:`SynthesisFailed` if Deepgram
    rejects a request or cannot be reached. A draft with nothing to say returns zero without a
    request, key or no key.
    """
    _, body = _split_front_matter(text)
    spoken = _spoken_text(body).strip()
    if not spoken:
        return SpokenLength(seconds=0.0, wav=b"", sample_rate=0)
    if not api_key:
        raise AudioUnavailable(
            "Measuring spoken length needs a Deepgram API key. Set DEEPGRAM_API_KEY in the "
            "project's dotenv file and restart."
        )

    pcm = b"".join(
        _synthesise(chunk, api_key=api_key, model=model, endpoint=endpoint, timeout=timeout)
        for chunk in tts_chunks(spoken)
    )
    # A torn final sample would shift every frame after it in the joined stream; drop it.
    pcm = pcm[: len(pcm) - len(pcm) % _BYTES_PER_SAMPLE]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(_BYTES_PER_SAMPLE)
        wav.setframerate(TTS_SAMPLE_RATE)
        wav.writeframes(pcm)

    return SpokenLength(
        seconds=len(pcm) / (_BYTES_PER_SAMPLE * TTS_SAMPLE_RATE),
        wav=buffer.getvalue(),
        sample_rate=TTS_SAMPLE_RATE,
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
