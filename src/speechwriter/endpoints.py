"""Ask an OpenAI-compatible server which models it currently serves.

This is the only module in the package that opens a socket on purpose, and it exists because
of a fact about local servers: they are *many-model*. ``mlx_lm.server`` answers ``/v1/models``
by scanning the whole HuggingFace cache and hot-swaps per request from the ``model`` field the
client already sends — its ``--model`` flag is optional. So one configured endpoint can serve
every model on the machine, while :func:`~speechwriter.config.model_choices` can only ever name
the single pair the environment happens to hold. This closes that gap on demand.

Three deliberate constraints, each the answer to a way this could go wrong:

* **Nothing here runs at import or at page render.** It is called from a click handler only.
  ``build_agent()`` must not touch the network — that invariant is what keeps the whole test
  suite free and offline — and the Streamlit app is rendered headlessly dozens of times per CI
  run. A probe on the render path would put a socket in both.
* **Stdlib, not the ``openai`` SDK.** ``openai`` reaches this environment only transitively via
  ``langchain-openai``; importing it here would repeat the undeclared-dependency trap this repo
  already documents for ``pyyaml``, and ``urllib`` is markedly faster on the common
  connection-refused path anyway.
* **Every failure is an empty list, never an exception.** The caller is a sidebar button. A
  server that is down, speaking a different protocol, or answering with a shape nobody
  predicted must degrade to "found nothing", not replace the page with a traceback.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# Seconds to wait. Not tuned for the fast paths — a listening server answers in ~2ms and a
# closed port refuses in ~0.15s, neither of which needs a budget — but for the slow *healthy*
# one: `mlx_lm.server` answers `/v1/models` by walking the whole HuggingFace cache with
# `scan_cache_dir()`, which on a machine with tens of gigabytes of models takes seconds. Too
# short a timeout turns that into an empty list, indistinguishable from a server with nothing
# loaded. Long enough to let a real scan finish, still bounded so a server that accepts and
# then stalls cannot hang the page — without a timeout `urlopen` waits forever.
DEFAULT_TIMEOUT = 10.0

# Only these reach the network. `urlopen`'s default opener also installs `FileHandler`,
# `FTPHandler` and `DataHandler`, so `SPEECHWRITER_BASE_URL=file:///Users/you/private` would
# make the Detect button read `/Users/you/private/models` straight off local disk and list
# whatever it found. A configured endpoint is meant to be an HTTP service; anything else is a
# typo at best, and `build_opener` cannot be trusted to drop those handlers (it re-adds every
# default whose class was not passed in), so the scheme is checked explicitly.
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class _CredentialSafeRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never carry the bearer token to a different origin.

    ``HTTPRedirectHandler.redirect_request`` copies every header except content-length and
    content-type onto the new request — the ``Authorization`` header included. So an endpoint
    that answers ``/v1/models`` with a 302 elsewhere receives the reader's real
    ``OPENAI_API_KEY``, and then so does wherever it points. That is not hypothetical for a
    mistyped hosted gateway, an http URL that redirects to https on another host, or a captive
    portal. Same-origin redirects keep the credential, because that is the same server it was
    configured for; anything else is stripped.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        following = super().redirect_request(req, fp, code, msg, headers, newurl)
        if following is not None and _origin(newurl) != _origin(req.full_url):
            following.remove_header("Authorization")
        return following


def _origin(url: str) -> tuple[str, str]:
    """Scheme and authority — what "the same server" means for carrying a credential."""
    parts = urllib.parse.urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


def same_origin(url: str, other: str | None) -> bool:
    """Whether two URLs name the same server — scheme and authority, nothing else.

    The public half of what :class:`_CredentialSafeRedirects` already decides privately, and
    it is the same question for the same reason. That handler protects the *second* hop; this
    protects the *first*, which stopped being safe the moment an endpoint could be typed into
    a text box rather than written into a dotenv by the operator. ``list_models``' docstring
    licenses forwarding the reader's key with "the chat client already sends the very same
    credential to this very same host" — an argument that holds only while the host is the
    configured one, and this is how a caller checks that it still is.

    Total by construction: ``other`` is routinely ``None`` (nothing configured), and
    ``urlsplit`` raises on an unclosed IPv6 bracket. Both answer ``False`` — "not the server
    we hold a credential for" is the safe reading of every input this cannot parse.
    """
    if not other:
        return False
    try:
        return _origin(url) == _origin(other)
    except ValueError:
        return False


def normalize_endpoint(text: str) -> str | None:
    """Read what a person typed as an OpenAI-compatible root, or ``None`` if it is not one.

    Two edits, each measured against what this repo's own servers actually answer:

    * **A missing scheme becomes ``http``.** ``localhost:8080`` is what a reader types, and
      ``urllib.request.Request`` raises at *construction* on a scheme-less URL — which
      :func:`list_models` files under "could not list", reporting a healthy server as dead.
    * **An empty path becomes ``/v1``.** ``http://127.0.0.1:8080`` is equally likely, and
      ``mlx_lm.server``, vLLM, LM Studio and Ollama all serve the OpenAI API under ``/v1``,
      never at the root. Measured: the bare form returns ``[]`` from a server listing eight
      models, again indistinguishable from one that is down.

    The test is ``"://" in raw``, not ``urlsplit(raw).scheme``, because ``urlsplit`` reads the
    two spellings of the same typo differently — ``"localhost:8080"`` parses as scheme
    ``"localhost"`` while ``"127.0.0.1:8080"`` parses as no scheme at all.

    Never raises. It is called on the render path (the sidebar reads it every rerun), so a
    ``ValueError`` here is not a bad caption but a page that throws on *every* rerun with the
    offending text still in session state — the unrecoverable state
    :func:`speechwriter.webui.get_bundle`'s own try/except exists to prevent.

    Normalising is not a second security boundary: a rejected scheme returns ``None`` here and
    is refused again in :func:`list_models`, which is where the test pins it. The value of
    returning ``None`` is that the caller can say "that is not an endpoint" instead of
    "found nothing", which are different facts.
    """
    raw = text.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"http://{raw}"
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.netloc:
        return None
    # `rstrip`, then a default: this is what makes the function idempotent, which the callers
    # rely on — the sidebar normalises the *seeded* environment value as well as the typed one,
    # so anything already in normal form must survive a second pass unchanged.
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{parts.netloc}{path or '/v1'}"


# Built once. The scheme check above is what keeps `file://` and `ftp://` out — `build_opener`
# re-adds every default handler whose class was not passed in, so this cannot do that job.
_OPENER = urllib.request.build_opener(_CredentialSafeRedirects)


def list_models(
    base_url: str, *, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT
) -> list[str]:
    """Model ids served at ``base_url``, sorted; ``[]`` if the server cannot be asked.

    ``base_url`` is the same value :class:`~speechwriter.config.Settings` carries — an
    OpenAI-compatible root such as ``http://127.0.0.1:8080/v1`` — and ``/models`` is appended
    to it, matching the endpoint every such server exposes.

    ``api_key`` is forwarded as a bearer token when given. That is no new disclosure: the chat
    client built by :func:`~speechwriter.agent._build_model` already sends the very same
    credential to this very same host on every turn. Local servers ignore it; a hosted
    OpenAI-compatible service needs it to answer at all.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        # Reading the scheme is inside the `try` for the same reason the Request build is, and
        # it was outside until an endpoint could be *typed*: `urlsplit` raises
        # `ValueError: Invalid IPv6 URL` on an unclosed bracket ("http://[::1"), so the one
        # line that decides whether to open a socket at all could itself escape as an
        # exception — the one thing this function promises never to do. Unreachable while the
        # endpoint came only from a dotenv the operator wrote; one keystroke away now.
        scheme = urllib.parse.urlsplit(url).scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            logger.info("Refusing to list models at %s: %r is not an HTTP scheme.", url, scheme)
            return []
        # Building the Request is inside the `try` deliberately, not merely tidily: a
        # `base_url` with no scheme raises `ValueError: unknown url type` from the constructor,
        # before any socket is opened. Left outside, a typo'd endpoint would escape this
        # function as an exception, which is the one thing it promises never to do.
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with _OPENER.open(request, timeout=timeout) as response:
            payload = json.load(response)
        return sorted(_ids(payload))
    except Exception as exc:
        # Bare `Exception`, not `URLError`. The reachable failures are not all network errors:
        # a `base_url` pointing at a server that speaks a *different* protocol answers 200 with
        # JSON of another shape, and one pointed at Ollama's native API (`/api/...` rather than
        # `/v1/...`) returns a body that raises inside parsing. Narrowing this to the errors
        # that came to mind is how a sidebar button becomes a page-level traceback.
        logger.info("Could not list models at %s: %s: %s", url, type(exc).__name__, exc)
        return []


def _ids(payload: object) -> list[str]:
    """Pull model ids out of an OpenAI ``/models`` body, tolerating what servers really send.

    The documented shape is ``{"data": [{"id": ...}, ...]}``, but a running Ollama with nothing
    pulled answers ``{"object": "list", "data": null}`` — so ``data`` is ``None``, not ``[]``,
    and iterating it raises. Everything here is defensive for that reason: entries are skipped
    rather than trusted, so a server that returns one malformed row still yields the rest.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id")
        if isinstance(identifier, str) and identifier:
            ids.append(identifier)
    return ids
