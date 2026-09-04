"""The eval datasets under ``evals/`` are gated here, so drift blocks like anything else.

``evals/validate_datasets.py`` recomputes every derived number in the 55 examples from
``config.WORDS_PER_MINUTE``, resolves their save paths through ``load_settings()``, and
checks every tool name, subagent name and skill slug against the live contract. Without
this test it was an advisory script nobody ran -- weaker than the hook CLAUDE.md deleted
in favour of tests, since it caught nothing on a push, a PR, or a hand edit.
"""

from __future__ import annotations

import os
import subprocess
import sys

from speechwriter import config

REPO_ROOT = config._PKG_DIR.parents[1]


def test_eval_datasets_match_the_live_contract():
    # A subprocess, and not an in-process import, for the same reason
    # test_import_speechwriter_is_lazy uses one. The validator calls load_settings(), whose
    # dotenv load would read the project's real secrets file once SPEECHWRITER_HOME points at
    # the repo -- and that loader sets any variable not already present, so a real Tavily key
    # would land in os.environ for every test that ran afterwards. That is an order-dependent
    # flake, in a suite whose whole premise is that it runs offline with no keys. The child's
    # own environment is inherited and harmless: it validates JSON and exits, and nothing it
    # sets propagates back here.
    #
    # SPEECHWRITER_HOME is pinned to the real repo rather than a tmp_path -- the one place in
    # the suite where that is right, because the committed datasets are exactly what is under
    # test. It is still set explicitly, never inherited: a developer with the variable exported
    # would otherwise validate someone else's tree.
    env = os.environ | {"SPEECHWRITER_HOME": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "evals" / "validate_datasets.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "evals/validate_datasets.py reported failures:\n" + result.stdout + result.stderr
    )
