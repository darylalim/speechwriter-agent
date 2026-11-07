# Upgrade plan

Written 2026-11-06 for a new implementation session. Workstream 1 is mostly done, and the Progress
table lists what is still open. Nothing after it has started.

Four workstreams and one release. Read `CLAUDE.md` first. It holds the invariants this plan relies
on, and it overrides this plan if the two disagree.

## Order of work

1. **Workstream 1 (dependency upgrade) goes first.** Workstream 2 measures the agent, and those
   results should describe the stack that will actually ship.
2. **Workstream 2 (model and effort tuning)** follows, on the upgraded stack.
3. **Workstream 3 (web UI theme)** is independent of both, and can happen at any point. Doing it
   after Workstream 1 means it is checked on the Streamlit version that will ship.
4. **Workstream 4 (web UI layout)** follows Workstream 3, because its widths were measured with
   the Chamber theme's font.
5. **Release 0.3.0** comes last, once, for everything together.

The README images are replaced in Workstream 4, after Workstream 2 has settled the default model,
because they show the model picker. If the workstreams run in a different order, re-check them.

## Progress

Update this table at the end of each session, so the next one starts from the facts. Add experiment
names and results under the relevant phase.

| Part | Status | Expected spend | Notes |
|---|---|---|---|
| Workstream 1: Dependency upgrade | In progress | Under $1 | Open items are listed below |
| Workstream 2, Phase 0: Prove the evals | In progress | Priced in Phase 0, Step 2 | Step 1 code done 2026-11-07; the cost check waits on the LangSmith limit |
| Workstream 2, Phase 1: Effort option | Done 2026-11-07 | Free | All 7 planned tests; 11 mutants, all killed |
| Workstream 2, Phase 2: Main sweep | Not started | About $45 to $50 | |
| Workstream 2, Phase 3: Per-role effort | Not started | Priced from the Phase 0 figures | |
| Workstream 2, Phase 4: Haiku researcher | Optional | Priced if it goes ahead | |
| Workstream 2, Phase 5: Adopt the results | Not started | A few dollars | |
| Workstream 3: Theme | Done 2026-11-07, except step 6 | Free | README images wait for Workstream 2, with Workstream 4's |
| Workstream 4: Layout | Done 2026-11-07, except the README images | Free | Images wait for Workstream 2's default model |
| Release 0.3.0 | Not started | Free | Needs the user's go-ahead |

### Workstream 1 status (2026-11-06)

Done: the lock upgrade, `ty` 0.0.99, the CI fixes and the `.gitignore` fix. A live draft, critique
and revise turn passed on API version `2026-10-21` with no 400. Both web UI pages load on Streamlit
1.67, and **Measure** worked (397 words: 2.6 min measured against a ~3.1 min estimate). The dataset
mirror is in sync.

Still open:

1. **Step 6, the `--langsmith --limit 1` experiment.** Not run. LangSmith rejects trace uploads with
   a 429, `Monthly unique traces usage limit exceeded`. The sync check still passes, because reading
   datasets is not metered the same way.
2. **The double-tracing check for `opentelemetry-api`.** It needs traces to land, so it waits on the
   same limit. Any `--langsmith` run in Workstream 2 is blocked by this too. The limit resets on
   2026-12-01.
3. **The startup check on the `.claude/settings.json` permission rules.** The user made this edit,
   and the wildcard form is valid. The docs warn that a `*` before the subcommand may trigger a
   startup warning, so check for one, and confirm that the pinned commands run without a prompt.

### Workstream 2, Phase 0 status (2026-11-07)

Step 1 done, apart from the cost check:

- `--trials N` works on both live paths. On `--langsmith` it passes `num_repetitions`, and the
  target counts its own repeats, because LangSmith does not say which repeat it is running. Each
  trial gets its own home and its own thread (`<id>-<trial>`; a single trial keeps the bare id).
  `--trials` is refused alongside `--dry-run` or `--regrade` (exit 2).
- `--keep` now keeps runs under `.speechwriter/eval-runs/<experiment>/<example>-<trial>/` and
  writes a `record.json` there. Every tool call in it carries the index of the model message that
  made it, which is what Step 2, item 6 needs. On `--langsmith`, the experiment's LangSmith name is
  written to `experiment.txt`. The `SPEECHWRITER_EVAL_KEEP` variable is gone, and the folder is
  passed down explicitly instead.
- `--regrade <dir>` scores saved records against the dataset as it is now, with no agent call.
  Exit codes: `0` all graded, `1` some could not be, `2` none found.
- Eleven mutants, and the tests killed all of them.

Still open: **the agent and judge cost per run**. It needs one `--langsmith` run, which the trace
limit blocks until 2026-12-01.

**Live check of the new harness (2026-11-07, about $1).** `--limit 1 --keep` on the plain path
ran `ceremonial-wedding-toast-best-man` in about 63 seconds and scored 9/9. `--regrade --no-judge`
on the kept folder gave the same 9/9 with no agent call. The 429s from the trace upload printed
throughout, but they did not affect the run. Until the limit resets, set `LANGSMITH_TRACING=false`
for plain-path runs to keep the output readable.

**Finding for Step 2, item 6: the critic was told to read the file.** In that run the draft was
written in message 5 and the critic's `task` was sent in message 7, so the file existed and the
review was sound. But the `task` description held a 684-character brief and "saved at
/workspace/speeches/joel-nadia-best-man.md (read it)", not the 1,761-character draft. The
orchestrator prompt says to give the critic the full draft. This is a real agent defect, reported
and not fixed, because Phase 0 changes no prompts. One run is not a rate: the Step 2 baseline
should count how often it happens, and a scored check must exist before Phase 2.

**The scored check now exists: `critic_received_draft`.** It is deterministic and free, and it
runs on every dataset whenever the critic is called. It rebuilds the draft as it stood at each
critic `task` from the `write_file` and `edit_file` calls before it, not from the file on disk,
which holds the final revision. It then measures how much of the draft's spoken text appears, in
order, in the `task` description. The bar is 90%, and a test pins it from both sides: half the
draft fails, and a lightly tidied draft passes. There is no row when the critic is never called,
and the row is unscored when no draft came before the call. Regraded with `--regrade`, the kept
2026-11-07 run now scores **9/10**, with the critic given 12% of the draft. Phase 2 can count this
row directly.

### Workstream 3 status (2026-11-07)

Steps 1 to 5 and 7 are done; step 6 (the README images) waits for Workstream 4, as planned.

- The theme is in `.streamlit/config.toml` with the `[server]` block unchanged. The browse page's
  header block is `st.markdown` now, and a new AppTest checks it renders as body text and never as
  a caption. No existing test read it as a caption, so none needed updating.
- Four contrast tests: links (now on shared `_surfaces` and `_theme` helpers), primary and body
  text, accent hue agreement (primary 2.1 degrees apart, links 4.3, bar 6), and no
  `type="primary"` button. Five mutants, all killed. The old `#61afef` dark primary fails only on
  white-on-primary (2.36:1): it still clears 4.5:1 as text on the new dark surfaces.
- Rendered with Playwright and Chrome in both modes on a scratch copy of the workspace: a long
  draft, the open **Storage** expander, and the chat input with text typed. The send button fills
  with primary in both modes. Type with `keyboard.type`, not `fill`, or React never enables it and
  the button renders faded.
- **Not checked: a finished turn.** That needs a live model call. Check it during the first paid
  run of Workstream 2, or during Workstream 4's render check.

### Workstream 4 status (2026-11-07)

Steps 1 to 6 and 8 are done, on branch `workstream-4-layout`. The CLAUDE.md half of step 7 is done
too. The README images wait for Workstream 2, because they show the model picker, and the default
model may change.

- **Tests read the built element tree, not the source.** `AppTest` exposes each container's pixel
  width, gap size and wrap flag. Five tests: the frame equals rail + gap + reading width, the row
  wraps at the large gap, the reading column is 560px, unwrapped and holds only the speech, Research
  and Memory share the column, a replayed chat reply takes the same width, and only Workspace is
  wide. The page layout is still read from source, because `set_page_config` is not in the tree.
- **Eleven mutants, all killed.** "Change `RAIL_WIDTH` alone" only fails if `PAGE_WIDTH` is a
  literal, since a computed one follows it, so the mutant tested was `RAIL_WIDTH = 300` with
  `PAGE_WIDTH = 896`. Not covered: the width of a reply while it streams, which needs a live call.
  The replay path, which is the same container, is covered.
- **Render check, Streamlit 1.67, Chrome via Playwright.** The fixtures were a 792-word draft
  with a three-field header, a 281-word draft with a 96-character slug, a research note with a
  six-column table and a 130-character URL, and one memory profile. A stubbed copy of the page
  returned a canned measurement and WAV for the three-metric rail.

| Window | Sidebar | Arrangement | Rail | Reading | Speech starts |
|---|---|---|---|---|---|
| 320x568 | closed | stacked | 288 | 288 | 650 |
| 360x740 | closed | stacked | 328 | 328 | 624 |
| 780x360 | open | stacked | 448 | 448 | 605 |
| 768x1024 | closed | stacked | 736 | 560 | 549 |
| 820x1180, 834x1194 | open | stacked | 488, 502 | 488, 502 | 549 |
| 1024x768, 1180x820 | open | stacked | 564, 720 | 560 | 549 |
| 1240x800 (added) | open | side by side | **156** | 560 | 255 |
| 1280x800 | open | side by side | 196 | 560 | 255 |
| 1366 to 2560 | open | side by side | 272 | 560 | 255 |
| 1280x800 at 200% zoom | closed | stacked | 608 | 560 | 517 |

- No sideways page scroll at any size, in any view (`scrollWidth` matched `clientWidth` on both the
  `section` and the document). The research table scrolls inside the column, and the long URL wraps.
- No element in the rail clips or spills out, including the three metrics plus audio at 1240.
- The longest line is about 73 characters at 560px, against the prototype's 74.
- **The sidebar check:** at 1280 the page is already side by side with the sidebar open, as the
  design table says, so the plan's "1280 moves from stacked" could not happen. At 1024 and 1180,
  closing the sidebar moved the page from stacked to side by side and lifted the speech from 549
  to 255.
- Moving Write, Workspace, Write, Workspace at 1440 switched the main block between 736 and 1140
  each time.
- **The narrow band is real:** a 156px rail at 1240 with the sidebar open, which is below the
  plan's 190px guide. The picker shows "margaret-retire" and the audio player drops its time
  readout, but it works. This is the known limit, now measured at one point.
- **The Write and Workspace links at 360px** move into the sidebar, behind its `»` control.
- Both colour modes were checked at 360, 768, 1280 and 1920.
- Recipe notes: `full_page` screenshots stop at the viewport, because Streamlit scrolls inside
  its `section`. The sidebar collapse button is `visibility: hidden` until hover, so click it with
  JavaScript. Scope the document picker to `stMainBlockContainer`, or the sidebar's model picker
  matches first.

## Commits

Commit or push only when the user asks. Per project memory, commits go straight to `main` with no
feature branch. After each push, check the result with `gh run list --limit 3`, because a red CI run
blocks nothing and is easy to miss.

## Workstream 1: Dependency upgrade

### Goal

Move every locked dependency to its latest version with `uv`, and confirm the agent still works
against the live API.

### What was tried

On 2026-11-06 the upgrade was trialled offline in a throwaway `git archive` copy. The real
`uv.lock` and `.venv` were not changed. Versions will have moved by the time this runs, so re-check.

| Check | Result on the upgraded lock |
|---|---|
| `uv run pytest` on Python 3.11, 3.13 and 3.15 | 152 passed, 1 skipped on each |
| `uvx ruff@0.16.0 check .` and `uvx ty@0.0.63 check` | Clean |
| `uvx ruff@0.17.4 check .`, `format --check`, and `uvx ty@0.0.99 check` | Clean |
| `printf 'exit\n' \| uv run speechwriter` | Exit 0 |
| `evals/run_experiment.py --dry-run` | 9/9 |
| Live API turn | **Not run.** This is the main open risk |

Python 3.12 was not run in the trial. CI covers it.

Notable changes:

- **`anthropic` 1.11.0 -> 2.6.0.** It arrives through `langchain-anthropic` 1.7.10, and nothing in
  this repo imports it directly. Version 2.x sends `anthropic-version: 2026-10-21`, so **every
  request moves to a new API version**. This is the change the offline suite cannot exercise.
- **Direct dependencies:** `deepagents` 0.7.34, `langchain-anthropic` 1.7.10, `langsmith` 0.14.17,
  `streamlit` 1.67.0, `python-dotenv` 1.2.4, `pytest` 9.2.0, `watchdog` 7.1.0.
- **Transitive major versions:** `google-genai` 3, `pillow` 13, `python-dateutil` 4, `multidict` 7,
  `websockets` 17, `pyarrow` 26, `wcmatch` 12.
- **Added:** `opentelemetry-api`, required by `anthropic` 2.x. It is the API package only, so it
  should not add a second tracer, but check that a live turn is not traced twice in LangSmith.
- **Haiku 5.5 becomes profiled:** `langchain-anthropic` 1.7.10 resolves `claude-haiku-5-5` at 128k.

`deepagents` 0.7.34 is the latest release, so the `<0.8` cap in `pyproject.toml` does not hold
anything back.

What `2026-10-21` changes for this project:

- `temperature`, `top_p` and `top_k` now return a 400 on every model. This repo never sends them,
  and `test_payload_omits_parameters_current_models_reject` guards that. The trial suite passed.
- `document` blocks in `assistant` messages and `citations` on `user` text return a 400. Neither is
  used here as far as is known, and the live turn is the check. Watch the path where a subagent's
  result is passed back.
- Admin API field removals do not apply.

Migration guide: https://platform.claude.com/docs/en/api/migration-2026-10-21

### Steps

1. Run `uv lock --upgrade`, then `uv sync`. Leave the floors in `pyproject.toml` unchanged, as
   `52fa457` did. CI tests only the locked versions, so a raised floor would claim nothing new.
2. Run the gates listed under Verification below.
3. Back up the real voice profiles and drafts, because the live turn below runs against them and
   nothing has exercised the upgraded stack yet. Keep the copy outside the repo:
   ```bash
   tar czf ~/speechwriter-backup-$(date +%F).tgz .speechwriter workspace
   ```
4. Run one live turn that drafts, calls the style critic and revises. The first live turn cost
   about $0.15 for a keynote. Get the user's approval first:
   ```bash
   printf 'BRIEF\nexit\n' | uv run speechwriter
   ```
   Confirm there is no 400, and that thinking, `fallbacks` and both betas are accepted on the new
   API version.
5. Run `uv run python evals/sync_datasets.py` to confirm the LangSmith client still works. It
   needs the network but no model calls.
6. Run `uv run python evals/run_experiment.py --langsmith --limit 1` to confirm experiments still
   record. Costs cents.
7. Start the web UI and load both pages, since Streamlit moves three minor versions.
8. In the web UI, press **Measure** on the draft from step 4, and confirm the measured length and the
   audio player appear. Project notes from 2026-09-30 list Deepgram as not yet verified live. It is
   billed per character, so this costs cents. Use the button rather than the opt-in
   `test_measured_length_is_in_the_right_ballpark`, because that test reads the key from the shell
   environment and the app loads it for itself.
9. Update `CLAUDE.md` wherever a fact changed, such as the API version and any pin it quotes.
10. Commit `uv.lock` alone. Name the `anthropic` major version and the API version change in the
   message. `release.yml` triggers on `pyproject.toml`, so this does not publish a release.

11. In a separate commit, move `ty` to the newest version across the five sites, as described under
    Tool pins. Check the newest version first, and run the gates.

**If a live call fails with a 400:** read the message first, and suspect the `fallbacks`, `betas`
or `thinking` fields. If it cannot be fixed quickly, hold `anthropic` on 1.x by adding
`constraint-dependencies = ["anthropic<2"]` under `[tool.uv]` in `pyproject.toml`, re-running
`uv lock --upgrade`, and telling the user. That edit fires `release.yml`, but its version check
stops it because the version is unchanged. If Streamlit stays behind 1.67 for any reason, leave `ty`
at 0.0.63. Do not remove `fallbacks` or `block_binding` to make the call pass: each protects against a specific
failure described in `CLAUDE.md`.

### Tool pins

`ruff` and `ty` are pinned outside `uv.lock`, across the five sites named in `CLAUDE.md`.
`test_tool_pins_agree_wherever_they_are_declared` checks that they agree, per tool.

- **`ruff` is done.** It moved from 0.16.0 to 0.17.4 on 2026-11-06, and all gates passed.
- **`ty` must move with this workstream.** Moving it to 0.0.99 is decided, but it cannot happen before
  the lock upgrade. On Streamlit 1.64 it reports `call-non-callable` at `st.navigation` in
  `streamlit_app.py`, and on 1.67 that error is gone. Both 0.0.63 and 0.0.99 pass on 1.67. Do not add a
  `ty: ignore`, because the newer version would reject it as unused after the upgrade.

### Small fix: permission rules for the pinned tools

The `allow` list in `.claude/settings.json` has `Bash(uvx ruff check:*)`,
`Bash(uvx ruff format:*)` and `Bash(uvx ty check:*)`. The project requires the pinned spelling
(`uvx ruff@0.17.4 check`), which those prefixes do not match, so they cover only the unpinned
commands the pin rule warns against.

1. Replace the three rules with wildcard forms that match any pinned version, such as
   `Bash(uvx ruff@* check:*)`, so a version bump does not add a sixth site to keep in step.
2. Confirm in the Claude Code permission docs that a wildcard is allowed in that position, then
   confirm the pinned commands run without a prompt. Neither has been checked yet.
3. If wildcards do not work there, spell out the pinned versions, and add `.claude/settings.json`
   to `test_tool_pins_agree_wherever_they_are_declared` and to the site list in `CLAUDE.md`.

Commit this separately from `uv.lock`.

### Small fixes: CI workflows

Do these in one commit, separate from `uv.lock`. Run the suite afterwards, because
`test_tool_pins_agree_wherever_they_are_declared` parses these workflow files.

1. **Stale comment on pins.** The `env:` comment in `ci.yml` says the hooks use bare `uvx ruff` and
   `uvx ty`. `.claude/hooks/ruff-ty-gate.sh` now pins both, so rewrite the comment.
2. **Stale comment on formatting.** The `lint` job's comment says `ruff format --check` is omitted
   because it would rewrite Markdown. `pyproject.toml` now excludes `*.md`, so the reason no longer
   holds. Remove it as part of item 3.
3. **Add a format check.** Add `uvx ruff@"$RUFF_VERSION" format --check .` to the `lint` job in
   `ci.yml`. It passed with `ruff` 0.17.4 on the trial copy, but run it at whichever pin is current.
   Leave `release.yml`'s gate re-run alone: formatting should not stop a release.
4. **Pin actions by SHA in `release.yml`.** It is the only workflow with `contents: write`. Replace
   `actions/checkout@v7` and `astral-sh/setup-uv@v9.0.0` with full commit SHAs, keeping the version
   in a trailing comment. Look each SHA up with `gh api`, and never write one from memory.
   `ci.yml` is read-only, so it can stay on tags.

### Small fix: `.gitignore`

Only the exact name `.env` is ignored today, so a copy such as `.env.local` could be committed with
real keys. Neither `env-guard.sh` nor CI stops a file being committed. Do this in its own commit.

1. In the `# Secrets` block, add these lines after `.env`, in this order, because a negation only
   works after the rule it overrides:
   ```gitignore
   .env.*
   !.env.example
   ```
2. Add `.claude/settings.local.json`, which holds personal Claude Code settings. `.claude/` is
   otherwise tracked.
3. Add `dist/` under `# Python`, since `uv build` writes there.
4. Check the result with `git status --short`, and with `git ls-files` to confirm the example
   template is still tracked.

Use the `Edit` tool for this. `env-guard.sh` blocks any Bash command containing the `.env` token,
and that includes a heredoc or `sed` writing these lines. The same applies to Bash checks, so use
the commands above, which do not name the file.

## Workstream 2: Model and effort tuning

### Goal

Replace the untested choice of one model and one effort level for every role with settings
measured against the evals. Spend less where quality holds, and spend more only where it shows.

### Starting point

Verified in the code on 2026-11-06, before Workstream 1, unless marked otherwise.

- `config.DEFAULT_MODEL` is `claude-sonnet-5-5` and `config.DEFAULT_EFFORT` is `medium`.
- `agent._chat_anthropic` reads `DEFAULT_EFFORT` directly, so effort cannot vary per run or role.
- `cli.py:138` prints `DEFAULT_EFFORT` in the banner, not the effort of the built client.
- Neither subagent spec sets `"model"`, so both inherit the orchestrator's client, effort included.
- deepagents 0.7.21 (re-check after Workstream 1) types `SubAgent["model"]` as `NotRequired[str | BaseChatModel]` and reads it
  with `spec.get("model", model)`, so a subagent can take its own constructed client.
- `run_experiment.py` has `--model` but no effort option. `invoke_agent` calls `build_agent()`
  with no arguments, and the judge goes through `agent._build_model`.
- The eval harness records no token usage or cost itself.
- `langchain-anthropic` profiles Sonnet 5.5 and Opus 5.5 at 128k. After Workstream 1 it also
  profiles `claude-haiku-5-5` at 128k (1.7.5 does not).

Previous live results, from project notes dated 2026-09-30:

- These predate Workstream 1, so re-run the Control arm rather than comparing with them directly.
- The trajectory baseline is 118/118 scored criteria on Sonnet 5.5 at `medium`, at about $1.60
  and 14 minutes for all 13 examples.
- `final_response`, `single_step` and `rag` had not been run live as of those notes.
- Sonnet 5.5 at `medium` once skipped the style critic. A prompt gate fixed it (`3bfaa58`), and
  lower effort could bring that back.

### Limits of the evals

- **Trajectory is at its ceiling.** At 118/118 it can show that a setting got worse, but not that
  one got better.
- **Criteria are pass/fail.** They check whether a draft meets the brief, and they cannot rank two
  acceptable speeches. The blind human read in Phase 2 therefore makes the decision, and the
  criteria only guard against regressions.
- **Runs vary.** The intake branch diverges between identical runs, so a difference of one or two
  criteria on single runs is noise. Use repeat trials.

### Rules for this workstream

- Do **not** add a `SPEECHWRITER_EFFORT` variable. Effort stays a repo decision, as the comment on
  `DEFAULT_EFFORT` explains, and a new `SPEECHWRITER_*` name in `config.py` trips
  `test_env_example_documents_every_setting`, which only the user can clear.
- The judge must not move with the system under test. Effort therefore travels as an explicit
  argument, never through the environment, so `grade()` keeps building the judge at the default.
- Never hard-code a model id on a subagent in Phases 1 to 3. Derive its client from `settings`, so
  `/model` and the picker still reach every role through the existing rebuild.
- `build_agent()` must stay offline. Constructing extra clients performs no network I/O.
- Live runs cost money. Get the user's approval, with the expected cost, before each paid phase.
- Do not bump the version here. A change to `[project] version` on `main` publishes a release, and
  that happens once, under Release 0.3.0.

### Phase 0: Prove the evals (code, then paid, then the user's reading)

Do this first. Three of the four datasets have never produced a live score, and nobody has checked
the judge against a human reader, yet the regression check in Phase 2 depends on both. Every
measurement bug found so far came out of a live run.

**Step 1: Harness additions (code, free).** Mutation-test each, as in Phase 1.

- **`--trials N`** in `evals/run_experiment.py`, default 1, so repeats no longer need a shell loop.
  On the plain live path, run each example N times and report per trial. On `--langsmith`, pass
  `num_repetitions=N` to `langsmith.evaluate`, which is confirmed in the installed version. Check that
  a recorded experiment still grades like a local run. Each trial needs its own temp home, and
  `--trials` with `--dry-run` exits 2, as `--model` does.
- **Agent and judge cost per run.** Run one `--langsmith --limit 1` experiment and confirm LangSmith
  reports a cost for `claude-sonnet-5-5` and separates the agent's cost from the judge's. If it
  cannot, add `langchain_core.callbacks.UsageMetadataCallbackHandler` to the config returned by
  `bundle.turn_config(...)` and record usage in the run output. The handler import is verified, but
  its propagation into subagents and its behaviour with streaming are not, so test both. This one
  experiment is the only paid call in Step 1, and it costs cents.
- **Runs that can be re-graded.** `--keep` leaves only the run's files today, in a system temp folder
  that macOS can clear. The tool-call sequence and reply are lost, so a fixed scorer cannot be applied
  without paying for the run again. Three changes:
  - Keep homes under `.speechwriter/eval-runs/<experiment>/<example>-<trial>/`, which is gitignored.
  - Write the `RunRecord` there as JSON, and include each call's message index, so calls made in the
    same model message can be told apart.
  - Add `--regrade <dir>`, which scores a saved record with no agent call. The judge still costs a
    little, and `--no-judge` makes it free.

**Step 2: Full baseline (paid).**

1. Run `uv run python evals/sync_datasets.py`. An experiment grades the server's copy of the criteria,
   so the mirror must be clean. Use `--push` if it has drifted. `--allow-delete` is irreversible, so
   ask the user first.
2. Price it. Run `--langsmith --limit 1 --keep` on `final_response`, `single_step` and `rag`. Only
   `trajectory` has a known cost. `rag` needs `TAVILY_API_KEY`, which the user has set.
3. Show the user the expected total and get approval.
4. Run all four datasets in full on the Control settings (Sonnet 5.5 at `medium`), one trial each,
   with `--langsmith --keep`. `--limit` defaults to 1, so pass a value above the dataset's size.
5. Triage every failure by reading the kept run, and sort it into one of three groups:
   - **Measurement bug:** fix the scorer, the dataset or the harness, then re-run that example.
   - **Real agent defect:** report it to the user. Do not change a prompt as part of this phase.
   - **Allowed variation:** the example licenses more than one outcome, so leave it alone.
6. Check how the style critic received each draft. On 2026-09-30 the orchestrator sent `write_file`
   and the critic's `task` in the same message and told the critic to read the file itself, so the
   critic could find a missing or old file. No current check would catch this: `check_order_constraint`
   compares list positions, so parallel calls pass, and Phase 2 only checks that the critic ran.
   `orchestrator_prompt` already says to give the critic the full draft. For each run, record whether
   the two calls shared a message, and whether the `task` description held the draft. Sharing a message
   is harmless if the draft was inside. If the critic relied on the file, report it to the user, and
   add a scored check before Phase 2.

Record the scores, costs and experiment names in this plan. If nothing the scorers depend on changes
afterwards, these runs count as one of the three Control trials in Phase 2.

**Step 3: Check the judge (the user's reading, no API cost).**

1. From the kept runs, draw about 25 judged criteria spread across the datasets, including every kind
   of verdict and more of the judge's fails than chance would give.
2. Write them to a sheet outside the repo with the criterion, the graded text and the example's
   `grading_notes`, leaving out the judge's verdict.
3. The user marks each pass or fail. Then compare the two.

The sheet takes about an hour. Proposed bar, for the user to confirm when they see the results: at
least 90% agreement, and no more than one case where the judge passed something the user failed. That
direction matters most, because a lenient judge would hide regressions and Sonnet is judging Sonnet's
work here.

If it falls short, fix the criteria or `grading_notes`, push the mirror and use `--regrade` on the
kept runs.
Keep the judge model as it is (see Out of scope). Continue to Phase 2 only once the user is satisfied.

### Phase 1: Make effort selectable (code, free)

1. **`config.py`:** append `effort: Effort = DEFAULT_EFFORT` as the last `Settings` field, so
   positional callers do not shift. `ModelChoice.applied_to` uses `dataclasses.replace`, so the
   value survives a model switch.
2. **`agent.py`:** `_chat_anthropic` sends `effort=settings.effort`. Remove the `DEFAULT_EFFORT`
   import if nothing else uses it.
3. **`cli.py`:** print `bundle.settings.effort` in the banner, and drop the import if it is unused.
4. **`evals/run_experiment.py`:**
   - Add `--effort`, with choices taken from the `Effort` literal.
   - Add a keyword `effort` to `invoke_agent` and thread it through `run_one` and the inner
     `run_agent` in `run_langsmith`.
   - In `invoke_agent`, build with `build_agent(dataclasses.replace(load_settings(), effort=effort))`.
     Call `load_settings()` inside the `try`, after `SPEECHWRITER_HOME` is set, or the run loses
     its temp home.
   - Reject `--effort` with `--dry-run`, as `--model` is rejected (exit 2).
   - Add the effort to `experiment_prefix` and to the experiment `metadata`.

Tests, each mutation-tested per `CLAUDE.md`:

- An overridden `Settings.effort` reaches `output_config.effort` in the request payload.
- The effort survives `ModelChoice.applied_to`.
- `--effort` with `--dry-run` exits 2.
- `invoke_agent` builds with the requested effort and keeps the temp home.
- With `--effort` set, `grade()` still builds the judge at `DEFAULT_EFFORT`, with and without
  `--model`. This test protects the comparison, so it matters most.
- The experiment prefix and metadata name the effort.
- The CLI banner reports the built effort.

### Phase 2: Sweep the main settings (paid)

**Arms:**

| Arm | Model | Effort | Purpose |
|---|---|---|---|
| Control | Sonnet 5.5 | `medium` | Current default |
| High | Sonnet 5.5 | `high` | Does more thinking help? |
| Opus | Opus 5.5 | `medium` | Does the bigger model help? |

`low` is deliberately not an arm. A 5-minute keynote cost about $0.15 in the 2026-09-30 notes, so
at personal volume (a few speeches a week) `low` would save cents per speech and could never repay
this sweep, while speech quality is the product. Lower effort is tested on the researcher in
Phase 3 instead. Revisit this if volume becomes much higher.

Run `trajectory` and `final_response`, with 3 trials per arm per dataset, using `--trials 3`,
`--langsmith` and `--keep`. Keep the judge on the configured Sonnet 5.5 throughout, so scores stay
comparable with the baseline. Run the arms one at a time: the harness is sequential by design.

**Cost estimate, to refine with the Phase 0 figures.** Trajectory alone is 9 runs, about $21 and at least 2.5
hours, extrapolated from the single $1.60 baseline with Opus at 2× and `high` somewhat above that.
`final_response` has no measured cost yet. Plan on roughly $45 to $50 in total, and confirm the
figure with the user before starting.

**Blind read.** Pick 5 varied commissions where all three arms produced a draft, taken from the
drafts `--keep` saved, so the read needs no new API spend. Strip the labels, shuffle them, and have
the user pick a favourite among the three for each commission without knowing the arm. Hold back
5 more commissions in case the rule below needs them. With three trials per arm, use the first trial
that produced a draft, so the choice cannot favour an arm. Write the key from label to arm to a
separate file, and do not show it or open it until the user has picked.

Record every read in `evals/blind_reads.jsonl`, one line per commission: the date, the example id,
the experiment names, the arm behind each anonymous label, the pick, and any note from the user.
Commit it with a copy of the drafts under `evals/blind_reads/`, because `--keep` homes are temporary.
This is the only record of what the user thinks is a better speech, and it stays useful after the
sweep. Keep it outside `evals/datasets/`, which the validator treats as the full list of datasets.

**Decision rule, confirmed by the user on 2026-11-06:**

1. **Regression check.** Every candidate (High and Opus) must pass all of these:
   - Its average score across trials is not below the Control arm's worst trial.
   - `style-critic` runs on every draft it presents.
   - No example fails all 3 trials unless it also fails all 3 on Control.
2. **Blind read.** For a candidate that passed the check:
   - 4 or 5 wins out of 5: it qualifies.
   - Exactly 3 wins: read the 5 held-back commissions, and it qualifies at 6 of 10.
   - Anything else: it does not qualify.
3. **Tiebreak.** If both candidates qualify, take the cheaper one.
4. **Cost.** Show the user the measured cost per speech before adopting a candidate. That final
   call is theirs, not a formula's.
5. **Default.** If nothing qualifies, keep Sonnet 5.5 at `medium`. If the user cannot tell the
   drafts apart, that is a valid result and means the cheaper setting is good enough.

### Phase 3: Per-role effort (code, then paid)

Do this after Phase 2, since the main setting is the baseline for these comparisons.

1. **`config.py`:** append `researcher_effort: Effort | None = None` and
   `critic_effort: Effort | None = None`. `None` means the role inherits the main client.
2. **`agent.py`:** when an override is set, build the role's client with
   `_build_model(dataclasses.replace(settings, effort=override))`. Pass the clients to
   `build_subagents`, because `subagents.py` cannot import `agent.py` without a cycle.
3. **`subagents.py`:** add a `"model"` key only when a client was passed. Keep the `list[SubAgent]`
   typing. With no override the key stays absent and inheritance is unchanged.
4. **`evals/run_experiment.py`:** add `--researcher-effort` and `--critic-effort`, threaded the
   same way as `--effort`.

Tests, in addition to the Phase 1 pattern:

- With no override, neither spec has a `"model"` key.
- An override reaches that role's payload and leaves the orchestrator's unchanged.
- After a `/model` switch, a role with an effort override runs on the new model.
- `test_prompts_only_advertise_tools_the_model_can_call` still passes. It stubs `_build_model`.

**Experiments, 3 trials each.** Price these from the Phase 0 figures, and get the user's approval first:

- Researcher at `low` against inherited, on `rag`. Published research curves are nearly flat, so
  expect savings here. `rag` is judged on the saved note as well as the reply, so read the saved
  notes with `--keep` before believing a surprising score.
- Critic at `high` against inherited, on `trajectory`, and compare the speech after revision. The
  set is at its ceiling, so include this in a blind read if the cost difference is meaningful.

### Phase 4: Haiku 5.5 for the researcher (optional)

Do this only if traces show research is a large share of the cost after Phase 3. It costs 20× less
than Sonnet ($0.10/$0.50 against $2/$10 per million tokens, at prices cached 2026-10-06), but it
adds the following:

- **Fallback settings.** Haiku 5.5 has no server-side refusal fallback, and `_chat_anthropic` sends
  `fallbacks` and two betas on every call. Confirm against the live docs what Haiku accepts, then
  give it its own call settings.
- **Refusals.** An unrescued refusal reaches the orchestrator as an empty success, so check
  `TruncationWarner`'s refusal count during the trial.
- **Model switching.** A pinned researcher stops following `/model`. Both front ends should say so,
  and the "subagents inherit the model" note in `CLAUDE.md` needs an exception.

Haiku stays off `MODEL_CHOICES`, since this is a researcher setting and not a drafting choice.

### Phase 5: Adopt the results (code and docs)

1. **Pin the judge first.** `grade()` builds the judge from the configured model and from
   `DEFAULT_EFFORT`, so changing either default would also change the judge, and later scores could
   not be compared with the baseline. Add `JUDGE_MODEL` and `JUDGE_EFFORT` constants in `evals/`, set
   to the values used for the baseline, and build the judge from them on every path. Add a test that
   changing the defaults leaves the judge alone, and mutation-test it. The judge's credentials must
   still come from the configured settings, so re-read the `--model` capture in `main()` before
   simplifying it. Do this even if the defaults stay the same.
2. **Check the adopted setting on all four datasets.** Phase 2 runs only `trajectory` and
   `final_response`, so a winner could still regress on intake or research. If the setting changed,
   run it once on `single_step` and `rag` and compare with the Phase 0 baseline. Show the user any
   drop before adopting. Get approval for the cost, which Phase 0 will have measured.
3. Set the measured values in `DEFAULT_EFFORT`, `DEFAULT_MODEL` and the role defaults. The id
   `claude-sonnet-5-5` appears in 11 lines across `src/`, `tests/`, `evals/` and `README.md`, so
   check each one if the model changes.
4. If the default model changed, ask the user to check their dotenv file. A `SPEECHWRITER_MODEL` there
   overrides `DEFAULT_MODEL`, and Claude cannot read that file.
5. Update the rationale comments in `config.py` with the measured results and the experiment names.
6. Update `CLAUDE.md`: the model section, the `DEFAULT_EFFORT` rationale, the judge pin, and a note
   that the Haiku exclusion reason concerns 4.5 while 5.5 was assessed separately.
7. Update `README.md` if the default model or banner changes.

### After this workstream: grow the datasets from real failures

Do not write more examples speculatively. When a real speech disappoints the user, turn that brief
into an example, with the criterion it should have failed. Each new example needs `EXPECTED_COUNTS`
updated, `evals/validate_datasets.py` passing and a `sync_datasets.py --push`. Real failures are also
the way to bring `trajectory` off its ceiling.

### Out of scope

- **Fable 5.1:** about 5× Sonnet's price, with long turns and a 30-day retention requirement. Its
  prose can also be denser, which is a poor fit for speech.
- **`xhigh` and `max`:** revisit only if the High arm shows clear headroom over `medium`.
- **Changing the judge model:** this would break comparability with the baseline. If the user wants
  a stronger judge, make it a separate change with a fresh baseline.
- **Live evals in CI:** they would need an API key in CI, which the project rules forbid, and a run
  takes too long. CI keeps the offline checks.
- **An AI judge of speech quality:** it would be unproven. Once `evals/blind_reads.jsonl` holds around
  30 picks, test whether a judge can predict them, and trust it only if it does.
- **Tightening the intake prompt:** this was measured as worse and reverted, as `CLAUDE.md` records.

## Workstream 3: Web UI theme

### Goal

Replace the One Light and One Dark code-editor palette with "The Chamber", a pair designed for
this app. The user approved this direction on 2026-11-06. It now lives in `.streamlit/config.toml`;
the draft file and its preview renders were deleted in step 7.

### Design

- **Concept:** the green of a parliamentary bench, in daylight and with the house lights down. The
  palette comes from where speeches are given. Green also leaves red free for errors.
- **Complementary modes:** the accent and links stay at hue 158 to 162 in both modes, and every role
  keeps its place: the page is the lightest surface, panels step away from it, and the sidebar
  takes the same step. Only the light level changes.
- **Type:** Literata, a serif drawn for long reading on screens, at 16px instead of 14px. Streamlit
  has no separate UI font, so it carries the controls too. Source Code Pro stays for code.

Measured values:

| Check | Light | Dark |
|---|---|---|
| Body text on page | 11.09:1 | 8.44:1 |
| Links, tightest surface | 6.23:1 | 8.36:1 |
| Primary as text, tightest surface | 5.34:1 | 4.79:1 |
| White icon on primary | 6.40:1 | 3.11:1 |
| Captions, as rendered | 3.49:1 | 4.04:1 |

Findings behind those numbers:

- **The current theme has a defect this fixes:** dark mode's primary gives white only 2.36:1.
- **Dark primary is a deliberate compromise.** Streamlit uses primary as text on selected tabs and
  pills, and as the fill under the white icon on the chat send button. No colour reaches 4.5:1 for
  both on this background, so it meets 4.5:1 as text and 3:1 as an icon. A `type="primary"` button
  with a white label would fall short at 3.11:1, and the app has none today.
- **Captions are under AA in both modes.** Streamlit renders them as a 60% mix of text and
  background, and no config key sets that colour. Reaching 4.5:1 through `textColor` would need
  about 16:1 body text in light mode and at least 11:1 in dark, so step 2 below handles it instead.
- **Widget borders are about 1.6:1**, close to the current theme and under the 3:1 guideline for
  control boundaries. This is unchanged rather than solved.
- **Measure rendered pixels, not computed styles.** A computed-style check reported captions at
  body contrast, which was wrong. Sampling a screenshot gave the true figure.

### Steps

1. Replace everything below the `[server]` block in `.streamlit/config.toml` with
   `docs/plans/chamber-theme.toml`. Keep the `[server]` block exactly as it is.
2. In `app_pages/browse.py`, change the speaker, audience and occasion block from `st.caption` to
   `st.markdown`, because readers need that information. It is safe: the string is rebuilt from
   parsed pairs, so it contains no `---` fence. Update the comment above it, and any test that finds
   this block as a caption.
3. Extend `tests/test_webapp.py`, near `test_theme_links_clear_wcag_aa_and_stay_visible_without_color`:
   - `primaryColor` is at least 4.5:1 on every surface, and white on it is at least 3:1, in each mode.
   - `textColor` is at least 4.5:1 on every surface, in each mode.
   - Primary and link hues agree between modes to within a few degrees, which is what keeps the pair
     complementary. Leave out the page surfaces, whose tints are too faint for hue to be stable.
   - No file in `app_pages/` or `streamlit_app.py` uses `type="primary"`, with a comment on why.
   - Update that test's comments, which quote One Light and One Dark values.
4. Mutation-test the new assertions by setting the old `#61afef` primary and confirming a failure.
5. Render the app with a draft in the workspace and check long text, an open expander and a
   finished turn, none of which the previews show. Use Playwright with the installed Chrome:
   `uv run --no-project --with playwright python <script>`, with `channel="chrome"` and
   `color_scheme` set per mode. Keep that script outside the repo, because `ruff` and `ty` check
   everything inside it. Plain headless Chrome screenshots catch only the loading screen.
6. Replace the four images in `docs/images/` that `README.md` uses. If Workstream 4 is going ahead,
   leave this until it is finished, so the images are replaced once.
7. Delete `docs/plans/chamber-theme.toml` and `docs/plans/theme-preview/`.

No dummy key is needed for a real check, but `ANTHROPIC_API_KEY=sk-ant-dummy` enables the chat input
for a render without sending anything, as long as the message is typed and never submitted.

## Workstream 4: Web UI layout

### Goal

Make the app fit any display by keeping one thing constant: the width of the text being read. Only
what surrounds that column changes. The user approved this direction on 2026-11-06.

It was prototyped in a scratch copy of the app on Streamlit 1.64 with the Chamber theme, and rendered
in Chrome at eleven sizes. Workstream 1 moves Streamlit to 1.67, so treat the wrap points as something
to re-measure. `st.container(wrap=...)` exists at the 1.62 floor, so the floor can stay. Nothing in the
repo has changed. Renders are in
`docs/plans/layout-preview/`.

### Design

- **Streamlit cannot tell Python the screen size.** `st.context` has no viewport, so a page cannot
  branch on it. All adaptation has to come from the browser's flex layout.
- **Workspace** becomes a row of two parts inside a centred 896px frame: a rail with the picker,
  stats, buttons and speaker details, and a 560px reading column. With room they sit side by side.
  Without room they wrap, rail first, so the picker and details stay above the speech.
- **Write** stays `layout="centered"`, because chat is one column. Prose inside a reply takes the
  same 560px cap.
- **No CSS and no breakpoints.** The row wraps on the space it actually has, so the sidebar counts
  too. At a fixed 1280px window, closing the sidebar moved the page from stacked to side by side.
  A rule based on screen width could not do that.

| Display | Workspace arrangement | Reading width |
|---|---|---|
| Phone, 360 | Stacked | 328px |
| Tablet, 768 | Stacked | 560px |
| iPad with sidebar open, 820 to 834 | Stacked | 488 to 502px |
| Small laptop or landscape tablet, 1024 to 1180 | Stacked | 560px |
| Laptop, 1280 | Side by side, rail 196px | 560px |
| 1366 and wider | Side by side, rail 272px | 560px |

Measured against today, using a 616-word draft:

| Check | Today | Prototype |
|---|---|---|
| Longest line, tablet and wider | 86 characters | 74 characters |
| Speech starts, phone | about 630px down | 595px |
| Speech starts, tablet | about 560px | 546px |
| Speech starts, 1280 and wider | about 560px | 252px |
| Reading width, phone | 296px | 328px |
| Wide table in a research note, phone | Page scrolls sideways by 150px | Only the table scrolls |

The existing suite passed against the prototype, 152 passed and 1 skipped. No test asserts layout,
so that only shows nothing else broke.

Findings that shaped the design:

- **The rail must stretch, and the frame must cap it.** A fixed rail stacks at 272px and wastes the
  row beside it, which pushed the speech down to 647px at tablet width. A stretchy rail with no cap
  grew to 1096px on a wide screen.
- **`gap="large"` is what sets the wrap point.** Streamlit lets a stretchy container shrink to roughly
  128px, and with `gap="medium"` a 768px tablet went side by side with a 144px rail. The 64px gap
  stacks it instead. `PAGE_WIDTH` includes that 64px, which scales with the root font size.
- **`wrap=False` on the reading column is the table fix.** It makes that column scroll sideways
  rather than the page, and prose still wraps at the column width. It only suits a container with
  one child, because `wrap=False` lays children out in a row.
- **`layout` can differ per page.** Calling `st.set_page_config(layout="wide")` at the top of
  `browse.py` overrides the router's `centered`, and moving back and forth between the pages
  switched cleanly each time.

### Target code

`src/speechwriter/webui.py`, single-sourced so both pages share the measure:

```python
# About 74 characters of Literata at 16px at the longest. Tune it against the rendered line, not by
# arithmetic, and re-check it if the font or `baseFontSize` changes.
READING_WIDTH = 560
```

Both `prose = st.container()` lines in `webui.py` become `st.container(width=READING_WIDTH)`.

`app_pages/browse.py`, at the top of the file:

```python
st.set_page_config(layout="wide")

RAIL_WIDTH = 272
LARGE_GAP = 64  # What `gap="large"` renders at the 16px root size.
PAGE_WIDTH = RAIL_WIDTH + LARGE_GAP + READING_WIDTH
```

The body of `document_browser`, after the empty check. Keep the existing comments, `help` text and
`Measured` metric. They are shortened here:

```python
labels = {f"{doc.slug}  ·  {doc.modified:%b %d, %H:%M}": doc for doc in documents}
with st.container(horizontal=True, gap="large"):
    with st.container():  # The rail. Stretchy, so it fills the row when stacked.
        picked = st.selectbox("Document", list(labels), ...)
        document = labels.get(picked) if isinstance(picked, str) else None
        if document is None:
            return
        measured, notice = _measure_if_requested(document) if spoken else (None, "")
        # One wrapping row: stats and buttons share a line wherever they fit.
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.metric("Words", ..., width="content")  # No border.
            if spoken:
                st.metric("Spoken length", ..., width="content")
                if measured is None:
                    _measure_button(document)
                else:
                    st.metric("Measured", ..., width="content")
            st.download_button("Download", ...)
        if notice:
            st.info(notice, icon=":material/volume_off:")
        if measured is not None and measured.wav:
            st.audio(measured.wav, format="audio/wav")
        if document.front_matter:
            st.markdown(...)  # Workstream 3, step 2.
    # `wrap=False` so a wide table scrolls here rather than taking the page with it.
    with st.container(width=READING_WIDTH, horizontal=True, wrap=False):
        st.markdown(document.body)
```

The page body, from `st.title` down, sits inside the centred frame:

```python
with st.container(horizontal_alignment="center"):
    with st.container(width=PAGE_WIDTH):
        st.title("Workspace")
        view = st.segmented_control(...)
        ...
```

In the Memory view, wrap the expander loop in `st.container(width=READING_WIDTH)`.

### Steps

1. Add `READING_WIDTH` to `webui.py` and apply it to both `prose` containers.
2. Rework `document_browser` and the page body as above. This drops the bordered box around the
   speech, the borders on the metrics, and the caption under the title. The segmented control
   already names the three views, and these removals are what pay for the phone's height.
3. Update the comments this makes false, including the one saying the spinner needs full width, the
   `_measure_if_requested` docstring and the note in `streamlit_app.py` if it mentions the layout.
4. Add tests to `tests/test_webapp.py`. `AppTest` does not render, so check what can be checked:
   - `PAGE_WIDTH` equals `RAIL_WIDTH + LARGE_GAP + READING_WIDTH`.
   - The Workspace page still renders the picker, metrics and body in all three views.
   - Read `browse.py` source for `wrap=False` on the reading column, with a comment saying a browser
     check is the real guard, and for `st.set_page_config(layout="wide")`.
   - Confirm `streamlit_app.py` still sets `layout="centered"`, so Write is not widened by accident.
5. Mutation-test the new assertions, by removing `wrap=False` and changing `RAIL_WIDTH` alone.
6. Run the render check below.
7. Update the **Web UI** section of `CLAUDE.md` with the design and the four findings above. Replace
   the four README images, as Workstream 3 step 6 describes.
8. Delete `docs/plans/layout-preview/`.

### Render check

CI has no browser, so this is manual and is the real test of the layout. Use the Playwright recipe in
Workstream 3 step 5, with a draft of at least 600 words, a three-field header, a research note holding
a wide table and a long URL, and a draft with a very long slug. Check these sizes: 320x568, 360x740,
780x360, 768x1024, 820x1180, 834x1194, 1024x768, 1180x820, 1280x800, 1366x768, 1440x900, 1920x1080 and
2560x1440.

- The page never scrolls sideways on Workspace, in any view. Compare `scrollWidth` and `clientWidth`
  on the `section` around `stMainBlockContainer`, because `stMain` is missing on some pages.
- The rail is never under about 190px while it sits beside the reading column.
- Closing the sidebar at 1280 moves the page from stacked to side by side.
- Going from Write to Workspace and back switches the page width each time.
- The rail holds three metrics and the audio player without clipping. Deepgram is not needed: in a
  scratch copy, return a stub `SpokenLength` from `_measure_if_requested`.
- Both colour modes look right.
- At 1280x800 with the browser zoomed to 200%, the page still fits. Zoom shrinks the space in CSS
  pixels, so it should stack like a smaller screen. The 320px width and the zoom were not prototyped.

### Known limits

These were measured and left alone, because each needs CSS or more change than it is worth:

- **A wide table in a chat reply still overflows on a phone,** by 156px. The `wrap=False` fix cannot
  apply, because a reply's prose container holds several elements and they would sit in a row. Speech
  replies rarely contain tables, and research notes, where tables belong, are fixed.
- **Chat text on a phone is about 33 characters wide.** The avatar and message padding take 88px, and
  `st.chat_message` has no way to remove the avatar.
- **A phone in landscape opens with the sidebar showing,** which takes 300 of its 780px. Streamlit collapsed
  it automatically at 768px but not at 780px, and Python cannot see the height. The reader can
  close it. Setting `initial_sidebar_state="collapsed"` for everyone would hide the model picker and
  status on desktop.
- **The rail scrolls away with a long speech.** Pinning it needs CSS or a fixed-height inner scroller,
  and that would trap scrolling on phones.
- **The Write and Workspace links are not in the header at 360px.** Check where they end up during
  the render check. This was seen in the renders and not explored.
- **The rail gets narrow in one band of window widths.** By calculation, with the sidebar open it is
  about 128 to 190px between roughly 1212 and 1274px wide, and the same happens at about 912 to 974px
  with the sidebar closed. It still works there, with the stats stacked and the date cut from the
  picker. None of the tested sizes fall in that band.
- **About 90px of Streamlit's top padding stays.** I found no config key for it, so removing it would
  need CSS.

## Release 0.3.0

One release, after all four workstreams. Phase 4 can be left out if it was not needed.

1. Confirm that CI is green on `main`, and ask the user for the go-ahead. `release.yml` publishes
   unattended the moment the version changes on `main`, with no draft step.
2. Set `[project] version` in `pyproject.toml` and `__version__` in `src/speechwriter/__init__.py` to
   `0.3.0`.
3. Run `uv lock`. `uv.lock` records the project's own version and CI runs `uv sync --locked`, so a
   stale lock fails the run. `2e97a9f` shows the same three-file commit.
4. Run the gates, then commit those three files alone, as `Release 0.3.0`.
5. Push, and follow the run with `gh run watch`. `release.yml` re-runs the gates before tagging.
6. Check the result with `gh release view v0.3.0`. Its notes are generated from commit messages, so
   add a short summary at the top that names the `anthropic` 2.x and API version change and any new
   default model.
7. Ask the user whether to keep `docs/plans/upgrade-plan.md` as a record or delete it.

## Open questions for the user

**Should CI block merges?** Nothing requires the checks today, so a red run is advisory. Making
them real means requiring `ruff check` and `pytest + ty (py3.11|3.12|3.13)` on `main` in the repo
settings. The user commits directly to `main`, so a strict rule could block those pushes, and a
required check only stops changes that go through a pull request. This is a repo setting and the
user's decision. Do not change it, and do not describe CI as enforcing anything until it does.

## Verification

For every workstream, after each code phase, all of these must pass:

```bash
uv run pytest
uvx ruff@0.17.4 check .
uvx ty@0.0.99 check
uv run python evals/run_experiment.py --dry-run
printf 'exit\n' | uv run speechwriter
```

Use the current pins. `ty` moves during Workstream 1.

Invoke `/astral:uv`, `/astral:ruff` and `/astral:ty` before working on the Python, as `CLAUDE.md`
requires. Add each import in the same edit as its first use, because the hook deletes unused ones.
Restore mutations from a working-tree copy, never with `git checkout`, and run mutation loops under
`PYTHONDONTWRITEBYTECODE=1`.

## Done when

**Workstream 1:**

- `uv.lock` is at the latest versions, all gates pass, and a live turn with the critic succeeded.
- Measure works live, `ty` is on the newest version, and the small fixes are committed.

**Workstream 2:**

- All four datasets have a live baseline, and the user has accepted the judge check.
- The critic is known to review the current draft, or a scored check now covers it.
- Effort is selectable per run and per role, with the judge provably unaffected.
- The judge is pinned, so a change of default cannot move it.
- Each adopted setting has an experiment name, a measured cost per speech and, for the main
  setting, a blind-read result behind it.
- `CLAUDE.md` and the code comments state the measured reason for the final values.

**Workstream 3:**

- The theme is live in both modes, the new contrast tests pass, and any preview files are removed.

**Workstream 4:**

- The render check passes at every listed size, the new tests pass, the README images match the app,
  and `CLAUDE.md` describes the layout.

**Release:**

- `v0.3.0` is published, and `import speechwriter` reports `0.3.0`.
