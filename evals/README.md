# Keyline evaluations

Run the commands below from the repository root with Python 3.11+, uv and Docker.
The evaluator and its dependencies live in the separate `evals` environment.

```sh
uv sync --project evals --locked
docker info
```

## Run SWE-bench trials

Set `DEEPSEEK_API_KEY` in your environment for commands with `--live`. Those
commands contact DeepSeek and can incur fees. The dataset and instance images
are pinned in `tasks.toml`.

```sh
uv run --project evals --locked python -m evals.swebench_run --live \
  --instance-id sympy__sympy-20590 --trials 1 --output-dir .keyline-agent/evals
```

Repeat `--instance-id` or select `--suite regression` or `--suite capability`.
Set `--max-requests`, `--max-steps` and `--agent-seconds` to control each
Trial's budget. Each invocation creates a new UUID directory under
`--output-dir` and prints the path to its `summary.json` on exit.
`experiment.json` records the Git commit, UTC start time, model settings, task
list with dataset revisions and image digests, Trial count and budget.

Each Trial uses one container with no external network access or host mounts.
File tools and commands share `/testbed`, which starts with the pinned base
commit and no Git remote. Files and installed environments in the container,
including `/tmp`, persist across commands. Each command starts a separate
process. The runner exports `candidate.patch` before removing the container
and submitting the patch for official scoring.

Shell and test commands default to 60 seconds and accept timeouts up to
that limit. On timeout the command's process group is killed and its captured
output is returned with a timeout message. Prefer invoking test runners directly
with `run_tests`. Direct `bash` and `sh` invocations in `run_tests` use Bash with
`pipefail` so a successful output filter cannot hide a failed test pipeline.

When a Trial pauses at the model-turn limit, `result.json` also records the
Docker image ID in `workspace_checkpoint`. A Budget Extension can copy the
Trial's state and artifacts, call `DockerWorkspace.prepare(checkpoint=...)`
with that ID on the same Docker host, and use `agent_setup` to resume the saved
Run with the same tool bindings. The checkpoint preserves files and installed
environments. Retain the Docker image while planning to extend that Trial.

### Verified80 historical artifacts

The original tree remains at `.handshake/evals/verified80-baseline`.
The archive and its index are in
`.handshake/eval-archives/verified80-baseline-2026-11-27`.
The historical Budget Extension scripts use the runner and workspaces saved
with those experiments.

## Score components

Set `TRIAL` to the directory of a completed Trial. Review the data before
sending it for Judge scoring. Choose a new `--output` for each scoring pass.
An existing result or sample capture is protected from reuse as an output.

```sh
uv run --project evals --locked python -m evals.tool_calling "$TRIAL" \
  --live --judge-model deepseek-flash --output "$TRIAL/tool_calling.json"

uv run --project evals --locked python -m evals.compaction \
  --sample retry-triage --live --compaction-model deepseek-flash \
  --judge-model deepseek-flash --output compaction.json
```

Use `--trial "$TRIAL"` to score captured compactions or `--saved CAPTURE.jsonl`
to rescore an existing capture. Omit `--judge-model` to compute available
deterministic compaction scores without a Judge.

## Generate and compare reports offline

Set `BATCH` to the saved `summary.json` or a single Trial's `result.json`.
Supply any subset of artifact types and repeat the component flags as needed.
Both commands write `.json` and `.md` files using the given output prefix.
Use a different prefix when either file already exists.

```sh
uv run --project evals --locked python -m evals.report generate \
  --batch "$BATCH" --tool-calling "$TRIAL/tool_calling.json" \
  --compaction compaction.json --output current

uv run --project evals --locked python -m evals.report compare \
  baseline.json current.json --output comparison
```

Component results retain each Trial or sample's own coverage. The official
resolved rate uses scored Trials as its denominator. Cost estimates use the
saved token counts and Task 5's USD price snapshot at peak rates by default.
Use `--off-peak` when all included requests qualify. Agent run totals include
compaction. Judge usage is reported separately. Component reports for the same
run are counted once, using the first report supplied for that component.
Compaction captures need a run ID. Missing measurements or comparison metadata
are shown as N/A.
