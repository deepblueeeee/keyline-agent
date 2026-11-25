# Handshake evaluations

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
  --instance-id sympy__sympy-20590 --trials 1 --output-dir .handshake/evals
```

Repeat `--instance-id` or select `--suite regression` or `--suite capability`.
Set `--max-requests`, `--max-steps` and `--agent-seconds` to control each
Trial's budget. The run prints the path to its `summary.json`.

## Score components

Set `TRIAL` to the directory of a completed Trial. Review the data before
sending it for Judge scoring.

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
