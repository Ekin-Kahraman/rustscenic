# Agentic fresh-environment validation

This harness turns "agentic testing" into reproducible evidence instead of
simulated user activity. Each scenario represents a tester persona and a
concrete workflow. The runner executes that workflow in a clean Docker
container, captures logs, wall time, peak RSS, the git SHA tested, and any
dirty-worktree warning, then writes a structured JSON report.

Use it for:

- fresh install and import checks from a source checkout
- collaborator-facing examples
- no-network smoke tests
- guardrail/error-message checks that mimic confused first-time users
- optional heavier validation runs before a release

Do not use it for fake GitHub users, artificial praise, or biological claims.
Agent reports are engineering evidence only. Real biological validation still
needs real datasets, commands, hashes, limitations, and preferably external
human review.

## Run locally

List available scenarios:

```bash
python validation/agentic/run_agentic_scenarios.py --list
```

Run one scenario:

```bash
python validation/agentic/run_agentic_scenarios.py \
  --scenario fresh_install_core
```

Run the smoke set:

```bash
python validation/agentic/run_agentic_scenarios.py \
  --scenario fresh_install_core \
  --scenario portable_preproc_smoke \
  --scenario bad_input_guardrails
```

Reports and logs are written to `validation/agentic/reports/` by default.
Generated reports are ignored by git so local runs do not pollute normal
development. Promote only curated, meaningful reports into committed validation
artefacts.

## What a report proves

A passing report proves that the named scenario completed from the archived git
`HEAD` in a fresh Linux container. It does not prove that every workflow is
production ready, and it does not replace the normal unit, integration, nightly
real-data, or release workflows.

The runner intentionally archives `HEAD` rather than copying a dirty worktree.
If you have uncommitted changes, the report will say so, but the container tests
the committed tree. Commit first when producing evidence for a release or public
claim.

## Scenario contract

Scenario files live in `validation/agentic/scenarios/*.json` and follow
`validation/agentic/report_schema.json` for emitted reports. A scenario contains:

- `id`: stable machine-readable name
- `title`: short human-readable label
- `agent_persona`: the tester viewpoint being simulated
- `goal`: the evidence the run should produce
- `base_image`: Docker base image, normally `rust:<version>-bookworm`
- `timeout_seconds`: hard timeout for the container run
- `tags`: searchable labels such as `smoke`, `no-network`, or `examples`
- `commands`: ordered shell commands executed inside `/repo`

Keep scenarios small and honest. Prefer multiple narrow scenarios over one
large report that is hard to interpret.

