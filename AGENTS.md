# Repository Guidelines

## Project Structure

BioNeMo-SynClaw (Agentic Retrosynthesis Planning) is a Python package under `src/synclaw`. The CLI
entry point is `src/synclaw/cli.py`, which stays at the package root alongside
`__main__.py`. Everything else is grouped by layer:

| package | holds |
|---|---|
| `synclaw/core/` | `chem`, `db`, `reactions`, `libraries`, `fingerprint_index` — the RDKit and SQLite foundation |
| `synclaw/tools/` | `contracts` (schema), `synthesis` (implementation), `compaction` |
| `synclaw/rollout/` | `loop`, `policy`, `prompts`, `budget` — the agent loop |
| `synclaw/runners/` | `model_runner` (hosted models), `mcp_server` |
| `synclaw/evaluation/` | `evaluate`, `results` |
| `synclaw/tasks/` | `datasets`, `emolecules`, `matrix`, `public_data`, `targets` |
| `synclaw/paths.py`, `synclaw/providers.py`, `synclaw/token_usage.py` | cross-cutting leaves |

Provider presets (API style, endpoints, credential variables) live in
`src/synclaw/providers.json`, not in Python. Add or repoint a backend by editing that
table. The API style comes from the preset or `--api-mode`, never from the model
name, so do not add model-name special cases. `token_usage.py` counts tokens and
passes through a provider-reported `cost`; BioNeMo-SynClaw never estimates prices.

**Adding a model-facing tool** touches three files in order: declare the schema in
`synclaw/tools/contracts.py`, implement it on `SynthesisTools` in `synclaw/tools/synthesis.py`,
and add the dispatch branch in `synclaw/rollout/loop.py`'s `call_local_tool`. The first two
are deliberately adjacent.

Two rules hold this together. Every subpackage `__init__.py` is a docstring and
nothing else, so importing a chemistry-free module such as `tools.contracts` does
not pull in RDKit. And repository-relative data paths resolve through `synclaw.paths`,
never a local `__file__` walk. `tests/test_package_layout.py` enforces both.

`scripts/` holds standalone helpers; `scripts/render_tool_trace.py` renders a trace
as Markdown. Generated tasksets, traces, reports, result CSVs and local SQLite
databases belong under the ignored `artifacts/` and `var/` directories.

## Data Boundaries

Do not commit licensed or generated chemistry data. The exceptions are the reaction
templates, annotations and attribution files under `data/reaction_templates/`, their
`src/synclaw/reaction_libraries.json` manifest, and the curated common-reagent building
blocks. Store each SMARTS in one source only and select unions at runtime. Keep every
license and provenance notice, including the per-template ORD provenance.

Building-block catalogs are prepared by users (see the README).
Public SynFormer templates and target sets are fetched with
`uv run synclaw download-public-data`. Keep new examples and tests fixture-sized.

## Setup And Checks

```bash
uv sync --extra dev            # add --extra mcp for the MCP server
uv run ruff check .
uv run pytest
```

Ruff also enforces the SPDX license header (`CPY001`) on every Python file. See the
README for building a local database and running models.

## Tool And Validation Semantics

`submit_pathway` is intentionally targetless from the model's perspective. The
harness already knows the task target, computes the final product from the
RDKit-validated final reaction step, and scores the product after the session.
Do not reintroduce model-controlled `target_smiles` or `final_smiles` fields in
the public submission schema.

`simulate_reaction` is permissive scratch space and can accept arbitrary SMILES.
It returns `generated_product_sets` for multi-product templates while preserving
the flat `generated_products` list for scoring. `simulate_reverse_reaction` is
also planning-only: generated precursors can be intermediates or starting
materials, and building-block matches are advisory evidence rather than
validation gates. Multi-product reverse simulation requires all product
components. Keep reverse-simulation responses concise: surface exact BB
grounding, cap near-match lists, and make ungrounded precursors clearly
intermediate-only. `search_reaction_templates` supports text search and
structural SMARTS-role matching from supplied reactant/product SMILES in the
same public tool; keep library selection and reverse-ranking policy
harness-controlled rather than model-controlled. `submit_pathway` is strict:
starting reactants must be grounded in the local building-block database by
`BB:<id>` or exact building-block SMILES, and intermediates must use
`INT:<step_number>`. Invalid tool calls should return structured, retryable
errors so the model can repair within the turn budget.

Raw TXT reaction-template imports use zero-based SynFormer-style IDs. For
example, the 45th SMARTS line imports as `synformer:R044`.

RDKit is required. The package should fail loudly if RDKit is unavailable; do
not add fallback chemistry validation that could make invalid routes look valid.

## Coding Style

Support Python 3.11–3.14 (`requires-python`, ruff `target-version = "py311"`), so avoid
syntax newer than 3.11. Use four-space indentation, type annotations for public
interfaces, and focused dataclasses for structured configuration. Ruff checks `E`, `F`,
`I`, `UP` and `CPY001` (every Python file starts with the NVIDIA Apache-2.0 SPDX header)
with a 100-character line length; `E501` is ignored for long chemistry strings and
provider payloads.

Prefer existing module boundaries over new abstractions. Keep MCP/tool schemas,
CLI behavior and evaluator expectations aligned when changing tool arguments or
pathway-validation output. Prefer explicit configuration with clear validation
errors; avoid silent fallbacks and compatibility shims.

## Testing

Tests are named `test_*.py` and live in the `tests/` subdirectory mirroring the
package they exercise. Keep basenames unique across subdirectories, because pytest
imports test modules by basename. Tests must not require network access, API keys,
licensed building-block data or persistent local state; use temporary directories and
fixture databases. Add regression tests for changes to provider handling, tool-call
parsing and repair, building-block grounding, reaction-template storage and search,
RDKit reaction execution, and trace metrics.

## Contributing

Use short imperative commit subjects, for example `Add reverse-simulation caps`, and
sign off every commit (`git commit -s`; see [CONTRIBUTING.md](CONTRIBUTING.md)). Keep
commits cohesive and never commit model-run artifacts. Pull requests should summarize
the behavior change, list the validation commands run, and note any data or API-key
assumptions.
