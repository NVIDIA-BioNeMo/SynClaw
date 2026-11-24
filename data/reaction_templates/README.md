# Reaction source libraries

Each executable SMARTS and its inline annotations are stored in one JSONL record;
`metadata.provenance` records where it came from, and `metadata.smarts_revision` any
local repair ([SMARTS_REPAIRS.md](SMARTS_REPAIRS.md)). Each reaction family's
description and search terms are stored once in `families.json` and added to every
template of that family on import.
[`src/synclaw/reaction_libraries.json`](../../src/synclaw/reaction_libraries.json) lists the
source files, their template counts, and checksums. The manifest ships in the wheel; the
template files themselves are read from a source checkout.

| Source | Templates |
| --- | ---: |
| SynFormer | 114 |
| PaRoutes | 176 |
| ORD | 203 |
| NVIDIA-authored (`nvidia`) | 8 |

No SMARTS appears in more than one source.

Import from a source checkout into a new or existing stock database:

```bash
uv run synclaw ingest-reactions --db var/synclaw.sqlite
```

This installs all 501 templates and preserves
building-block rows and prepared stock indexes. Reimporting replaces every row
stored under `synformer`, `paroutes`, `ord`, or `nvidia`. Other libraries already in the
database are retained and available alongside them. The import records each source
file's hash from that manifest, and model runs reject a database whose bundled
source file, or `families.json`, has changed since, until you reimport.

The tools use every template in the database. Build a smaller union by removing
sources, e.g. `synclaw db remove-reactions paroutes nvidia --db <path>`.

| Sources in the database | Unique templates |
| --- | ---: |
| `synformer` | 114 |
| `paroutes` | 176 |
| `ord` | 203 |
| `nvidia` | 8 |
| `synformer paroutes` | 290 |
| `synformer ord` | 317 |
| `synformer paroutes ord` | 493 |
| `synformer paroutes ord nvidia` | 501 |

In all four sources, the template annotations (names, families, descriptions, search terms,
role queries and caveats) are interpretations of the SMARTS, not ground truth. They are
hybrid human- and LLM-generated.

SMARTS are unchanged from their sources except 18 repaired rules (10 SynFormer, 1 PaRoutes, 7 ORD), listed in
[SMARTS_REPAIRS.md](SMARTS_REPAIRS.md) with their imported strings kept in
`metadata.smarts_revision`. Prompt hints are generated from the imported records.

Source files name templates `synformer:R...`, `paroutes:PR...`, `ord:R...`, and `nvidia:R...`.
SynFormer IDs keep the original zero-based line numbers, so `synformer:R033` (a duplicate of
R032) does not exist. Import renumbers every template in the database `R001` onward in
SMARTS-hash order and keeps the source ID as `metadata.source_id`; tools see only the numbered
ID. Every ORD record's `metadata.provenance` names its source patent and line in Lowe's USPTO
grants file.

Attribution and licenses are in [synformer/NOTICE.md](synformer/NOTICE.md),
[paroutes/NOTICE.md](paroutes/NOTICE.md), [ord/NOTICE.md](ord/NOTICE.md), and
[nvidia/NOTICE.md](nvidia/NOTICE.md). Use of the data in this directory is governed by
the OpenMDW License Agreement, version 1.1 ([LICENSE](LICENSE)), except for the Open
Reaction Database data in `ord/`, which is governed by CC BY-SA 4.0; see
[ord/LICENSE](ord/LICENSE) and [ord/NOTICE.md](ord/NOTICE.md). The ORD templates,
including their annotations, come from Lowe's USPTO grants data as distributed by the
Open Reaction Database. Additional information: the SynFormer templates are under the
Apache License 2.0 and the PaRoutes templates under CC BY 4.0 from their sources.
The BioNeMo-SynClaw code is licensed separately under the Apache License 2.0 in the
project [LICENSE](../../LICENSE). [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
summarizes every source. These are computational planning templates;
RDKit execution and stock grounding do not establish experimental feasibility.
