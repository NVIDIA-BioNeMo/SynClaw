# Third-Party Notices

BioNeMo-SynClaw is licensed under the [Apache License 2.0](LICENSE), except the reaction
templates in `data/reaction_templates/ord/`, which are licensed under CC BY-SA 4.0.
This file lists the third-party material distributed in this repository and the
third-party packages BioNeMo-SynClaw depends on. No third-party source code is vendored.

## Data distributed in this repository

### SynFormer reaction templates

- Path: [`data/reaction_templates/synformer/`](data/reaction_templates/synformer/NOTICE.md)
- Source: Wenhao Gao, Shitong Luo, and Connor W. Coley, *Generative AI for
  Navigating Synthesizable Chemical Space*.
  <https://github.com/wenhao-gao/synformer>, commit
  `bef02bc4481adf3ed792090fc6169331d4d0553a`.
- License: Apache License 2.0. The license terms are the Apache 2.0 terms in
  this project's [LICENSE](LICENSE). The upstream repository does not
  ship a NOTICE file.
- Changes: one duplicate SMARTS is not stored, local SMARTS repairs are listed in
  [`SMARTS_REPAIRS.md`](data/reaction_templates/SMARTS_REPAIRS.md), and names,
  roles, search terms, and descriptions are locally authored annotations.

### PaRoutes reaction templates

- Path: [`data/reaction_templates/paroutes/`](data/reaction_templates/paroutes/NOTICE.md)
- Source: Samuel Genheden and Esben Bjerrum, *PaRoutes: towards a framework for
  benchmarking retrosynthesis route predictions*, Digital Discovery (2022),
  <https://doi.org/10.1039/D2DD00015F>. PaRoutes 2.0 dataset, Samuel Genheden,
  <https://doi.org/10.5281/zenodo.7341155>.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  <https://creativecommons.org/licenses/by/4.0/legalcode>.
- Changes: retrosynthetic templates were selected and inverted into forward
  rules, local SMARTS repairs are listed in
  [`SMARTS_REPAIRS.md`](data/reaction_templates/SMARTS_REPAIRS.md), and names,
  roles, search terms, and caveats are locally authored annotations. These
  changes are not endorsed by the original authors.

### ORD reaction templates

- Path: [`data/reaction_templates/ord/`](data/reaction_templates/ord/NOTICE.md)
- Source: 203 templates extracted from reactions obtained through the Open
  Reaction Database (ORD), <https://open-reaction-database.org> (Kearnes et al.,
  J. Am. Chem. Soc. 2021), from its import of Daniel Lowe, *Chemical reactions
  from US patents (1976-Sep2016)*, figshare (2017),
  <https://doi.org/10.6084/m9.figshare.5104873.v1>. Each record names the patent
  and line in Lowe's file that reproduces it. NVIDIA-authored templates are in
  [`data/reaction_templates/nvidia/`](data/reaction_templates/nvidia/NOTICE.md)
  and are not third-party material.
- License: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0), <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.
  It covers the templates and the annotations stored with them in that
  directory (see its [LICENSE](data/reaction_templates/ord/LICENSE)), and does not
  extend to the rest of BioNeMo-SynClaw.
- Changes: radius-1 templates were extracted and deduplicated, seven SMARTS were
  repaired (listed in
  [`SMARTS_REPAIRS.md`](data/reaction_templates/SMARTS_REPAIRS.md)), and names,
  descriptions, planning notes, and provenance records were added; see the
  [ORD notice](data/reaction_templates/ord/NOTICE.md). These changes are not
  endorsed by ORD or the original authors.

## Runtime dependencies

These packages are installed separately by the package manager under their own
licenses. They are not distributed with BioNeMo-SynClaw.

| Package | Used by | License |
| --- | --- | --- |
| [NumPy](https://github.com/numpy/numpy) | core | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [RDKit](https://github.com/rdkit/rdkit) | core | BSD-3-Clause |
| [mcp](https://github.com/modelcontextprotocol/python-sdk) | `mcp` extra | MIT |
| [packaging](https://github.com/pypa/packaging) | `dev` extra | Apache-2.0 OR BSD-2-Clause |
| [pytest](https://github.com/pytest-dev/pytest) | `dev` extra | MIT |
| [ruff](https://github.com/astral-sh/ruff) | `dev` extra | MIT |

## Data downloaded on demand

`synclaw download-public-data` fetches SynFormer files and the ReaSyn watchlist
(`data/mols_to_filter.txt`) from their respective public repositories, and
`synclaw prepare-emolecules-bbs` downloads the eMolecules catalog from
downloads.emolecules.com. These files are not distributed with BioNeMo-SynClaw and remain under their
providers' terms.
