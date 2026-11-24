---
metadata:
  domain: science
  modality: [text]
  task_categories: [other]
  source: blend
  size_bin: "n<1K"
  license: "OpenMDW-1.1; Open Reaction Database data CC BY-SA 4.0; see License/Terms of Use."
---

# BioNeMo-SynClaw Reaction Templates v0.1.0

## Dataset Description:

BioNeMo-SynClaw Reaction Templates v0.1.0 is a set of 501 curated, annotated chemical reaction templates (reaction SMARTS) that ship with [BioNeMo-SynClaw](https://github.com/NVIDIA-BioNeMo/SynClaw), NVIDIA's open-source harness for agentic retrosynthesis planning. When a language model plans a synthesis route to a target molecule, BioNeMo-SynClaw's tools use these templates to search for applicable reactions and to simulate and validate each reaction step with [RDKit](https://github.com/rdkit/rdkit).

Each template is one JSONL record containing an executable reaction SMARTS, a name, a reaction family, a description, search terms, reactant role queries, execution caveats or planning notes where applicable, and a provenance record naming its source. A separate file describes each reaction family.

The templates come from four sources: 114 from [SynFormer](https://github.com/wenhao-gao/synformer), 176 from [PaRoutes 2.0](https://doi.org/10.5281/zenodo.7341155), 203 extracted from Daniel Lowe's US patent reactions as distributed by the [Open Reaction Database](https://open-reaction-database.org) (ORD), and 8 written at NVIDIA. NVIDIA selected, deduplicated, inverted (PaRoutes) or extracted (ORD) the source templates, repaired 18 SMARTS while keeping the original strings, and wrote the annotations. The changes for each source are documented with the templates.

The templates are not released as a standalone dataset. They are distributed with the BioNeMo-SynClaw source code under `data/reaction_templates/` in the GitHub repository.

These are computational planning templates. Executing a template in RDKit does not establish experimental feasibility, reaction conditions, yield or selectivity.

This dataset is ready for commercial or non-commercial use.

## Dataset Owner(s):

NVIDIA Corporation

## Dataset Creation Date:

Nov 23, 2026

## Version:

V0.1.0

Previous Version(s): None

## License/Terms of Use:

Governing Terms: Use of the data is governed by [OpenMDW License Agreement, version 1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1), except for the Open Reaction Database data, which is governed by the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/legalcode). Additional Information: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) for the SynFormer data and [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) for the PaRoutes data.

Attribution, source licenses and a description of NVIDIA's changes are in the NOTICE.md file for each source, [LICENSE](LICENSE), [ord/LICENSE](ord/LICENSE) and [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) in the repository.

## Intended Usage:

To be used with BioNeMo-SynClaw for LLM-assisted retrosynthesis planning: searching, simulating and validating reaction steps in multi-step synthesis routes, including evaluation and reinforcement-learning environments in [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym). Not intended as a source of experimental procedures.

## Dataset Characterization

**Data Collection Method**

* Hybrid: Automated, Human

**Labeling Method**

* Hybrid: Human, Synthetic, Automated

## Dataset Format

Text only; JSONL (one template per line) and JSON (reaction-family descriptions).

## Dataset Quantification

Record Count: 501 reaction templates (SynFormer 114, PaRoutes 176, USPTO/ORD-derived 203, NVIDIA-authored 8) and 137 reaction-family descriptions.

Feature Count: each template has a reaction SMARTS plus name, family, description, search terms, reactant roles and provenance; some also carry execution caveats, planning notes and a record of SMARTS repairs.

Total Data Storage: 0.7 MB

## Reference(s):

- [BioNeMo-SynClaw](https://github.com/NVIDIA-BioNeMo/SynClaw)
- Gao, Luo and Coley, Generative AI for Navigating Synthesizable Chemical Space: [SynFormer](https://github.com/wenhao-gao/synformer)
- Genheden and Bjerrum, PaRoutes: towards a framework for benchmarking retrosynthesis route predictions, Digital Discovery (2022): [doi:10.1039/D2DD00015F](https://doi.org/10.1039/D2DD00015F); PaRoutes 2.0 dataset: [doi:10.5281/zenodo.7341155](https://doi.org/10.5281/zenodo.7341155)
- Kearnes et al., The Open Reaction Database, J. Am. Chem. Soc. (2021): [doi:10.1021/jacs.1c09820](https://doi.org/10.1021/jacs.1c09820)
- Lowe, Chemical reactions from US patents (1976-Sep2016), figshare (2017): [doi:10.6084/m9.figshare.5104873.v1](https://doi.org/10.6084/m9.figshare.5104873.v1)

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
