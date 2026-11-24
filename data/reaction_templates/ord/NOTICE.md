# ORD reaction-template source (203 templates)

The 203 templates in this directory, including the SMARTS and the names,
descriptions, planning notes and provenance records stored with them in
`templates.jsonl`, are licensed under the Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/legalcode. See [LICENSE](LICENSE).
This license covers only this directory. The rest of BioNeMo-SynClaw is under the project
[LICENSE](../../../LICENSE) (Apache-2.0) or the terms in the other NOTICE files.

## Source

The reactions were obtained through the Open Reaction Database (ORD),
https://open-reaction-database.org (Kearnes et al., J. Am. Chem. Soc. 2021), which
distributes them under CC BY-SA 4.0. They come from ORD's `uspto-grants` dataset,
ORD's import of:

Daniel Lowe, Chemical reactions from US patents (1976-Sep2016), figshare (2017),
https://doi.org/10.6084/m9.figshare.5104873.v1.

## Changes made in this repository

- Extracted a radius-1 template with reaction-utils 1.9.3 from each of 203
  selected atom-mapped reactions, and removed duplicates.
- Repaired the SMARTS of seven templates. The extracted strings are kept in
  `metadata.smarts_revision`; see [SMARTS_REPAIRS.md](../SMARTS_REPAIRS.md).
- Added names, families, descriptions, search terms, role queries, execution
  caveats and planning notes. These are local interpretations of the SMARTS, not
  upstream ORD labels and not ground truth.
- Added a provenance record to each template:
  `{"source_id": "uspto-grants-lowe-2017", "source_record_id": "<patent>:<line>"}`,
  where `<line>` is the 1-based line of Lowe's file
  `1976_Sep2016_USPTOgrants_smiles.rsmi`, counting the header as line 1 (SHA-256 of
  the distributed `1976_Sep2016_USPTOgrants_smiles.7z`:
  d10f8ecb013d3f7bc85abdf6a8c5da8a7dd992931c1fda5db14fbd764343bc4e). Running the
  same extraction on that file reproduces every template as originally extracted,
  before the seven repairs.

These changes are not endorsed by ORD or the original authors.
