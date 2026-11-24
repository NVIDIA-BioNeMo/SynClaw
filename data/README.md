# Data Inputs

Downloaded corpora and large building-block catalogs are untracked; only the small
`building_blocks/common_reagent_building_blocks.txt` is tracked. The curated
source catalogs under `reaction_templates/{synformer,paroutes,ord,nvidia}/` are tracked
with inline annotations and source notices. Install them with
`synclaw ingest-reactions --db <path>`; the default union contains 501 unique rules.
See [the catalog README](reaction_templates/README.md) for source composition and
attribution. Search and prompts use the annotations stored in the database; raw-text
imports have none.

Download the public target lists (the SynFormer ChEMBL 1k set and the ReaSyn watchlist
`mols_to_filter`) with:

```bash
uv run synclaw download-public-data
```

`--name` (repeatable) downloads only the named files: `synformer-chembl-1k` (the ChEMBL
targets for `make-taskset`) or `reasyn-mols-to-filter` (the watchlist for the optional
toxicity screen).

For building-block catalogs, see the "Building-Block Catalogs" section of the top-level
[README](../README.md).
