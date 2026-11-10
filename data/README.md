# Data Inputs

Building-block catalogs and downloaded corpora remain untracked. The curated
source catalogs under `reaction_templates/{synformer,paroutes,ord,nvidia}/` are tracked
with inline annotations and source notices. Install them with
`synclaw ingest-reactions --db <path>`; the default union contains 501 unique rules.
See [the catalog README](reaction_templates/README.md) for source composition and
attribution. Search and prompts use the annotations stored in the database; raw-text
imports have none.

Download the public SynFormer assets with:

```bash
uv run synclaw download-public-data
```

For building-block catalogs, see the "Building-Block Catalogs" section of the top-level
[README](../README.md).
