# NVIDIA-authored reaction templates (8 templates)

Generic templates for common transformations, written at NVIDIA rather than
extracted from reaction records. Each record's `metadata.provenance` starts with
`{"source_id": "nvidia-authored"}`. The remaining entries, marked
`"relation": "example"`, cite one to five recorded reactions in Lowe's USPTO grants
file that the template reproduces: for each main reaction type, the first one in the
file whose recorded yield is at least 70% and whose atom-contributing reactants are
exactly the template's. They illustrate each template; none was derived from them.

Use of the templates and their annotations is governed by the OpenMDW License
Agreement, version 1.1; see [../LICENSE](../LICENSE).

The table compares each template with the closest published template and
summarizes its scope. Behavior was checked on reactions from Daniel Lowe's USPTO
grants data (https://doi.org/10.6084/m9.figshare.5104873.v1). Published
references: Hartenfeller et al., J. Chem. Inf. Model. 51, 3093 (2011); Button et
al., Nat. Mach. Intell. 1, 307 (2019), both as distributed with SynNet; and the
SynFormer templates in [../synformer/](../synformer/NOTICE.md).

| ID | Name | Closest published template | Scope |
| --- | --- | --- | --- |
| `nvidia:R000` | General amine acylation to an amide | Button amide coupling; `synformer:R008` | Acids, esters, acyl chlorides, chloroformates, carbonates, anhydrides, thioesters and imide-type donors such as N-Boc amides. Plain amides, ureas, hydrazides and ring-opening donors are excluded. |
| `nvidia:R001` | N-substitution of an aromatic primary or secondary amine | `synformer:R009` (amine and organic halide) | Chloride, bromide, iodide, sulfonate, ester and thio leaving groups on aromatic or aliphatic carbon; ring leaving atoms are excluded. The O and S alternatives need case-specific assessment. |
| `nvidia:R002` | O-linked carboxyl derivative reduction to alcohol | Patent-extracted acid and ester reductions | Acids, esters and anhydrides; lactones are excluded. |
| `nvidia:R003` | Mitsunobu substitution of a primary or secondary alcohol with inversion | Hartenfeller Mitsunobu reactions | Primary or secondary alcohol with an acidic O, N or S pronucleophile (acids, phenols, N–OH, imides, sulfonamides, HN₃, N-H heterocycles such as tetrazoles, triazoles, imidazoles, purines and pyridones, thiols and thioacids), inverting the alcohol carbon. Reproduces 97% of the 3,123 alcohol substitutions run with Mitsunobu reagents in Lowe's USPTO data. Only the ring nitrogen drawn with the hydrogen reacts, and reverse simulation does not propose N-H heterocycle precursors. |
| `nvidia:R004` | Ether formation from an oxygen donor and carbon halide | Hartenfeller Williamson ether; `synformer:R056` | Alcohols, phenols and carboxylic acids with organic chlorides, bromides or iodides, including aryl, vinyl and acyl halides. Aryl halides need activation or a copper or palladium catalyst that the rule does not encode. |
| `nvidia:R005` | Carbonyl olefination with a phosphorus-bound carbon | Hartenfeller Wittig; `synformer:R102` | Aldehydes and ketones with phosphonium, phosphonate or phosphine-oxide carbons, as patents record Wittig and Horner-Wadsworth-Emmons reactions. E/Z geometry is not encoded. |
| `nvidia:R006` | Broad chloride or bromide replacement by a heteroatom donor | `synformer:R009`, `synformer:R056`, `synformer:R031` | General substitution, broader than the published templates; the donor atom is neutral in the product. |
| `nvidia:R007` | N-carbamate cleavage to the parent nitrogen | Patent-extracted Boc removal, including `ord:R031` | Boc and other acyclic carbamates; cyclic carbamates are excluded. |
