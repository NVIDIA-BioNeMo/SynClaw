# SMARTS repairs

Eighteen imported SMARTS are revised for chemical correctness; the other 483 are
unchanged. Each revised record keeps `metadata.smarts_revision` with the imported
string (`original_smarts`), a one-line `reason`, and stable `sources` identifiers
(patent numbers, ORD reaction IDs, PaRoutes reaction hashes, DOIs) for the recorded
procedure or example that supports the corrected rule.

**Origin** is where the imported rule came from. SynFormer's 115 rules implement the
January 2022 Enamine REAL reaction set ([Grygorenko et al. 2020](https://doi.org/10.1016/j.isci.2020.101681)),
except 29 selected from the [Hartenfeller](https://doi.org/10.1021/ci200379p) and
[Button](https://doi.org/10.1038/s42256-019-0067-7) collections via
[SynNet `hb.txt`](https://github.com/wenhao-gao/SynNet/blob/3ccf4fcd6eb0605cb60f6376ad7b4a7accaf1549/data/assets/reaction-templates/hb.txt);
SynFormer does not list which rule came from which set, so attribution below is by
matching the SMARTS. **Evidence** is a recorded example of the corrected reaction.
Encoding fixes marked “—” need none: the defect and its correction are visible in the
SMARTS.

| Template | Change | Origin | Evidence |
| --- | --- | --- | --- |
| `synformer:R031` | Neutral product ester oxygen for carboxylate inputs | [SynFormer L32](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L32), Enamine REAL | [US6391885B1](https://patents.google.com/patent/US6391885B1/en) |
| `synformer:R035` | Inserted quinazoline C2 carries one hydrogen (was a radical) | [SynFormer L36](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L36), Enamine REAL | [PaRoutes](https://doi.org/10.5281/zenodo.7341155) record with DMF-DMA as the C2 source |
| `synformer:R041` | Unchanged thiourea nitrogen keeps its hydrogen count (was a radical) | [SynFormer L42](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L42), Enamine REAL | — |
| `synformer:R051` | Hydrogen atom query uses `#1`; acid oxygen remains an oxygen in reverse | [SynFormer L52](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L52) | — |
| `synformer:R062` | Hydrogen atom query uses `#1`, avoiding H1 heteroatom matches and reverse radicals | [SynFormer L63](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L63) | — |
| `synformer:R065` | `[nHo+0]` (holmium) corrected to `[nH0+0]` | [SynFormer L66](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L66), Enamine REAL | — |
| `synformer:R070` | N-substituent carried over unchanged; reactant hydrogen atom query uses `#1` to avoid reverse radicals | [SynFormer L71](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L71), Enamine REAL | — |
| `synformer:R077` | Hydrogen atom query uses `#1`, avoiding H1 heteroatom matches and reverse radicals | [SynFormer L78](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L78) | — |
| `synformer:R090` | Neutral pyrrole nitrogen for ammonium-salt inputs | [SynFormer L91](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L91), adapted from Hartenfeller’s Paal–Knorr ([`hb.txt` L63](https://github.com/wenhao-gao/SynNet/blob/3ccf4fcd6eb0605cb60f6376ad7b4a7accaf1549/data/assets/reaction-templates/hb.txt#L63)) | [US5972975A](https://patents.google.com/patent/US5972975A/en) |
| `synformer:R105` | Exclude only hydroxyl (`!$([OH1])`), admitting ortho acyl and nitro groups | [SynFormer L106](https://github.com/wenhao-gao/synformer/blob/bef02bc4481adf3ed792090fc6169331d4d0553a/data/rxn_templates/comprehensive.txt#L106) = [`hb.txt` L44](https://github.com/wenhao-gao/SynNet/blob/3ccf4fcd6eb0605cb60f6376ad7b4a7accaf1549/data/assets/reaction-templates/hb.txt#L44), which already contains `!OH1` | [Org. Synth. 85, 196](https://doi.org/10.15227/orgsyn.085.0196); [Gooßen et al. 2006](https://doi.org/10.1126/science.1128684) |
| `paroutes:PRECF08F5BBF5F` | Claisen allyl atom correspondence (labels only) | [PaRoutes](https://doi.org/10.5281/zenodo.7341155) | [US20160272611A1](https://patents.google.com/patent/US20160272611A1/en) |
| `ord:R053` | Stille vinyl carbons from the tin vinyl ligand | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 834621 | [US7998968B2](https://patents.google.com/patent/US7998968B2/en), [ORD](https://open-reaction-database.org/id/ord-1c7bbb37542f4442864c225989c21285) |
| `ord:R065` | Both pyrazole nitrogens from the hydrazine | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 1273572 | [US8778944B2](https://patents.google.com/patent/US8778944B2/en), [ORD](https://open-reaction-database.org/id/ord-0469be35018c46d1877518211c6ceb86) |
| `ord:R173` | Nitromethane required as the carbon donor | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 967782 | [US7504392B2](https://patents.google.com/patent/US7504392B2/en), [ORD](https://open-reaction-database.org/id/ord-7468f6f5425a4ab6ada581796ad9d8dc), [PaRoutes](https://doi.org/10.5281/zenodo.7341155) |
| `ord:R194` | Wittig methylene from methyltriphenylphosphonium | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 383273 | [US6518431B2](https://patents.google.com/patent/US6518431B2/en), [ORD](https://open-reaction-database.org/id/ord-0c74ab08b43b4ea88e6acef0f57747f4) |
| `ord:R199` | Methyl from methyl iodide; oxazoline directing group required | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 280485 | [US6162808A](https://patents.google.com/patent/US6162808A/en), [ORD](https://open-reaction-database.org/id/ord-010ea32c8a874abea4931b1b6381ba7d) |
| `ord:R200` | Both ethyls from ethylmagnesium bromide; OH from the carbonyl | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 931810 | [US7601850B2](https://patents.google.com/patent/US7601850B2/en), [ORD](https://open-reaction-database.org/id/ord-242830405a974cde80be3d03316dbf31) |
| `ord:R202` | N-methyl from formaldehyde; secondary dialkylamines only | ORD `uspto-grants`; [Lowe USPTO](https://doi.org/10.6084/m9.figshare.5104873.v1) line 1299528 | [US7122538B2](https://patents.google.com/patent/US7122538B2/en), [ORD](https://open-reaction-database.org/id/ord-16bd1f3da68643d1b29743bd5cd88952) |

Methyl iodide, formaldehyde, nitromethane and ethylmagnesium bromide are required
reactants of the revised ORD rules; stocks without them cannot ground those steps.
