# MasakhaNER 2.0 inventory

Retrieved 2026-10-09 from pinned metadata only. No split file was downloaded and no model was run. This page does not report any benchmark measurement.

## Pins

- Hugging Face dataset `masakhane/masakhaner2` at `60512e89e68841b6b5ed1be59caf97b169f0d27a`.
- GitHub repository `masakhane-io/masakhane-ner` at `ba5843cd08aa491d5f96a5e809e71eb9ec461391` (2025-10-15T20:37:20Z).
- Data root `MasakhaNER2.0/data`, with one folder per configuration.

## License

Status: conflicting: the dataset license is not resolved.

| Source | Value |
| --- | --- |
| Hugging Face dataset metadata (license tag) | afl-3.0 |
| Hugging Face dataset card, Licensing Information | CC 4.0 Non-Commercial |
| GitHub MasakhaNER2.0/README.md | CC-BY-4.0-NC |
| GitHub repository LICENSE (code) | Apache 2.0 for the code derived from Hugging Face transformers |

## Coverage of the 85 canonical targets

- Covered by MasakhaNER 2.0: 5 (ha, ig, xh, yo, zu).
- Missing: 80 of 85.
- WikiANN test split lacks: ha, xh, zu. MasakhaNER 2.0 supplies: ha, xh, zu.
- Covered examples across train, validation and test: 31793, 4542 and 9081.

| Code | Status | MasakhaNER config | WikiANN test split | Train | Validation | Test |
| --- | --- | --- | --- | ---: | ---: | ---: |
| af | missing | - | present | - | - | - |
| am | missing | - | present | - | - | - |
| ar | missing | - | present | - | - | - |
| az | missing | - | present | - | - | - |
| be | missing | - | present | - | - | - |
| bg | missing | - | present | - | - | - |
| bn | missing | - | present | - | - | - |
| ca | missing | - | present | - | - | - |
| ceb | missing | - | present | - | - | - |
| cs | missing | - | present | - | - | - |
| cy | missing | - | present | - | - | - |
| da | missing | - | present | - | - | - |
| de | missing | - | present | - | - | - |
| el | missing | - | present | - | - | - |
| en | missing | - | present | - | - | - |
| eo | missing | - | present | - | - | - |
| es | missing | - | present | - | - | - |
| et | missing | - | present | - | - | - |
| eu | missing | - | present | - | - | - |
| fa | missing | - | present | - | - | - |
| fi | missing | - | present | - | - | - |
| fr | missing | - | present | - | - | - |
| fy | missing | - | present | - | - | - |
| ga | missing | - | present | - | - | - |
| gd | missing | - | present | - | - | - |
| gl | missing | - | present | - | - | - |
| gu | missing | - | present | - | - | - |
| ha | covered | hau | missing | 5716 | 816 | 1633 |
| he | missing | - | present | - | - | - |
| hi | missing | - | present | - | - | - |
| hu | missing | - | present | - | - | - |
| hy | missing | - | present | - | - | - |
| id | missing | - | present | - | - | - |
| ig | covered | ibo | present | 7634 | 1090 | 2181 |
| is | missing | - | present | - | - | - |
| it | missing | - | present | - | - | - |
| ja | missing | - | present | - | - | - |
| jv | missing | - | present | - | - | - |
| ka | missing | - | present | - | - | - |
| kk | missing | - | present | - | - | - |
| km | missing | - | present | - | - | - |
| kn | missing | - | present | - | - | - |
| ko | missing | - | present | - | - | - |
| ku | missing | - | present | - | - | - |
| ky | missing | - | present | - | - | - |
| la | missing | - | present | - | - | - |
| lt | missing | - | present | - | - | - |
| lv | missing | - | present | - | - | - |
| mg | missing | - | present | - | - | - |
| mk | missing | - | present | - | - | - |
| ml | missing | - | present | - | - | - |
| mn | missing | - | present | - | - | - |
| mr | missing | - | present | - | - | - |
| ms | missing | - | present | - | - | - |
| mt | missing | - | present | - | - | - |
| my | missing | - | present | - | - | - |
| ne | missing | - | present | - | - | - |
| nl | missing | - | present | - | - | - |
| no | missing | - | present | - | - | - |
| pa | missing | - | present | - | - | - |
| pl | missing | - | present | - | - | - |
| ps | missing | - | present | - | - | - |
| pt | missing | - | present | - | - | - |
| ro | missing | - | present | - | - | - |
| ru | missing | - | present | - | - | - |
| si | missing | - | present | - | - | - |
| sk | missing | - | present | - | - | - |
| sl | missing | - | present | - | - | - |
| sq | missing | - | present | - | - | - |
| sr | missing | - | present | - | - | - |
| sv | missing | - | present | - | - | - |
| ta | missing | - | present | - | - | - |
| te | missing | - | present | - | - | - |
| tg | missing | - | present | - | - | - |
| th | missing | - | present | - | - | - |
| tr | missing | - | present | - | - | - |
| uk | missing | - | present | - | - | - |
| ur | missing | - | present | - | - | - |
| uz | missing | - | present | - | - | - |
| vi | missing | - | present | - | - | - |
| xh | covered | xho | missing | 5718 | 817 | 1633 |
| yi | missing | - | present | - | - | - |
| yo | covered | yor | present | 6877 | 983 | 1964 |
| zh | missing | - | present | - | - | - |
| zu | covered | zul | missing | 5848 | 836 | 1670 |

## Excluded configurations

| Config | ISO 639-3 | ISO 639-1 | Name | Reason |
| --- | --- | --- | --- | --- |
| bam | bam | bm | Bambara | its ISO 639-1 code is not in the 85-language target list |
| bbj | bbj | - | Ghomala | no ISO 639-1 code, so it cannot match the target list |
| ewe | ewe | ee | Ewe | its ISO 639-1 code is not in the 85-language target list |
| fon | fon | - | Fon | no ISO 639-1 code, so it cannot match the target list |
| kin | kin | rw | Kinyarwanda | its ISO 639-1 code is not in the 85-language target list |
| lug | lug | lg | Luganda | its ISO 639-1 code is not in the 85-language target list |
| luo | luo | - | Dholuo | no ISO 639-1 code, so it cannot match the target list |
| mos | mos | - | Mossi | no ISO 639-1 code, so it cannot match the target list |
| nya | nya | ny | Chichewa | its ISO 639-1 code is not in the 85-language target list |
| pcm | pcm | - | Nigerian Pidgin | no ISO 639-1 code, so it cannot match the target list |
| sna | sna | sn | chiShona | its ISO 639-1 code is not in the 85-language target list |
| swa | swa | sw | Kiswahili | its ISO 639-1 code is not in the 85-language target list |
| tsn | tsn | tn | Setswana | its ISO 639-1 code is not in the 85-language target list |
| twi | twi | tw | Akan/Twi | its ISO 639-1 code is not in the 85-language target list |
| wol | wol | wo | Wolof | its ISO 639-1 code is not in the 85-language target list |

## Caveats

- Spans are character offsets in the tokens joined by single spaces. The upstream tokenization pipeline is not documented, so these spans are not offsets in the original article text.
- Only LOC is scored as a location. PER, ORG and DATE tags are parsed but not scored.
- NER annotations carry no coordinates. A coordinate-resolution claim needs a separate resolver run against a gazetteer.
- The dataset license is conflicting. Resolve it before any publication or commercial use.
- Training overlap is not assessed. The upstream repository ships baseline result files for several encoders, so any evaluated checkpoint must be checked against its own model card before its scores are reported.
- The example counts are README claims. The split files were not downloaded, so the counts are not checked against their contents.
- The Hugging Face loader reads the raw GitHub main branch. Check each file against its pinned git blob id before use.
