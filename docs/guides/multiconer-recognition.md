# MultiCoNER II place recognition

This guide defines the recognition-only comparison for issue #167. It uses the public MultiCoNER II release, pinned by revision. It does not run models, download the data, or report coordinate accuracy. Use the public benchmark protocol in [Public benchmark protocol](benchmark-protocol.md) for every run.

## Pinned source

| Item | Value |
| --- | --- |
| Dataset | `MultiCoNER/multiconer_v2` on the Hugging Face Hub |
| Revision | `4be2d62c912977ee26ed14d2553a4fe17ca3d980` (2023-07-06) |
| Official homepage | <https://multiconer.github.io/dataset> |
| License | CC BY 4.0, from the README front matter at the pinned revision |
| Manifest | `scripts/multiconer_benchmark/multiconer_manifest.json` |

The verification used metadata only. It compared the Hub tree at the pinned revision, the git blob ids of `README.md` and `multiconer_v2.py`, and the dataset-viewer row counts. The per-split counts below come from the official data-statistics table. They sum to the viewer totals for every language.

Test files for `bn`, `de`, `hi` and `zh` are not stored in LFS. Their sha256 is unknown until a file is downloaded, so the manifest records only the git blob SHA-1 for them. The other test files record their LFS sha256.

## Language intersection

The dataset has 12 languages. Each one is in the 85-code inventory in `scripts/panx_benchmark/target_languages.json`. The intersection is therefore the same 12 codes:

`bn`, `de`, `en`, `es`, `fa`, `fr`, `hi`, `it`, `pt`, `sv`, `uk`, `zh`.

The `MULTI` configuration is an aggregate, not a language, so it is excluded. Its relation to the per-language files is not verified. The dataset tree also holds three `MULTI` split files, which are not pinned or scored here. The manifest pins the 36 per-language split files. Together with those three, the tree holds 39 `.conll` files. Each split file is parsed on its own, and the splits are never merged.

| Code | Language | Train | Dev | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| `bn` | Bangla | 9,708 | 507 | 19,859 | 30,074 |
| `de` | German | 9,785 | 512 | 20,145 | 30,442 |
| `en` | English | 16,778 | 871 | 249,980 | 267,629 |
| `es` | Spanish | 16,453 | 854 | 246,900 | 264,207 |
| `fa` | Farsi | 16,321 | 855 | 219,168 | 236,344 |
| `fr` | French | 16,548 | 857 | 249,786 | 267,191 |
| `hi` | Hindi | 9,632 | 514 | 18,399 | 28,545 |
| `it` | Italian | 16,579 | 858 | 247,881 | 265,318 |
| `pt` | Portuguese | 16,469 | 854 | 229,490 | 246,813 |
| `sv` | Swedish | 16,363 | 856 | 231,190 | 248,409 |
| `uk` | Ukrainian | 16,429 | 851 | 238,296 | 255,576 |
| `zh` | Chinese | 9,759 | 506 | 20,265 | 30,530 |

## Label policy

The release uses fine-grained entity types. The place decision follows the coarse Location group of the published tagset. Gold labels are kept exactly as released. The mapping is applied only when the span set is built, and the full mapping is recorded in the protocol configuration.

| Policy | Fine-grained types | Scoring effect |
| --- | --- | --- |
| `LOC` (place) | `Facility`, `OtherLOC`, `HumanSettlement`, `Station` | Gold spans enter the recall denominator. |
| `ignore` (non-place) | All other types, including groups and organisations (`PublicCORP`, `PrivateCORP`, `ORG`, `SportsGRP`, `CarManufacturer`), creative works, people, products and medical types | Gold spans leave the recall denominator. |

Metonymy is handled by the gold label, not by re-annotation. A sports team named after a city is `SportsGRP` in gold, so it is not a place. A prediction that marks it as a place is a false positive. Precision therefore penalises place predictions over organisations. Recall counts only gold places.

The README and the loader script disagree on five names. `OtherCW`, `OtherCorp` and `TechCORP` appear only in the loader. `PublicCorp` and `PrivateCorp` are case variants of the README names. All five are non-place and map to `ignore`. The four place names agree in both sources. The parser refuses any other name, so an unexpected label is an invalid record and is never guessed. Confirm the five names against the data bytes before a run.

The policy is a proposal for owner review. Changing it changes the configuration digest, so it must be decided before any run.

## Text, offsets and Unicode

The release has no original whitespace. Each sentence's text is its tokens joined with one space. Offsets are half-open Python code-point spans into that text. Every system must read the same reconstructed text. Code points keep Devanagari combining marks and astral emoji as the Python standard counts them.

Lines are split on `\n` only. A `\r` before a line break is removed. A leading byte-order mark is removed. Tokens may contain Unicode line separators or no-break spaces. A no-break space is not a column separator, so a line with one counts as a wrong column count.

## Invalid records

Every sentence block is a record. It is either a valid sentence or an invalid record with its first line and a reason. Invalid records are counted and never dropped. The parser reports each one. `require_no_invalid_records` refuses to freeze a source that contains any, because a silent change of the denominator would make the source incomparable.

Reasons:

- `missing sentence header`, for tokens with no `# id` line.
- `missing sample id`, `missing domain`, `malformed header`.
- `duplicate sample id`.
- `domain mismatch: expected X, found Y`, when the expected language is given.
- `empty sentence`.
- `wrong column count: expected 4, found N`.
- `invalid separator columns`.
- `unknown tag format` or `unknown entity type`.
- `I- tag does not continue an entity`, for an orphan or a type change inside an entity.

The loader script at the pinned revision skips a sentence whose token and tag counts differ, without a count. This adapter reports the same case as an invalid record.

## Model arms and the shared protocol

The arms are the three #98 pipelines in `scripts/panx_benchmark/constants.py`: `spacy_en` (English only, so the other languages are explicit `unsupported` outcomes), `gliner2_multi`, and `xlmr_ner_hrl`. Their model pins are reused as written. This phase runs none of them.

Scoring reuses the PAN-X `Counts` accumulator and the protocol `RecognitionCounts` contract. Invalid model outputs count as false positives and are counted once per distinct raw output. A predicted span is valid only if it is a pair of integers with `0 <= start < end <= len(text)`.

## Not covered by this release

- The clean and noisy breakdown is not supported. The header carries only the sentence id and the domain code. No per-sentence corruption flag or separate noisy file is listed at the pinned revision.
- Gold span counts per language need the data bytes. A complete protocol inventory cannot be frozen offline.
- The test-file gold tags are not verified from metadata. The loader reads them as `ner_tags`, but the metadata does not say they are gold.
- The sha256 of the four non-LFS test files, and of all training and dev files, is unknown without a download.

## Dry run and local validation

Print the pinned inventory, which needs no network:

```bash
uv run python -m scripts.multiconer_benchmark --manifest
```

Validate one local split file, after you have obtained it outside this phase. Exit status 2 means a read error or at least one invalid record:

```bash
uv run python -m scripts.multiconer_benchmark --validate-conll path/to/en_test.conll --language en
```
