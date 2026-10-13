# Universal NER recognition adapter

The adapter prepares public Universal NER source files for location recognition. It does not run models or download data. Issue #166 remains open until an approved model evaluation has retained predictions and complete run evidence.

## Release scope and pins

[Universal NER v2](https://arxiv.org/abs/2604.12744v1) describes 30 configurations across 22 languages. Norwegian has two written standards. The official [UniversalNER organization](https://github.com/UniversalNER) stores the source files. Its dataset repositories have no version tags for v2. The inventory pins the latest commit before 15 April 2026 UTC in each of the 29 repositories. This is a reproducible snapshot of the paper's scope. It is not an official immutable release archive. The linked Hugging Face aggregate dates to 2024 and does not establish the v2 inventory.

`scripts/uner_benchmark/sources.json` records all 30 configurations and 62 split files. Each file has its Git blob checksum. The inventory also retains SHA-256 and Git checksums for 93 inspected README, license and statistics files. These are metadata checks, not proof that every dataset sentence is valid. Later Indonesian CSUI and GSD additions are outside this snapshot.

The test sources intersect the canonical 85-language target list in 19 languages. Seventeen are eligible under the recorded license policy: `ceb, cs, da, de, en, he, id, ja, no, pt, ro, ru, sk, sl, sr, sv, zh`. This covers 24 configurations. Greek `el_gdt` and Korean `ko_pud` remain visible as `license_unverified`: their pinned UNER trees have no annotation-license declaration. Do not infer permission from a UD text license. The other 66 target languages have no source test split in this inventory. The CLI lists every missing code.

Keep source distinctions:

- Map `nno_norne` and `nob_norne` to canonical `no`, with separate source configurations for Nynorsk and Bokmål.
- Keep `zh_gsd`, `zh_gsdsimp` and `zh_pud` separate. Do not convert scripts. GSD and GSDSIMP share underlying material and nearly identical annotations.
- `hr_set`, `qaf_arabizi`, `tl_trg` and `tl_ugnayan` are outside canonical85. Do not substitute Serbian for Croatian, or `ar` or `fr` for code-switched Romanized NArabizi.
- The unsuffixed `ro_legalnero.iob2` is test-only, as stated in the v2 paper. Its name does not imply a training or development split.

The source licenses differ. Five PUD configurations use CC-BY-SA-3.0; most other declared licenses use CC-BY-SA-4.0. Tagalog Ugnayan uses CC-BY-NC-SA-4.0 and is outside canonical85. The Romanian UNER README and license declare CC-BY-SA-4.0 for this export; the referenced Zenodo original has different terms. Preserve each source notice. The repository's MIT code license does not replace dataset licenses.

## Domains and annotation provenance

Each configuration records its domains and the relevant paper. The [v1 paper, Table 5](https://aclanthology.org/2024.naacl-long.243.pdf) and [v2 paper, Table 3](https://arxiv.org/html/2604.12744v1) describe domains of the underlying source corpora. The CLI's `domain_configuration_counts` counts selected configurations with each domain tag. Tags overlap. These are not measured sentence proportions.

For the 24 eligible test configurations, 18 include news and 11 include wiki text. The inventory also includes blogs, email, reviews, social and other web text, fiction, nonfiction, spoken text, grammar examples and Romanian legal text. Development data covers 13 eligible configurations in 10 target languages. Test-only sources cannot supply development data.

Human provenance includes both direct UNER annotation and conversions of earlier manual work. The inventory identifies DaN+, hr500k, SETimes.SR, NArabizi, NorNE, JANES and LegalNERo transfers. Converted human labels are not WikiANN-style silver labels, but conversion and guideline differences remain part of the evidence.

## Preserve the source text

Each record keeps the release's `# text` field exactly, including repeated whitespace, punctuation, combining characters and script. Offsets are half-open Python character indices in that field. The reader does not rebuild the sentence by joining tokens. Source text can already contain UD segmentation conventions. Preservation of a release field does not establish that it is the original pre-tokenization document.

Each sentence retains its source sentence ID and document ID. Its evaluation ID contains the source configuration, split and sentence ID. Missing document IDs use the sentence ID as a separate unit. The optional `source_document_id` distinguishes those fallback IDs from explicit document IDs with the same spelling. The adapter does not join sentences or infer missing document boundaries.

Only `B-LOC` and `I-LOC` make location spans. Person, organization and retained `OTH` labels remain negative examples. The paper says OTH is removed from the final release, but the pinned Slovenian export retains it. The adapter records this difference instead of relabeling OTH as LOC. The source [annotation guidelines](https://www.universalner.org/guidelines/) include buildings, facilities, fictional places and planets as locations. They use context to distinguish an organization from a place. Keep those source distinctions. No coordinates or entity links are assigned.

The parser requires the five UNER columns and sequential integer token IDs. It validates IOB2 transitions for every class. It accepts the released Croatian/Serbian document header with no space after `=`. An orphan inside label, unknown label, missing sentence metadata, duplicate ID or nonliteral token alignment fails the source. Unsupported multiword or empty-node IDs fail explicitly. The parser returns no partial corpus after a failure. Empty texts and sentences without locations stay in the denominator.

`benchmark-evidence/uner/adapter-source-audit.json` records a format check of 643 complete sentences across all 30 test-source prefixes. Each read was capped at 8,192 bytes. The check exposed the document-header and OTH conventions above. It is not a full-corpus validation or a model result. Full local file validation remains mandatory before inference.

## Validate a plan

Run the metadata check without opening a dataset or model:

```bash
uv run python -m scripts.uner_benchmark --dry-run --split test
uv run python -m scripts.uner_benchmark --dry-run --split dev
```

Choose exact configurations with repeated `--configuration` options. The report retains unavailable splits, sources outside canonical85, and unresolved licenses. `present_target_languages` describes source presence; `covered_target_languages` describes the eligible subset; `unavailable_target_languages` identifies present but blocked targets. `missing_target_languages` identifies absent source splits within the selected scope. A registry-only success does not validate file contents or authorize an experiment.

Place previously acquired public files under `CACHE/REPOSITORY/SOURCE_PATH`. To check local bytes and every sentence, use:

```bash
uv run python -m scripts.uner_benchmark --dry-run --split test --cache-dir CACHE
```

This command verifies each file against the pinned Git blob ID before parsing it. It reports a SHA-256 digest of the actual bytes and complete sentence, document, token, location and empty-text counts. Missing files, checksum errors and malformed sources remain separate failed configurations. The command does not download replacements or fall back to another split. Exit status 2 indicates invalid arguments, failed local data or no available selected configuration. Split paths and payload hashes cannot alias within a configuration.

## Keep evaluation separate

Import `load_local` from `scripts.uner_benchmark.inventory`. Each loaded sentence exposes the existing `scripts.panx_benchmark.data.Example` interface through its `example` field. Use its stable `identifier` for recognition units and retain its source document ID when selecting a document-level sampling policy. Retain the local file's SHA-256 in experiment provenance. Do not pass these records to the coordinate-based resolution runner.

Use `dev` for development choices. Keep `train`, `dev` and `test` files separate. Freeze thresholds, labels, prompts and model selection before held-out evaluation. The small source-prefix inspection was for adapter correctness only. Separate approval is required for model downloads and execution.

Human annotation does not establish independence from model training. Source texts overlap Universal Dependencies, PUD and other published corpora. Parallel translations and GSD script variants are related samples. Some UNER v1 data have also been incorporated into later multilingual training resources. Mark model-specific training overlap as unknown unless a pinned audit establishes otherwise. Retain known overlap explicitly. Do not infer absence from a missing training-data disclosure.
