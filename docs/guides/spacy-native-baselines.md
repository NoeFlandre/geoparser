# Native-language spaCy baselines

This guide describes the native-language spaCy baselines for issue #162. They
replace the idea of one English spaCy model acting as a multilingual baseline.
Each of the 85 target language codes is either routed to one pinned native
pipeline or recorded as unsupported. An unsupported code is never sent to the
English pipeline.

Only the roster, routing, label harmonization, loading guards and matched-versus-transfer
scoring are implemented. No pipeline has been installed or run, and no quality,
speed or memory result exists yet. `roster.json` records that the 23 selected
wheels were downloaded once to take their sha256 digests, then deleted; see the
gaps below.

## Roster

The checked-in roster is `scripts/spacy_native_baselines/roster.json`. It has
the format `spacy-native-baseline-roster-v1` and covers the 85 codes in
`scripts/panx_benchmark/target_languages.json`, exactly once each.

- Every native pipeline pins spaCy model version `3.8.0`, the GitHub release tag
  `<package>-3.8.0`, and its wheel file name, byte size and sha256 digest.
- The compatibility list is `compatibility.json` in `explosion/spacy-models`,
  retrieved on 2026-10-09. It lists 23 target codes with a native pipeline and
  62 without one.
- Target code `no` (Norwegian) uses the spaCy Norwegian Bokmål pipeline
  `nb_core_news_sm`. Nynorsk coverage is not claimed.
- The Croatian pipeline `hr_core_news_sm` is outside the 85-code list and is not
  routed. The multilingual pipeline `xx_ent_wiki_sm` is not a native-language
  baseline and is not routed either.
- One small (`_sm`) pipeline is selected per language. Larger variants are listed
  under `other_3_8_packages_not_selected` but are not selected.

| Target | spaCy code | Package (3.8.0) | Hub license | Place labels mapped to LOC | Label basis |
|---|---|---|---|---|---|
| ca | ca | `ca_core_news_sm` | gpl-3.0 | `LOC` | ontonotes names |
| da | da | `da_core_news_sm` | cc-by-sa-4.0 | `LOC` | ontonotes names |
| de | de | `de_core_news_sm` | mit | `LOC` | ontonotes names |
| el | el | `el_core_news_sm` | cc-by-nc-sa-3.0 | `GPE`, `LOC` | ontonotes names |
| en | en | `en_core_web_sm` | mit | `FAC`, `GPE`, `LOC` | ontonotes names |
| es | es | `es_core_news_sm` | gpl-3.0 | `LOC` | ontonotes names |
| fi | fi | `fi_core_news_sm` | cc-by-sa-4.0 | `FAC`, `GPE`, `LOC` | ontonotes names |
| fr | fr | `fr_core_news_sm` | lgpl-lr | `LOC` | ontonotes names |
| it | it | `it_core_news_sm` | cc-by-nc-sa-3.0 | `LOC` | ontonotes names |
| ja | ja | `ja_core_news_sm` | cc-by-sa-4.0 | `FAC`, `GPE`, `LOC` | ontonotes names |
| ko | ko | `ko_core_news_sm` | cc-by-sa-4.0 | `LC` | unverified label meaning |
| lt | lt | `lt_core_news_sm` | cc-by-sa-4.0 | `GPE`, `LOC` | ontonotes names |
| mk | mk | `mk_core_news_sm` | cc-by-sa-4.0 | `FAC`, `GPE`, `LOC` | ontonotes names |
| nl | nl | `nl_core_news_sm` | cc-by-sa-4.0 | `FAC`, `GPE`, `LOC` | ontonotes names |
| no | nb | `nb_core_news_sm` | mit | `GPE_LOC`, `LOC` | inferred from label name |
| pl | pl | `pl_core_news_sm` | gpl-3.0 | `geogName`, `placeName` | inferred from label name |
| pt | pt | `pt_core_news_sm` | cc-by-sa-4.0 | `LOC` | ontonotes names |
| ro | ro | `ro_core_news_sm` | cc-by-sa-4.0 | `FACILITY`, `GPE`, `LOC` | inferred from label name |
| ru | ru | `ru_core_news_sm` | mit | `LOC` | ontonotes names |
| sl | sl | `sl_core_news_sm` | cc-by-sa-4.0 | `LOC` | ontonotes names |
| sv | sv | `sv_core_news_sm` | cc-by-sa-4.0 | `LOC` | ontonotes names |
| uk | uk | `uk_core_news_sm` | mit | `LOC` | ontonotes names |
| zh | zh | `zh_core_web_sm` | mit | `FAC`, `GPE`, `LOC` | ontonotes names |

The other 62 codes are recorded as `unsupported` with a reason. Examples include
`ar`, `fa`, `he`, `hi`, `tr` and `vi`. The three codes missing from the WikiANN
test split (`ha`, `xh`, `zu`) are also unsupported.

The licenses are the Hub `cardData.license` values of the same package names.
The Hub model cards describe spaCy 3.7.x content, not the 3.8.0 wheels, so these
licenses are not verified against the wheels. Check them before publication.
Several entries are non-commercial or copyleft (`cc-by-nc-sa-3.0`,
`gpl-3.0`, `lgpl-lr`), which matters for any publication of results.

## Routing

`Roster.route(code)` returns one of two outcomes:

- A native `Route` with the pinned pipeline.
- A `Route` with `pipeline=None` and the recorded unsupported reason.

A code outside the 85 raises `UnknownLanguageError`. No code is ever routed to
the English pipeline unless it is `en` itself. The English pipeline is also the
cross-language control, and it is reported only as a labelled transfer score,
never as a native result. The control never scores an unsupported code.

## Label harmonization

Native label sets differ, so each pipeline maps its own place labels explicitly.

- `FAC`, `GPE` and `LOC` map to `LOC` where the pipeline defines them.
- Labels that are not in the map are dropped and never emitted. Persons,
  organizations, miscellaneous labels and dates are dropped in every language.
- In Norwegian, `GPE_LOC` maps to `LOC`, and `GPE_ORG` is dropped because it is an
  organization-type label.
- In Polish, `geogName` and `placeName` map to `LOC`. `orgName` and `persName`
  are dropped.
- In Romanian, `FACILITY` maps to `LOC` as the equivalent of `FAC`.
- In Korean, `LC` maps to `LOC`. The pinned model card does not define the
  label, so this is recorded as unverified.

The label meanings in the Norwegian, Polish, Romanian and Korean entries are
inferred from label names, not stated in the pinned model cards. Check these
mappings before any result is reported.

## Loading and missing pipelines

`scripts/spacy_native_baselines/loading.py` never downloads a pipeline. A
missing or differently versioned package raises `MissingPipelineError`, and the
message names the pinned wheel URL to install. That URL carries the recorded
SHA-256 digest as a fragment, which pip checks before installing, so a replaced
release asset is refused. The message also lists the tokenizer requirements the
pipeline records (`sudachipy` and `sudachidict_core` for Japanese, `natto-py`
for Korean, `spacy-pkuseg` for Chinese). Before loading, each recorded tokenizer
requirement is checked against the installed release. A missing tokenizer, or
one outside its specifier, raises `TokenizerRequirementError` and nothing is
loaded. After loading, the NER labels must
equal the roster's label set exactly, or `LabelSchemeError` is raised. Only the
`ner` component and its `tok2vec` are kept.

## Recognizer and scoring

- `geoparser/modules/recognizers/spacy_native.py` provides
  `NativeSpacyRecognizer`. It takes a pipeline that the caller has already loaded.
  It implements the `Recognizer` interface and emits character spans for mapped
  labels only. Its `id` is deterministic for the same configuration.
- `scripts/spacy_native_baselines/evaluation.py` scores all arms on the same
  examples and reports three groups that are never merged:
  - `matched`: a native pipeline on examples in its own language.
  - `transfer`: the English control on every non-English language that also has
    a matched native recognizer, labelled as cross-language transfer. Transfer
    and matched therefore cover the same non-English languages; the matched
    group may additionally contain the native English score.
  - `unsupported`: counts of examples in languages with no native pipeline. No
    model predicts them.
  - Macro averages: `matched_macro` may include the native English score, so
    compare `paired_matched_macro` (matched scores over exactly the transfer
    languages) with `transfer_macro` for the headline gap.
- `configuration_id(pipeline)` returns a deterministic 16-hex identity from the
  pinned package, wheel, wheel URL, SHA-256 digest, tokenizer requirements,
  label set and label harmonization.
- `run_identity(pipeline)` extends that identity with the installed versions of
  spaCy, the pipeline package and each tokenizer requirement. The tokenizer
  specifiers (`sudachipy`, `sudachidict_core`, `natto-py`) are open-ended and no
  reviewed release is pinned, so clean installs at different times can tokenize
  differently. Use `run_identity` for result records: two runs are comparable
  only when their run identities match.

## Verification status and gaps

Verified:

- Native and unsupported status for all 85 codes, from the compatibility list.
- Release tags, wheel names and byte sizes for the 23 selected 3.8.0 packages,
  from the GitHub release API.
- Hub license values and model-card NER label sets for the same package names.
- The installed spaCy 3.8.16 extras that declare the Japanese tokenizer
  dependencies (`sudachipy`, `sudachidict_core`) and the Korean `natto-py` extra.

Not verified, and recorded as gaps:

- The sha256 digests for the 23 selected wheels are recorded in
  `roster.json` (`verification.sha256_note`): each wheel was downloaded once,
  hashed, checked against its byte size and its NER labels in `meta.json`, and
  then deleted. That record was written with commit 893ab8a. It has not been
  re-checked, except that the zh wheel's digest and size match its release URL.
- The GitHub release API reports no digest for these assets. The sha256 values
  come only from the one-time download recorded above, so no published digest
  confirms them.
- The licenses have not been checked against the 3.8.0 wheels. The Hub cards
  describe 3.7.x content.
- The Korean system dependency (`mecab`) is not verified.
- No Hub revision is pinned, because the Hub main describes 3.7.x content.

Not done in this phase: no model run, no inference, no compute benchmark, no
recognition-quality or speed/memory comparison, no mutation or CRAP gate run, no
independent review, and no change to the existing English-only PAN-X runner.
Those steps wait for separate approval.
