# Six-model benchmark matrix

## Completion target

Make a draft pull request from a dedicated branch that defines the six
recognizer/resolver combinations, keeps the existing result identifiers
readable, emits per-example references and predictions, and publishes only
validated evidence in separate strict-raw and NewsLi exact-dedup views. The
publisher must make one parent-SHA-guarded Hub commit and retain a recovery
path for the prior public tree.

The new matrix covers 20 registered corpus slices by six combinations (120
cells). The separate dedup view covers the seven affected NewsLi languages by
six combinations (42 cells). Recognition and resolution remain independently
scored; resolution uses gold spans. No end-to-end score is reported.

## Scoring unit and duplicate rows

Both metrics use distinct `(document_id, start, end)` span identities.
Recognition computes set-based exact-match scores; resolution looks up one
coordinate pair per distinct gold span identity. Exact duplicate reference
rows in the strict-raw view stay in the inspected source evidence and row
counts, but do not add to either metric denominator. The NewsLi dedup view
removes the audited exact duplicate rows before scoring. Its record counts may
differ from strict raw while its metric values match. Reports expose both the
gold reference row count and the distinct scoring-span count. These counts
expose the existing entity-unique scoring behavior while keeping reference row
counts available for audit.

## Work units

1. Add one declarative component registry for the six combinations, exact
   model pins, execution settings, and explicit legacy-ID mappings.
2. Add a named NewsLi exact-full-record-dedup view without changing the
   strict-raw default or modifying the source archive.
3. Emit complete per-document gold and prediction records without article
   text, including explicit phase failures and incomplete states.
4. Validate all expected model roles and pins before a result can enter a
   public bundle. Keep strict and dedup results in distinct schemas/paths.
5. Replace multi-write publishing with one parent-SHA-guarded commit, a
   concise card, versioned current outputs, and a recoverable archive pointer.
6. Verify each unit with focused behavior tests, run repository CI checks,
   push the branch, open a draft PR, and review the resulting diff independently.

## Fresh execution plan

Do not launch these shards until the runner change has been reviewed and the
private staging and dataset-write permissions below are settled. Use a new
matrix run ID and the full reviewed Git commit in every manifest.

### Matrix and pins

Each corpus/view run requests all six pipeline IDs and both phases. The six
rows are the Cartesian product of these recognizers and resolvers:

| Pipeline ID | Recognizer | Recognizer revision | Resolver | Resolver revision(s) |
| --- | --- | --- | --- | --- |
| `gliner-minilm` | `fastino/gliner2.5-multi-v1` | `cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3` | `dguzh/geo-all-MiniLM-L6-v2` | `d678769f195c194ffdc6f3735a9360118a855b43` |
| `gliner-jina` | `fastino/gliner2.5-multi-v1` | `cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3` | Jina embeddings + reranker | `jinaai/jina-embeddings-v5-text-small@dd76d535f5447ca3897a9c893fb1e612ead98192` + `jinaai/jina-reranker-v3.5@e8a93f33f0b22108f8c2364f8484ce3422552fbc` |
| `xlm-minilm` | `Davlan/xlm-roberta-base-ner-hrl` | `253f557bd8249b8515114cfd7f71974fe5fa4d2f` | `dguzh/geo-all-MiniLM-L6-v2` | `d678769f195c194ffdc6f3735a9360118a855b43` |
| `xlm-jina` | `Davlan/xlm-roberta-base-ner-hrl` | `253f557bd8249b8515114cfd7f71974fe5fa4d2f` | Jina embeddings + reranker | `jinaai/jina-embeddings-v5-text-small@dd76d535f5447ca3897a9c893fb1e612ead98192` + `jinaai/jina-reranker-v3.5@e8a93f33f0b22108f8c2364f8484ce3422552fbc` |
| `otter-minilm` | `whoisjones/otter-cross-mmbert` | `8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d` | `dguzh/geo-all-MiniLM-L6-v2` | `d678769f195c194ffdc6f3735a9360118a855b43` |
| `otter-jina` | `whoisjones/otter-cross-mmbert` | `8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d` | Jina embeddings + reranker | `jinaai/jina-embeddings-v5-text-small@dd76d535f5447ca3897a9c893fb1e612ead98192` + `jinaai/jina-reranker-v3.5@e8a93f33f0b22108f8c2364f8484ce3422552fbc` |

No population prior or inflection trimming is part of the current run. Pin the
reviewed code commit and `uv.lock` digest. For every corpus, freeze the fetched
source URL and source-content SHA256 in the run manifest; for HIPE/NewsEye and
TopRes also record the committed Wikidata coordinate-cache SHA256. The
GeoVirus and HIPE source URLs use mutable upstream branch refs and the matrix
registry does not freeze their byte digests, so a shard must record the exact
bytes it validated before scoring.
NewsLi is additionally checked against the release archive at
`https://ndownloader.figshare.com/files/59465342`: 63,356,728 bytes, MD5
`8e6f82a0931cb909e6f18217e80ce1f3`, SHA256
`d4672d28045868cd64c52febedb2f62ae85288b9572e70bb7026b410f355d192`.
Record the installed GeoNames artifact SHA256, device, threshold, execution
settings, and any model-specific source/weight manifest pins as well. Use the
current default configuration: `--min-similarity 0.0`, `--chunk-size 5`, and
`--device auto` on a one-GPU/four-core Grid job; record the device actually
selected and the remaining component-specific settings in each manifest.

### Corpus/view shards

Use one corpus and one scoring view per resumable job. Strict raw has 20
corpora and 120 cells. The separate NewsLi dedup view has seven corpora and 42
cells. The counts below are documents and gold reference rows. Dedup counts
exclude the audited exact duplicate records.

| Corpus | Strict raw docs / gold | Dedup docs / gold |
| --- | ---: | ---: |
| `geovirus` | 229 / 2,167 | — |
| `hipe2020-de` | 48 / 558 | — |
| `hipe2020-en` | 38 / 158 | — |
| `hipe2020-fr` | 43 / 800 | — |
| `newseye-de` | 8 / 628 | — |
| `newseye-fi` | 18 / 214 | — |
| `newseye-fr` | 33 / 665 | — |
| `newseye-sv` | 18 / 265 | — |
| `topres19th-en` | 110 / 880 | — |
| `newsli-ar` | 500 / 3,319 | 500 / 2,848 |
| `newsli-de` | 500 / 1,019 | 500 / 1,015 |
| `newsli-es` | 500 / 1,270 | 500 / 1,252 |
| `newsli-fa` | 71 / 326 | 71 / 179 |
| `newsli-ja` | 500 / 1,577 | 500 / 1,528 |
| `newsli-pl` | 186 / 196 | — |
| `newsli-ro` | 226 / 385 | — |
| `newsli-sr` | 500 / 985 | — |
| `newsli-ta` | 500 / 1,578 | 500 / 1,017 |
| `newsli-tr` | 500 / 1,578 | 500 / 1,362 |
| `newsli-uk` | 164 / 258 | — |
| **Totals** | **4,692 / 18,826** | **3,071 / 9,201** |

Strict GeoVirus input is fetched from
`https://raw.githubusercontent.com/milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation/master/data/Corpora/GeoVirus.xml`.
HIPE-2022 v2.1 test TSVs use
`https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/{dataset}/{language}/HIPE-2022-v2.1-{dataset}-test-{language}.tsv`.
The runner records source-byte SHA256 and parsed-corpus digest in each shard
manifest. The Wikidata coordinate cache is pinned to SHA256
`a6779144c089430c9c3b55067b5a1a3f766d0a2f13eab928d354c79c0ad481bb`; `uv.lock`
is pinned to SHA256
`4a3b13edc893cecab72784d9d7f3d1e297af57ce59b6eca8f84111b5eea8821e` at this
plan revision. NewsLi's seven dedup shards use the same pinned archive as their
strict counterparts, but have a distinct scoring-view identity and digest.
Never reuse checkpoints between strict and dedup views.

### Reuse, resume, and missing work

Within a shard, the six pipeline rows need only five distinct component-phase
executions: three recognizers and two resolvers. Recognition can be shared
between the two resolver pairings for the same recognizer. Resolution can be
shared between recognizers only on the gold spans, with the same resolver.
This is component reuse, not an end-to-end score. Keep recognition and
resolution results and failures separately inspectable.

Resume a shard every five documents and give each shard its own output path,
for example `RESULTS/evidence/<run-id>/<view>/<corpus>`, while sharing the
verified GeoNames artifact at `RESULTS/artifacts/geonames.db`. Use a unique OAR
job name per shard, and retain that name only for a deliberate resume after
checking scheduler state. Run one small seed shard to build and hash the
shared gazetteer artifact before starting parallel shards: use the complete
eight-document `newseye-de` strict-raw shard, so it remains a target result
instead of a disposable warm-up. After validating that seed, start with up to
two concurrent one-corpus jobs as a conservative rollout level, not as a
Grid'5000 policy limit. Increase safe parallelism across available sites only
after checking live scheduler capacity and running `usagepolicycheck -t` for
the planned reservations; do not make duplicate or speculative reservations.
Keep each job at one GPU/four cores with the existing two-hour walltime. Use
the existing Grid coordinator and credentials only; do not use paid Hugging
Face Jobs or change access. A failed or length-limited document remains in
the checkpoint and report with its explicit failure status and empty
prediction where applicable; it is never dropped or silently truncated.

Do not count old published rows as current coverage solely from their pipeline
alias. Reuse an old checkpoint or result only after its source digest, view,
full reviewed commit, every role pin, device and phase settings, and complete
document coverage pass the new compatibility and provenance gates. The 19
held jobs with no staged manifests/checkpoints are not resumable as-is. They do
not block fresh, correctly pinned shards under new job/run IDs. The public 50
rows are legacy three-pipeline evidence (29 strict-raw and 21
dedup-view rows), not current six-pipeline coverage. Inspect these legacy
results before considering component reuse: NewsLi-SR GLiNER/Jina (job
4217426), NewsLi-PL GLiNER/Jina (job 6981801), and NewsLi-PL XLM/MiniLM (job
6981821). For job 6981801, the initial OAR wrapper exited 256, but the
corrected worker attempt and result bundle both recorded exit 0 with matching
hashes. Check both phases and each complete manifest against the new identity
gate. Job 341320's Otter/NewsLi-PL checkpoint
records all 186 documents and both phases but has 15 sequence-length failures;
preserve those failures if it passes the new identity gate. Count no old
checkpoint as reusable until it passes that gate. If none pass, plan for 27
shard runs, 54 shard-level phase outputs, 162 pipeline cells, and up to 135
unique component-phase executions after within-shard reuse.

### Output and transfer bounds

Each job writes only one corpus/view's checkpoints, reports, inspection files,
and run manifest. Do not include article text, model caches, or the shared
gazetteer in the transfer bundle. Keep model raw-output diagnostics at the
runner's existing 4,096-character cap; full structured predictions and gold
references remain available for inspection. Before any transfer, build and
list the exact bundle and report its compressed byte count and SHA256. Private
staging is still pending approval. The proposed limit is 16 MiB total staged
transfer bytes at any one time across all active shards, with canonical Grid
checkpoints retained separately under the storage ledger; this is a proposal,
not an approved limit. Do not split or stage data until the workflow and cap
are approved. Do not delete temporary staged files unless approval explicitly
covers deletion after successful Hub readback. Retain the canonical Grid
checkpoints and archive.

Two permissions are still missing. Approve the proposed private staging and
cleanup workflow, including the proposed aggregate 16 MiB in-flight cap.
Grant write access on the public dataset
`NoeFlandre/geoparser-benchmark-results` to an already authorized token. Do
not create credentials or change access settings. Once both are available,
publish the validated current/archive tree in one parent-SHA-guarded Hub commit
and verify it by readback.

## Stop conditions

- Do not start, cancel, or resubmit Grid jobs until current scheduler state is
  available and reconciled.
- Do not stage private data for transfer without the pending approval.
- Do not publish to Hugging Face while the available token lacks write access.
- Do not merge the pull request.

## Verification evidence

Record the test command and outcome, branch head, PR URL, CI state, and review
findings in the handoff. A complete benchmark result is not claimed until all
expected cells have per-example inspection files, model/config/source pins,
and explicit failures or completed metrics.
