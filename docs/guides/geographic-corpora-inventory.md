# Geographic corpora inventory

This guide records which corpora from the UniTopRank release and from TopoResolve could extend the benchmark, which already cover them, and what is still unverified. The machine-readable pin is `scripts/benchmark/data/geographic-corpora-snapshot.json`. The inventory was taken on 2026-10-09 from public metadata only. No corpus was downloaded for scoring, and no model ran.

## Status

Only the registered corpora are scored. No new corpus is registered by this inventory. Every candidate below is blocked on a schema check that needs the gold files, and the TopoResolve data has no licence.

The UniTopRank archive (`data.zip`, 63,356,728 bytes, MD5 `8e6f82a0931cb909e6f18217e80ce1f3`) holds 25 dataset folders, which matches the issue. Each folder has a gold JSON file beside it. The archive has 59,519 members in total. The schema of the 12 candidate gold files has not been inspected, so no adapter exists for them yet.

## Eligibility

A corpus is added to the benchmark only when all four conditions hold:

1. Its licence is stated by the source, and the guide records that statement as found. A licence is never inferred.
2. Its gold names toponyms with coordinates, or with gazetteer IDs that resolve to coordinates, in a text the harness can read.
3. It is not a duplicate of a corpus already registered, judged by content and not by name.
4. Its revision is pinned by checksum, and its counts are checked against the source.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| `registered` | Already scored by the harness through the corpus registry. |
| `duplicate` | The same corpus is already registered from another source, so it is not added again. |
| `possible_duplicate` | The name matches a registered corpus. A content comparison decides the status. |
| `candidate` | New. The gold schema, licence or language is not yet verified, so no adapter exists. |
| `excluded` | Not eligible until the stated blocker is resolved. |

## UniTopRank release

The text-file counts come from the member listing of the archive. The language column is what the names and the registry show. Only the NewsLi folders are verified against an adapter. The other language and genre labels are guesses from folder names or ID formats and must be checked before use.

| Folder | Text files | Status | Registration or note | Language, as found |
| --- | --- | --- | --- | --- |
| `19th` | 455 | `possible_duplicate` | The name matches TopRes19th, registered as `topres19th-en` from the HIPE-2022 split. Compare article IDs or text before deciding. | Unverified |
| `ITA-DSTR` | 1807 | `candidate` | Gold not inspected. IDs look like post IDs. | Unverified; name suggests Italian |
| `LDC` | 62 | `candidate` | Gold not inspected. | Unverified |
| `TUD` | 152 | `candidate` | Gold not inspected. | Unverified |
| `ar_geotoponyms` | 757 | `registered` | `newsli-ar` | Arabic |
| `de_geotoponyms` | 13174 | `registered` | `newsli-de`, capped at 500 articles | German |
| `es_geotoponyms` | 8054 | `registered` | `newsli-es`, capped at 500 articles | Spanish |
| `fa_geotoponyms` | 85 | `registered` | `newsli-fa` | Persian |
| `fingernews_gold` | 84 | `candidate` | Gold not inspected. | Unverified; name suggests Finnish |
| `fingertweets_gold` | 980 | `candidate` | Gold not inspected. IDs look like post IDs. | Unverified; name suggests Finnish |
| `geocorpora` | 6648 | `candidate` | Gold not inspected. IDs look like post IDs. | Unverified |
| `geovirus` | 229 | `duplicate` | Already registered as `geovirus` from the original repository. The count matches the 229 articles noted in `scripts/benchmark/corpus.py`. Content was not compared. | English |
| `gwn` | 200 | `candidate` | Gold not inspected. | Unverified |
| `ja_geotoponyms` | 2868 | `registered` | `newsli-ja` | Japanese |
| `lgl` | 588 | `candidate` | Gold not inspected. | Unverified |
| `pl_geotoponyms` | 321 | `registered` | `newsli-pl` | Polish |
| `ro_geotoponyms` | 241 | `registered` | `newsli-ro` | Romanian |
| `semeval` | 90 | `candidate` | Gold not inspected. IDs look like PubMed and PMC articles, so the text is probably scientific rather than news. | Unverified |
| `sr_geotoponyms` | 13950 | `registered` | `newsli-sr`, capped at 500 articles | Serbian |
| `ta_geotoponyms` | 873 | `registered` | `newsli-ta` | Tamil |
| `tr_geotoponyms` | 805 | `registered` | `newsli-tr` | Turkish |
| `trnews` | 118 | `candidate` | Gold not inspected. Its documents are not NewsLi documents, so the overlap with NewsLi is by language only. | Unverified; name suggests Turkish |
| `uk_geotoponyms` | 282 | `registered` | `newsli-uk` | Ukrainian |
| `wiktor` | 5000 | `candidate` | Gold not inspected. Its gold JSON is the largest in the release, about 14.5 MB uncompressed. | Unverified |
| `wotr` | 1644 | `candidate` | Gold not inspected. | Unverified |

The 11 `*_geotoponyms` folders are exactly the NewsLi languages the registry offers. The adapter reads the layout of those folders: `<code>_geotoponyms.json` beside `<code>_geotoponyms/<id>.txt`.

### Language coverage

The registry offers 20 corpora in 15 languages: `ar`, `de`, `en`, `es`, `fa`, `fi`, `fr`, `ja`, `pl`, `ro`, `sr`, `sv`, `ta`, `tr` and `uk`. The candidates would add no verified language. Only `ITA-DSTR`, by its name, would add Italian. The Finnish and Turkish candidates would deepen languages the registry already covers, with different genres. The issue cites 14 languages and seven text types for the release. Neither figure is confirmed here: the release README has no such table, and the paper's dataset table was behind a publisher bot challenge.

## TopoResolve

TopoResolve is pinned at commit `1aae0e0c6a81c708ed47dd974912c2e30f49a727`, dated 2024-12-28. It has three gold files: `GPE_2024_05_21T134100Z.jsonl`, `LOC_2024_05_21T134100Z.jsonl` and `FAC_2024_05_21T134100Z.jsonl`. Its README says each holds 102 records, which was not counted because the files were not downloaded.

Its status is `excluded`, for three reasons:

- **No licence.** The repository metadata has no licence and the repository has no licence file. The README states none. Nothing is claimed.
- **Granularity.** Each gold record names one entity with coordinates and context sentences. It has no character offsets into a text, so it cannot be scored as gold-span resolution without deriving spans.
- **Geometry.** Its README scores GPE by containment in a state or country polygon, and uses a 25-mile radius for the other classes. The gold holds a point, not a polygon, so the polygon would come from outside the gold.

## Licences, as found

| Source | Statement | Note |
| --- | --- | --- |
| UniTopRank, figshare licence field | Apache 2.0 | Used by the repository's current documentation for GeoVirus and NewsLi. |
| UniTopRank, `README.md` in the release | "Free for research and educational use. Please credit the author when reusing or modifying the code or datasets." | Conflicts with the figshare field. It is not resolved here. |
| UniTopRank paper (Crossref) | CC BY 4.0 | The licence of the article, not of the data. |
| TopoResolve repository | None stated | No licence field and no licence file. |

The two UniTopRank statements conflict. `docs/guides/benchmark.md` and `scripts/benchmark/publish.py` describe GeoVirus and NewsLi as Apache 2.0 from this release. That matches the figshare field, but the README terms are narrower. Which statement governs the data is a decision for the owner, and this inventory does not settle it.

## Open decisions

- **Gold schema inspection.** The 12 candidate gold JSON files need their schema read before any adapter. That needs the 63 MB `data.zip` in a scratch location, which is a download the owner must approve. A byte-range request would avoid the full copy. One attempt during this inventory returned the whole file instead, so the range request cannot be relied on. That copy was used only to list the members and was then deleted.
- **Licence conflict.** Decide which UniTopRank statement governs the data before any corpus from the release is added or reported as Apache 2.0.
- **TopoResolve.** Decide whether it may enter the benchmark without a licence, and if so as a separate resolution-only track. Gold-span and end-to-end results stay separate.
- **TopRes19th.** Decide whether `19th` is the registered `topres19th-en` corpus. A comparison of article IDs or text is needed for that.
- **Thresholds.** Choose the distance thresholds to report beside Acc@161 km. None has been chosen.
- **Gazetteer crosswalk.** A pinned ID crosswalk for any new corpus is not built. It needs the gold IDs first.

## Checks available now

`scripts/benchmark/corpus_checks.py` checks any loaded corpus offline. It reports offset and surface alignment, coordinate ranges and finiteness, duplicate and overlapping spans, documents without gold, and totals that differ from the published counts. It reports problems and does not repair them. Region geometry is not checked, because the gold has no polygons.

## Not verified

- The gold schema, record counts and coordinates of the 12 candidates and of the `19th` folder.
- The language and text type of every folder other than the 11 NewsLi folders and `geovirus`.
- The contents of the paper's dataset table, which was behind a publisher bot challenge that was not bypassed.
- Whether `geovirus` in the release matches the GeoVirus XML by content.
