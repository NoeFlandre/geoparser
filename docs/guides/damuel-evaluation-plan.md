# DaMuEL evaluation plan

This plan belongs to issue #100. It covers the zero-shot evaluation of the existing pretrained pipelines on DaMuEL, a multilingual Wikipedia entity-linking dataset. It is a plan and a provenance record. It does not run models, train or fine-tune anything, download dataset archives, or schedule compute. The only code is an offline coverage report over a checked-in metadata snapshot.

## Status

| Item | State |
| --- | --- |
| Official release, licence, schema, dumps, and file inventory | Verified from public metadata (table below) |
| Canonical language coverage against the 85 targets (#99) | Computed offline; reproducible with the command below |
| Geographic-entity filter | Options written down; **decision needed** |
| Gazetteer and coordinate pins | Not started; **decision needed** |
| Adapter, mention-span mapping, evaluator | Not implemented; the issue does not authorize implementation yet |
| Validation sample of automatic annotations | Not drawn |
| Comparative results and recommendation | Not produced |

Regenerate the coverage report offline:

```bash
uv run --no-sync --offline python -m scripts.damuel_inventory
```

The command reads `scripts/damuel_inventory/damuel_1_0_release.json`, which records public LINDAT metadata retrieved on 2026-10-09. It never opens a dataset archive.

## Provenance

Every value below was read from the public record or from metadata inside the archives. No archive was downloaded in full. Items labelled "read from the archive tail" came from an HTTP range request for the last few hundred kilobytes of a tar file.

| Item | Value | Source and check |
| --- | --- | --- |
| Dataset | DaMuEL 1.0, Charles University UFAL | LINDAT/CLARIAH-CZ handle [11234/1-5047](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5047) |
| Paper | Kubeša and Straka, arXiv:2306.09288, submitted 15 June 2023 | arXiv abstract and HTML version |
| Licence | CC BY-SA 4.0 | Stated on the LINDAT record; the bundled `LICENSE` file, read from the archive tail, is the Attribution-ShareAlike 4.0 text |
| Wikipedia dump | 2022-11-01 | Bundled `README` (read from the archive tail) and paper |
| Wikidata dump | 2022-11-09 | Bundled `README` and paper |
| Archives | 53 per-language text archives plus one Wikidata knowledge-base archive | LINDAT listing; 54 entries in the snapshot |
| Knowledge-base archive | `damuel_1.0_wikidata.tar`, 2,715,955,200 bytes, MD5 `778ccad6e829d59938419064c9f8de4d` | DSpace REST API; MD5 matches the listing |
| Text archive checked through the API | `damuel_1.0_wo.tar`, 10,721,280 bytes, MD5 `bc04313f6d6727963ebc77d5a353b7a0` | DSpace REST API; MD5 matches the listing |
| Largest text archive | `damuel_1.0_en.tar`, listed as 24.5 GB | LINDAT listing only |

The `README` and `LICENSE` files are byte-identical in the two archives read from the tail. The two `schema.json` files differ, one per archive type.

The Hugging Face dataset `FrancophonIA/DaMuEL_1.0_fr` is an unofficial French-only mirror tagged `cc-by-sa-4.0`. This plan does not use it as a source.

Not verified: the licence terms of the upstream Wikipedia text and Wikidata data. The arXiv licence on the paper is a different licence from the dataset's. Check both before any redistribution.

## Schema and annotation construction

From `schema.json` in each archive type, and from the paper:

- **Text archives** hold one JSON object per entity record, in JSON Lines, compressed into `part-*.xz` files. The published schema requires only `qid` and `lang` at the top level. `label`, `aliases`, `description`, and `wiki` are optional, so a record can describe an entity that has no Wikipedia article in that language. When present, the `wiki` block holds `text`, `tokens`, `sentences`, `paragraphs`, `redirects`, `anchors`, `links`, and `removed_links`. How records without a `wiki` block are skipped and counted is set out in the mention accounting policy.
- **Token offsets.** `wiki.links` gives `start` and `end` as token indices, not character offsets. Each token has its own `start` and `end`, which the schema describes as positions in the `text` property. Confirm on a sample that these are Python character offsets, with `end` exclusive, so that `text[token.start:token.end]` is the token, before any span mapping relies on them.
- **Link-end convention.** The plan adopts the half-open reading: `end` is the index of the token after the mention, so a link covers tokens `start` to `end - 1`. This is an assumption. The published schema says only that `end` is "the end position of the link in tokens", and no reader code or other primary source found here defines it. The conversion to a character span is `char_start = tokens[start].start` and `char_end = tokens[end - 1].end`, giving the half-open span `text[char_start:char_end]`. The inclusive reading would instead use `tokens[end].end`, and the two readings differ by one token at every boundary. Sampled assertion, run on a seeded sample of links before any scoring: for each sampled link, `0 <= start < end <= len(tokens)`, the slice `text[char_start:char_end]` is non-empty and neither begins nor ends with whitespace, and a person confirms on the validation sample that the slice is the annotated mention. If any sampled link fails, the run stops and the owner decides the convention. The adapter does not switch readings silently.
- **Original versus expanded links.** Each link has an `origin`. The value `wiki` marks a link present in the Wikipedia article. The other values (`title`, `label`, `alias`, `redirects`, `anchors`, and their `_lemmatized` forms) mark mentions added by the authors. The `flat` flag marks a link as the dominant one where no word belongs to more than one link. This plan treats the flat links as the non-overlapping selection and scores only them (see the mention accounting policy). Confirm on a sample that no two flat links share a token before any run relies on this.
- **Expansion rule.** The paper says Wikipedia annotates only one mention per entity on a page. The authors therefore add further mentions only of entities already linked on that same page, so expansion adds no new entity to a page. The paper reports no precision or recall for the expanded mentions, so their accuracy is unknown.
- **Text processing.** The paper says templates, lists, tables, images, galleries, references, category links, and inline formatting were removed. Mention offsets refer to this processed text, not to the rendered article.
- **Knowledge base.** Entities are keyed by Wikidata QID. Claims use typed values, including `wikibase-item` and `globe-coordinate`. The published schema has no rank field, although the paper's text says the knowledge base keeps ranks. The two conflict, and the sample check decides which holds. `named_entities.type` is an array whose elements are drawn from seven automatic classes: PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART, and MANUFACTURED. Figure 2 of the paper declares it as an array of these enum values and sets no maximum length, so one entity can carry several classes. The paper says the classes come from the Wikidata class hierarchy. The `schema.json` lists `fictional` outside its `properties` block, so the schema does not validate it. Check it on data before use.
- **Splits.** Neither the paper, the README, nor the schemas describe a train, development, or test split. The paper names NER and NEL training and evaluation as future work and reports no evaluation. The evaluation slices therefore have to be defined in this repository, as described below. Nothing here preserves an official split.

## Language coverage against the canonical 85

The canonical list is `scripts/panx_benchmark/target_languages.json`, the upstream-pinned set that #99 adopted. Codes are matched exactly. The release has 53 text archives.

- **Covered (48):** af, ar, be, bg, ca, cs, da, de, el, en, es, et, eu, fa, fi, fr, ga, gd, gl, he, hi, hu, hy, id, it, ja, ko, la, lt, lv, mr, mt, nl, pl, pt, ro, ru, sk, sl, sr, sv, ta, te, tr, uk, ur, vi, zh.
- **Missing from the release (37):** am, az, bn, ceb, cy, eo, fy, gu, ha, ig, is, jv, ka, kk, km, kn, ku, ky, mg, mk, ml, mn, ms, my, ne, no, pa, ps, si, sq, tg, th, uz, xh, yi, yo, zu.
- **In the release but not canonical (5):** hr, nn, se, ug, wo.

Points to carry into any run:

- An archive's presence is not usable coverage. Among covered languages, archives run from about 24 MB (gd) to 24.5 GB (en). The 48 covered archives together total about 113 GiB as listed. Sample sizes per language will differ by orders of magnitude.
- `no` (Norwegian, canonical) is missing. `nn` (Nynorsk) is in the release. The matcher does not map one to the other.
- DaMuEL uses one code per language. Script and regional variants are not verified. The canonical list keeps script distinctions, so confirm which script the `sr` and `zh` articles use before counting them as covered for a given variant.
- `hr` is in the release but not in the canonical list. Changing the canonical list is a protocol change, so this plan leaves it alone.

## Geographic-entity filter (decision needed)

The issue requires a documented filter that separates physical places from people, organisations, metonymic references, and other non-place entities. It also rules out treating coordinates as sufficient. The schema does not settle which entities count as places, and this plan does not choose for you. No option below separates metonymic references from literal places, because the KB is entity-level. Item 3 below records how they are counted. The options are:

- **A. Type only.** Keep mentions whose KB entity passes the type test below. This is cheap to apply. The class is coarse, and the audit has to test whether it admits non-physical or administrative entities.
- **B. Wikidata classes.** Keep entities that pass the type test below and whose instance-of (`P31`) or subclass-of (`P279`) claims place them under a pinned list of place classes (bucket 4). The predicate reads only fields that the pinned DaMuEL KB provides. Each claim is a positional array of a datatype tag, a value, and optional qualifiers, keyed by property ID. An accepted claim is a `P31` or `P279` claim whose datatype is `wikibase-item`. Qualifiers and references are not used. Membership has a base case and a transitive step. An entity passes when one of its accepted `P31` values is itself a listed class, or when a listed class is reached from one of its accepted `P31` values through one or more accepted `P279` claims. The search follows accepted `P279` claims with no depth limit and a visited set against cycles. The KB has no rank field, so competing and deprecated claims are all accepted. This is a known limit, and the audit counts entities whose accepted `P31` or `P279` claims disagree. Option B does not run until two preconditions hold. (1) The sample check confirms that the pinned records carry `P31` and `P279` claims, which the schema does not name. (2) The class list is pinned: a set of Wikidata item IDs, reviewed by a person and committed with its review date. It is not written yet. *Rank rule, dependent on the source.* The paper's text says the KB keeps all claims and ranks, but the schema has no rank field, so this is unverified. The rank rule applies only if a rank field is verified on the pinned records, or a rank-bearing Wikidata snapshot is pinned with its own digest. Neither is pinned here. If the rule applies, an entity's accepted values for a property are those of its `preferred` statements if it has any, and otherwise those of its `normal` statements. `deprecated` statements are never accepted, as in Wikidata's ranking help. Otherwise the rule is void, and the predicate above is the whole rule. Gazetteer mapping and coordinates are not part of this predicate. They are tested only for resolution, in buckets 5 to 7.
- **C. Audited rule.** Start from A or B, then correct the rule on a stratified sample. A correction to the KB-based rule is counted in bucket 4.

**Type test (fixed for every option, bucket 3).** An entity passes when the string `LOC` is an element of its `named_entities.type` array. The test is membership, not equality with a one-element array, because the array can hold several classes. An entity that carries `LOC` together with other classes passes, so a place is not lost because its KB entry also carries another class. Mentions of such multi-typed entities are counted separately, so the audit can see how many of them admit an entity with another class. An entity with no `named_entities` block, or with an empty `type` array, does not pass.

Proposal, not decided: option B with an audit of a stratified sample. Decide these before any adapter exists:

1. Which option to use.
2. Whether administrative units (for example, a country, or a municipality) count as place mentions.

Metonymic uses are not an open question, because no option can separate them. A country name standing for its national team shares its QID with the literal country, and the KB and the pinned snapshot are entity-level. Gold therefore counts a metonymic mention as a place mention, and the validation sample reports the metonymic share as a rate. This is a limitation of the plan. An owner can change it only by adding a mention-level annotation with its own bucket.

## Gazetteer and coordinate pins (decision needed)

Resolution needs a gazetteer. The repository's pre-configured gazetteers are GeoNames, with `geonameid` identifiers, and SwissNames3D. A QID-to-gazetteer mapping must be pinned, with its source and snapshot. The KB schema allows `external-id` values, so a GeoNames identifier claim might be extractable from the Wikidata dump. That is not verified here. Unmappable QIDs, entities without coordinates, and granularity mismatches must each be reported as separate counts, not dropped silently. The accounting below gives their order. Decide the gazetteer, the mapping source, and the coordinate snapshot before any run.

## Mention accounting (policy)

Every count starts from all mentions in `wiki.links`, not from the mentions that already carry a QID. Only flat mentions are scored. The plan treats them as the non-overlapping selection, and the schema section asks for a sample check of that before any run. Overlap-aware scoring, where one span can match several overlapping gold mentions, would need a matching rule that this plan does not define, so it is not used. The published text schema requires `start`, `end`, `title`, and `origin` for each link, but not `qid`. The schema does not say why a QID can be absent; one possible cause is an annotation that was not resolved upstream. Such a mention has no KB entry to type and no QID to map. Its `title` is not used to recover a QID, because that would be a new resolver, and this plan does not authorize one. If a recovery method is added later, its recovered mentions get their own count and are not merged into `eligible`.

Records without a `wiki` block have no article text and no `wiki.links`, so they carry no mentions. The adapter tests for the `wiki` key before it reads `wiki.links`, and it skips such a record without raising an error. Each skipped record is counted per language as `no_wiki_block`. That is a record count, not a mention count. It sits beside the number of records that have a `wiki` block, so the two sum to the language's record total. No mention bucket changes because of this rule.

Policy for mentions with unknown place status:

A mention has unknown place status when it has no QID, or when its QID has no entry in the pinned KB snapshot. Such a mention cannot be typed from the pinned snapshot, and this plan does not authorize recovering its type from another source. Recovering it would need a second snapshot and its own pin.

- They are never typed, mapped, or guessed into the place set, and they are never dropped without a count.
- Flat mentions with no QID are reported as `no_qid`. Flat mentions whose QID is absent from the pinned KB are reported as `qid_not_in_kb` (bucket 2). Both are counted per language and per `origin`, beside every denominator they leave out. Non-flat mentions of either kind are counted under `non_flat`.
- A prediction is set aside only when all three hold: it is valid, its character span is exactly equal to the span of an unknown-status mention (flat or not), and no mention whose QID is in the pinned KB, flat or not, has that same span. A set-aside prediction is neither a true positive nor a false positive, because its place status is unknown. It is reported per language as `predicted_on_no_qid` when a no-QID mention has that span, and otherwise as `predicted_on_qid_not_in_kb`.
- Unknown-status mentions are not recognition gold and not resolution gold, so a missed one is neither a false negative nor a true positive. A valid prediction on one therefore cannot lower precision, and the set-aside counts show how many predictions this choice removed from the score.
- Every other valid prediction is scored as usual against flat gold. A prediction that only overlaps an unknown-status span, including one that is wider, narrower, or shifted at a boundary, is a false positive unless its span exactly matches a flat gold span. Invalid outputs remain false positives. A valid false positive whose span exactly matches a non-flat mention is also counted as `predicted_on_non_flat`, per language. That count shows how many false positives come from the flat-only rule, which penalises a valid place prediction when the dataset marks another mention as dominant.

The buckets below are applied in this order. Bucket 0 applies first to every mention, and buckets 1 to 8 apply only to flat mentions. Each mention falls in exactly one bucket, so the counts sum to the total. Every rejection has a bucket: bucket 0 is an overlap rejection, buckets 1 and 2 are missing KB data, bucket 3 is the type test alone, and bucket 4 takes every other rejection by the KB-based predicate. Buckets 0 to 4 are therefore never recognition gold. Buckets 5 to 7 are rejections from resolution by the gazetteer-based tests, and they stay recognition gold.

0. `non_flat`: the mention's `flat` flag is not true. Only flat mentions are scored, so this mention is not gold and is not placed in buckets 1 to 8. It is counted per language and per `origin`.
1. `no_qid`: the link has no `qid`.
2. `qid_not_in_kb`: the QID has no entry in the pinned KB snapshot, so it has no type. It is not recognition gold, and its valid predictions are set aside under the unknown-status policy below, not scored as false positives.
3. `not_place`: the entity fails the type test.
4. `filter_rejected`: the entity passes the type test but fails the rest of the selected KB-based predicate. Under option A this bucket is always empty. Under option B, the entity's instance-of or subclass claims fall outside the pinned place classes. Under option C, the audited correction excludes it. Any further test that option C adds needs its own named bucket in this list before the audit is run.
5. `place_unmapped`: the QID has no gazetteer ID in the pinned mapping.
6. `place_no_coordinates`: the gazetteer ID has no coordinates in the pinned snapshot. This is the only coordinate test, and it applies under every option. It is not a place test. It removes the mention from the distance mask `M_coord`, defined below, and leaves it in recognition gold.
7. `granularity_mismatch`: the granularity rule, defined with the gazetteer, rejects the match.
8. `eligible`: the remaining mentions.

`place_multi_typed` is a sub-count of flat mentions in buckets 4 to 8: the mentions that pass the type test and whose entity has `LOC` and at least one other class. Recognition gold is the union of buckets 5 to 8, the flat mentions that pass the KB-based predicate, whether or not they map. The gazetteer only divides that union into buckets, so recognition gold does not depend on the mapping. Flat mentions are assumed not to overlap, so each gold span has one mention, one gold QID, and at most one mapped gazetteer ID.

Two resolution masks are defined here, and the evaluation design refers to them by these names only. **`E_id` (exact-ID eligible)** is the flat mentions in bucket 8, and it is used for exact-ID accuracy and candidate recall only. Each `E_id` span carries its mapped canonical gazetteer identifier, the ID that the pinned QID-to-gazetteer mapping gives it. That gazetteer ID is the only ID scored for candidate recall, exact-ID accuracy, and end-to-end, and it is the namespace the resolver returns, whether GeoNames or SwissNames3D. The QID stays as source provenance and is never compared with resolver output. Bucket 5 has no gazetteer ID, bucket 6 has an ID that the granularity test has not checked, and bucket 7 has an ID that the granularity rule rejects, so none of them is in `E_id`. **`M_coord` (coordinate eligible)** is the set of gold spans whose validated gold gazetteer ID has coordinates in the pinned snapshot, where "validated" means the ID passed buckets 5 to 7. Distance accuracy is computed over `M_coord`. Bucket 6 has no coordinates and bucket 7 has an ID that the granularity rule rejects, so both are outside `M_coord`. At present `E_id` and `M_coord` both select bucket 8. They are kept as separate masks so that each denominator is named. Coverage, abstention, and invalid-output rates use all gold spans, which are buckets 5 to 8, because every gold span is resolved, abstained, or invalid. The counts of buckets 5 to 7 are reported beside every resolution denominator. End-to-end uses the article-level slice described in the evaluation design below, whose gold spans all lie in `E_id`.

## Evaluation design (proposal, not run)

Keep the three tasks of the [public benchmark protocol](benchmark-protocol.md) separate, and record annotation quality as `silver`:

- **Recognition.** Exact, deduplicated character spans. Gold is the flat place mentions in buckets 5 to 8 of the mention accounting. Non-flat mentions are not gold. Invalid outputs count as false positives. A valid prediction on an unknown-status span is set aside only under the rule in the mention accounting policy.
- **Gold-span resolution.** Gold spans are the flat mentions in buckets 5 to 8, and they are passed to the resolver. Each keeps its QID as provenance, and each `E_id` span carries the canonical gazetteer ID used for scoring. Score only on the masks defined in the mention accounting policy: exact-ID accuracy on `E_id`; candidate recall before ranking on `E_id`, meaning the share of `E_id` spans whose gold canonical gazetteer ID is among the retrieved candidates, with retrieval failures counted as misses rather than excluded; exact-ID successes cannot exceed candidate hits, so a correct final gazetteer ID must first be retrieved; coverage, abstention, and invalid-output rates over all gold spans, buckets 5 to 8; and distance accuracy at inclusive 1 km, 10 km, and 50 km thresholds on `M_coord`. Report original (`wiki`) and expanded spans separately. Keep the eligible denominators visible.
- **End-to-end.** A true positive needs both the span and the canonical gazetteer ID to match. Use an article-level slice, so that every gold span in it has an eligible ID. An article is included only when every flat gold mention in buckets 5 to 8 of that article is in `E_id`. An article with a flat gold mention in buckets 5 to 7 is excluded as a whole, not filtered mention by mention, so no true place in an included article goes unscored. Articles with no flat gold mention are included, and predictions on them count as false positives. The selection uses gold annotations only, never predictions, and is fixed before any output is seen. The number of excluded articles, and their gold mentions per bucket, are reported beside the slice.

Slices and checks, to be fixed before any output is seen:

- Hold out whole articles by a seeded hash of the QID. No threshold, label, prompt, or example is chosen on the evaluated articles.
- Draw a validation sample of flat mentions, stratified by language and by `origin` (original `wiki` mentions against each expanded origin). A person checks span boundaries and QIDs on that sample, and labels each mention as literal or metonymic. The metonymic share is reported as a rate beside the scores. It is not a bucket and does not change gold. Report the agreement before trusting any aggregate score.
- Report per-language sample sizes and a macro summary over languages, with the uncertainty method from the protocol. Report throughput separately.
- Record the training overlap of each pinned model as `known`, `unknown`, or `verified_absent`. Absent a disclosure, record `unknown`. The DaMuEL text is Wikipedia from 2022, and the overlap question is open until the model cards or an audit say otherwise.

The sample sizes, the number of languages, and the thresholds are not chosen here.

## Costs and storage

- Download sizes are large. The English archive alone is listed at 24.5 GB, and the Wikidata archive is 2,715,955,200 bytes. Keep any extraction on the HDD, under the project's working storage, and delete intermediate files as soon as a run completes.
- Any run needs a pinned revision for every artifact, under the protocol's digest rules.

## Decisions needed before implementation

1. **Scope.** The issue says that opening it does not authorize implementation, model inference, or dataset uploads. This branch adds only the documentation and the offline coverage report. Confirm whether the adapter and evaluator should be built next.
2. **Geographic filter.** Choose option A, B, or C, and answer the administrative-units question above. Metonymic uses are counted as place mentions, as stated under the options. For option B, the predicate reads only the pinned KB fields. The rank rule waits on a verified rank field or a pinned rank-bearing snapshot. The place-class list still needs review and commit.
3. **Gazetteer and pins.** Choose the gazetteer and the QID mapping source, and pin the coordinate snapshot.
4. **Slices and sizes.** Choose the languages to include, the held-out fraction, and the validation sample size.
5. **Canonical list.** Confirm how to treat `no`, `nn`, `hr`, and the script variants of `sr` and `zh`.

## Not done

- No adapter, mention-span mapper, or evaluator.
- No geographic filter implementation and no gazetteer or coordinate pin.
- No validation sample and no measurement of annotation accuracy.
- No model run, benchmark, or recommendation on usefulness, coverage, mapping loss, or cost.
- No download of any language archive or of the full Wikidata knowledge base. The Wikidata archive was read only at its tail.
