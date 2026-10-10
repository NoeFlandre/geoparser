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
uv run python -m scripts.damuel_inventory
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

- **Text archives** hold one JSON object per Wikipedia article in JSON Lines, compressed into `part-*.xz` files. Each object has `qid`, `lang`, `label`, `aliases`, `description`, and a `wiki` block with `text`, `tokens`, `sentences`, `paragraphs`, `redirects`, `anchors`, `links`, and `removed_links`.
- **Token offsets.** `wiki.links` gives `start` and `end` as token indices, not character offsets. Each token has its own `start` and `end`, which the schema describes as positions in the `text` property. Confirm on a sample that these are Python character offsets before any span mapping relies on them.
- **Original versus expanded links.** Each link has an `origin`. The value `wiki` marks a link present in the Wikipedia article. The other values (`title`, `label`, `alias`, `redirects`, `anchors`, and their `_lemmatized` forms) mark mentions added by the authors. The `flat` flag marks a link as the dominant one where no word belongs to more than one link.
- **Expansion rule.** The paper says Wikipedia annotates only one mention per entity on a page. The authors therefore add further mentions only of entities already linked on that same page, so expansion adds no new entity to a page. The paper reports no precision or recall for the expanded mentions, so their accuracy is unknown.
- **Text processing.** The paper says templates, lists, tables, images, galleries, references, category links, and inline formatting were removed. Mention offsets refer to this processed text, not to the rendered article.
- **Knowledge base.** Entities are keyed by Wikidata QID. Claims use typed values, including `wikibase-item` and `globe-coordinate`. `named_entities.type` is an array whose elements are drawn from seven automatic classes: PER, ORG, LOC, EVENT, BRAND, WORK_OF_ART, and MANUFACTURED. Figure 2 of the paper declares it as an array of these enum values and sets no maximum length, so one entity can carry several classes. The paper says the classes come from the Wikidata class hierarchy. The `schema.json` lists `fictional` outside its `properties` block, so the schema does not validate it. Check it on data before use.
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

The issue requires a documented filter that separates physical places from people, organisations, metonymic references, and other non-place entities. It also rules out treating coordinates as sufficient. The schema does not settle which entities count as places, and this plan does not choose for you. The options are:

- **A. Type only.** Keep mentions whose KB entity passes the type test below. This is cheap to apply. The class is coarse, and the audit has to test whether it admits non-physical or administrative entities.
- **B. Wikidata classes plus coordinates.** Keep entities that pass the type test below, whose instance-of or subclass claims fall under a pinned list of place classes, and whose coordinates are present in the pinned snapshot. The list is human-reviewed, and the 2022-11-09 dump supplies the claims.
- **C. Audited rule.** Start from A or B, then correct the rule on a stratified sample.

**Type test (fixed for every option).** An entity passes when the string `LOC` is an element of its `named_entities.type` array. The test is membership, not equality with a one-element array, because the array can hold several classes. An entity that carries `LOC` together with other classes passes, so a place is not lost because its KB entry also carries another class. Mentions of such multi-typed entities are counted separately, so the audit can see how many of them admit an entity with another class. An entity with no `named_entities` block, or with an empty `type` array, does not pass.

Proposal, not decided: option B with an audit of a stratified sample. Decide these before any adapter exists:

1. Which option to use.
2. Whether administrative units (for example, a country, or a municipality) count as place mentions.
3. Whether metonymic uses (for example, a country name standing for its national team) count as place mentions, or are excluded and reported as a separate category.

## Gazetteer and coordinate pins (decision needed)

Resolution needs a gazetteer. The repository's pre-configured gazetteers are GeoNames, with `geonameid` identifiers, and SwissNames3D. A QID-to-gazetteer mapping must be pinned, with its source and snapshot. The KB schema allows `external-id` values, so a GeoNames identifier claim might be extractable from the Wikidata dump. That is not verified here. Unmappable QIDs, entities without coordinates, and granularity mismatches must each be reported as separate counts, not dropped silently. The accounting below gives their order. Decide the gazetteer, the mapping source, and the coordinate snapshot before any run.

## Mention accounting (policy)

Every count starts from all mentions in `wiki.links`, not from the mentions that already carry a QID. The published text schema requires `start`, `end`, `title`, and `origin` for each link, but not `qid`. The schema does not say why a QID can be absent; one possible cause is an annotation that was not resolved upstream. Such a mention has no KB entry to type and no QID to map. Its `title` is not used to recover a QID, because that would be a new resolver, and this plan does not authorize one. If a recovery method is added later, its recovered mentions get their own count and are not merged into `eligible`.

Policy for mentions without a QID:

- They are never typed, mapped, or guessed into the place set, and they are never dropped without a count.
- They are reported as `no_qid`, per language and per `origin`, beside every denominator they leave out.
- A valid prediction that overlaps one is neither a true positive nor a false positive. It is reported as `predicted_on_no_qid`, because its place status is unknown. Invalid outputs remain false positives.

The buckets below are applied in this order. Each mention falls in exactly one bucket, so the counts sum to the total:

1. `no_qid`: the link has no `qid`.
2. `qid_not_in_kb`: the QID has no entry in the pinned KB snapshot, so it has no type.
3. `not_place`: the entity fails the type test.
4. `place_unmapped`: the QID has no gazetteer ID in the pinned mapping.
5. `place_no_coordinates`: the gazetteer ID has no coordinates in the pinned snapshot.
6. `granularity_mismatch`: the granularity rule, defined with the gazetteer, rejects the match.
7. `eligible`: the remaining mentions.

`place_multi_typed` is a sub-count of buckets 4 to 7: the mentions whose entity has `LOC` and at least one other class. Recognition gold is every mention in buckets 4 to 7, whether or not it maps, because recognition needs no gazetteer. Gold-span resolution scores its accuracy on bucket 7, and reports the counts of buckets 4 to 6 beside it, so the mentions that cannot be mapped stay visible. End-to-end uses only bucket 7, as the slice described in the evaluation design below.

## Evaluation design (proposal, not run)

Keep the three tasks of the [public benchmark protocol](benchmark-protocol.md) separate, and record annotation quality as `silver`:

- **Recognition.** Exact, deduplicated character spans. Gold is the place mentions in buckets 4 to 7 of the mention accounting. Invalid outputs count as false positives.
- **Gold-span resolution.** Gold spans from the annotations are passed to the resolver. Report original (`wiki`) and expanded spans separately. Report exact-ID accuracy where a mapping exists, distance accuracy at 1 km, 10 km, and 50 km, abstention, and coverage. Keep the eligible denominators visible.
- **End-to-end.** A true positive needs both the span and the canonical ID to match. Use a slice where every gold span has an eligible ID.

Slices and checks, to be fixed before any output is seen:

- Hold out whole articles by a seeded hash of the QID. No threshold, label, prompt, or example is chosen on the evaluated articles.
- Draw a validation sample stratified by language and by `origin` (original `wiki` links against each expanded origin). A person checks span boundaries and QIDs on that sample. Report the agreement before trusting any aggregate score.
- Report per-language sample sizes and a macro summary over languages, with the uncertainty method from the protocol. Report throughput separately.
- Record the training overlap of each pinned model as `known`, `unknown`, or `verified_absent`. Absent a disclosure, record `unknown`. The DaMuEL text is Wikipedia from 2022, and the overlap question is open until the model cards or an audit say otherwise.

The sample sizes, the number of languages, and the thresholds are not chosen here.

## Costs and storage

- Download sizes are large. The English archive alone is listed at 24.5 GB, and the Wikidata archive is 2,715,955,200 bytes. Keep any extraction on the HDD, under the project's working storage, and delete intermediate files as soon as a run completes.
- Any run needs a pinned revision for every artifact, under the protocol's digest rules.

## Decisions needed before implementation

1. **Scope.** The issue says that opening it does not authorize implementation, model inference, or dataset uploads. This branch adds only the documentation and the offline coverage report. Confirm whether the adapter and evaluator should be built next.
2. **Geographic filter.** Choose option A, B, or C, and answer the three questions above.
3. **Gazetteer and pins.** Choose the gazetteer and the QID mapping source, and pin the coordinate snapshot.
4. **Slices and sizes.** Choose the languages to include, the held-out fraction, and the validation sample size.
5. **Canonical list.** Confirm how to treat `no`, `nn`, `hr`, and the script variants of `sr` and `zh`.

## Not done

- No adapter, mention-span mapper, or evaluator.
- No geographic filter implementation and no gazetteer or coordinate pin.
- No validation sample and no measurement of annotation accuracy.
- No model run, benchmark, or recommendation on usefulness, coverage, mapping loss, or cost.
- No download of any language archive or of the full Wikidata knowledge base. The Wikidata archive was read only at its tail.
