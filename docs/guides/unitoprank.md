# UniTopRank baseline

UniTopRank (Hu et al., *UniTopRank: A scalable and language-independent method for toponym resolution*, IJGIS 2026) is a rule-based toponym ranker. It scores each candidate by name similarity, administrative level, population, and spatial coherence with the other toponyms of the document. It needs no neural model and no training.

This page covers what the repository can do with it now: pin the reviewed code, map gazetteer candidates onto the ranker's input, and run the ranker on a local checkout. It does not run the held-out comparison. That run needs separate approval, and it is not part of this change.

## Pinned code

| | |
| --- | --- |
| Repository | [`dlr-dw/UniTopRank`](https://gitlab.com/dlr-dw/UniTopRank) on GitLab |
| Commit | `346deb166f12b0c97c8c0f9759b593ff178ceafd`, the head of `main` when it was reviewed |
| Tags or releases | None. The commit is the pin. |
| Licence | Apache-2.0, from the `LICENSE` file at the pinned commit |
| Release metadata | [figshare release 30445541](https://figshare.com/articles/software/UniTopRank/30445541), version 10, DOI `10.6084/m9.figshare.30445541.v10`, published 2025-12-15, licence field "Apache 2.0" |

The module `scripts/unitoprank_benchmark/pins.py` pins 15 files by their Git blob IDs, which are the IDs GitLab's tree API reports for the commit. A checkout is checked before anything is imported:

```bash
git clone https://gitlab.com/dlr-dw/UniTopRank /path/to/UniTopRank
git -C /path/to/UniTopRank checkout 346deb166f12b0c97c8c0f9759b593ff178ceafd
uv run python -m scripts.unitoprank_benchmark --checkout /path/to/UniTopRank
```

The command reads files and prints a report. It opens no connection and imports nothing from the checkout. It exits with status 2 and names each missing or changed file when the tree does not match the pin.

### Third-party licences, as recorded upstream

These are copied from the pinned commit's `THIRD_PARTY_LICENSES.md` and `requirements.txt`. They were not re-checked against the packages' own licence files.

| Package | Pinned version | Licence as recorded upstream |
| --- | --- | --- |
| numpy | 2.4.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| rapidfuzz | 3.14.3 | MIT |
| requests | 2.32.5 | Apache Software License |

The figshare copy of `requirements.txt` is not pinned. It lists fourteen packages without versions, among them `nltk`, `pandas`, `geopandas` and `fiona`. The GitLab file is the one this pin uses.

## What the review found

The reviewed code is the GitLab tree at the pinned commit. The figshare release carries a second copy of the ranker and some evaluation scripts. The findings below were checked by running the pinned code or by comparing the two copies.

- **`geo_rank_api.py` does not import.** Both of its import branches name a package `unitoprank`, but the package is `unitorank`. The README example `from geo_rank_api import resolve_toponyms` fails with `ModuleNotFoundError: No module named 'unitoprank'`. The adapter imports `unitorank.ranker` directly, so this does not affect it.
- **The ranker imports a module from the repository root.** `unitorank/ranker.py` imports `thread_weight_rank_algorithm_3_beam`, so the checkout root must be on the import path. The loader puts it there.
- **Two copies of the ranking code exist.** The GitLab `thread_weight_rank_algorithm_3_beam.py` differs from the figshare copy in one place: a pure-Python fallback for `rapidfuzz`, at lines 7 to 36. The ranking is otherwise the same. The pin uses the GitLab copy.
- **The figshare evaluation scripts make network calls.** `util.py` downloads a Natural Earth shapefile, and queries Photon, Nominatim, Wikidata and a GeoNames service on `localhost`. `rank_evaluation.py` imports it. The adapter uses neither file, and neither is pinned.
- **`evaluation.py` (figshare) imports `numpy.trapz`,** which the NumPy 2.5.3 installed here does not provide. It was not tried on NumPy 2.4.2.
- **Distances are approximate.** `min_fast_distance_to_group` measures distance in degrees and multiplies by 111 km, a planar approximation. At high latitudes that overstates east-west distance. The haversine helper in the same file is not used by the scoring path.
- **Ranking is order-dependent.** Toponyms are scored in text order. After each one, its candidate list is cut to the top few distinct addresses before the next toponym uses it as spatial evidence. A toponym's score therefore depends on the toponyms before it.
- **Ties follow input order.** The ranker sorts stably, so an exact score tie goes to the candidate offered first. The adapter offers candidates in identifier order, so ties resolve to the lower identifier.

## Mapping gazetteer candidates

UniTopRank ranks the candidates it is given and does not retrieve any. `scripts/unitoprank_benchmark/candidates.py` turns gazetteer candidates into its input. The rules are:

| Input | Handling | Counted in |
| --- | --- | --- |
| Surface form | Stripped and lower-cased, as the ranker does. Surfaces that normalize alike share one list. | none |
| Candidates | Offered in identifier order, whatever order the gazetteer returned them in. | none |
| Repeated identifier | Kept once. | `dropped.duplicate_identifier` |
| Missing or invalid coordinates | Dropped. Latitude must be in [-90, 90], longitude in [-180, 180], both finite. | `dropped.no_coordinates` |
| Blank name | Dropped. | `dropped.no_name` |
| Two candidates with the same address string | The lower identifier is kept. The ranker would merge them anyway. | `dropped.address_collision` |
| Missing, `None`, or negative population | Scored as 0. | `missing.population` |
| Missing administrative level | Left out of the address. The ranker's hierarchy score then works on a shorter address. | `missing.admin_level`, and `missing.admin_path` when there is no path at all |
| Missing feature code | Sent as an empty `admin_level`, which the ranker scores as 0. | `missing.feature_code` |
| Alternate names | Stripped, deduplicated, sorted. | none |

The candidate set is built from names and admin paths the caller supplies. The adapter does not look up administrative names from codes. A gazetteer that stores only codes needs a lookup before the admin path can be filled.

### Candidate retrieval differences

- **UniTopRank's full pipeline** retrieves candidates through `CandidateRetriever`, which calls a GeoNames service and a Photon service. Their defaults are `localhost:8091` and `localhost:2322`. Its retrieval set therefore depends on those services, not on the geoparser gazetteer.
- **geoparser's pipeline** retrieves candidates from the installed gazetteer, with exact, phrase, partial and fuzzy search, widened over up to three tiers.
- **This adapter** uses neither service. It offers UniTopRank the candidates that the caller passes in. A common-candidate comparison gives both systems the same set. A native-pipeline comparison uses each system's own retrieval, and it must be reported separately from the common one.
- **Multilingual coverage** is not guaranteed by language-independent ranking. The ranker can only place a toponym whose name or alternate name is in the candidate set, so coverage depends on the gazetteer's names in each language.

## Running the offline tests

The unit tests need no checkout and make no network calls. Every test in `tests/unit/test_unitoprank_benchmark` runs with external sockets disabled.

```bash
uv run pytest tests/unit/test_unitoprank_benchmark -o addopts="" -p no:cacheprovider
```

`test_upstream.py` also runs the pinned ranker on a three-place example. It needs the checkout from the commands above, and is skipped without it:

```bash
UNITORANK_CHECKOUT=/path/to/UniTopRank uv run pytest tests/unit/test_unitoprank_benchmark/test_upstream.py -o addopts="" -p no:cacheprovider
```

CI runs this test as well. The Ubuntu, Python 3.12 test cell and the quality gauntlet each clone the pinned commit into the runner's temporary directory and set `UNITORANK_CHECKOUT`. The other test cells skip the test by design, because they do not fetch the checkout. Without that setting the test is skipped, and its lines would count as uncovered in the CRAP gate. The workflow steps and `tests/unit/test_quality/test_unitoprank_ci_checkout.py` keep the fetched commit equal to the one in `pins.py`.

## Not done

- No held-out public-corpus run, and so no accuracy, distance, coverage or runtime figures for UniTopRank.
- No figshare `data.zip`, no evaluation scripts, no Natural Earth data, and no gazetteer snapshot were downloaded or pinned.
- No native-pipeline candidate retrieval was run. The adapter does not connect to the GeoNames or Photon services.
- No scoring module exists yet. The mapping and ranking are checked by the tests, not against gold spans.
