# PAN-X / WikiANN place recognition

**Provenance warning (1 October 2026):** the declared source commit below cannot
be retrieved from this repository or GitHub. These historical sample results
are retained for traceability, but are not reproducible evidence and must not
be used to select a pipeline. The original timing also includes metric
bookkeeping; throughput is not inference-only. No replacement commit or
corrected timing is fabricated. A future validated run must supersede this
sample before scientific use.

**Feasibility sample only:** metrics below are descriptive of the bounded prefix, not a full-test quality comparison.

- Repository commit: `dd3a7e78836f9560fe24f7546b3ecc0c202a0fa7`
- Dataset: `unimelb-nlp/wikiann` at `f0a3be6dc5564c0cc4150bb660144800a1f539d4`, split `test`
- Test intersection: 82 languages, 423,100 rows; this run used 656 rows
- Target languages absent from this test split: `ha`, `xh`, `zu`
- Hardware: `Linux-6.18.44-x86_64-with-glibc2.41`, 4 CPU threads, CUDA available: `False`
- Seed: `0`; batch size: `8`; no training or fine-tuning
- Dataset acquisition/materialization: 300.04s

The canonical 85-code list is copied from the upstream sentence-splitting guide at revision `c6b503908b4687517f546cecce40c621d4ee56ac`. The exact upstream source is [docs/sentence-splitting.md](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/c6b503908b4687517f546cecce40c621d4ee56ac/docs/sentence-splitting.md).

## Model summary

| Recognizer | Documented language count | Evaluated sentences | Macro P | Macro R | Macro F1 | Steady sentences/s | Checkpoint fetch s | Model load s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `en_core_web_sm` | 1 | 8 | 0.5000 | 0.5000 | 0.5000 | 515.6239 | 0.0000 | 2.3881 |
| `fastino/gliner2.5-multi-v1` | unspecified | 656 | 0.3220 | 0.5099 | 0.3758 | 6.3031 | 13.2817 | 7.2064 |
| `Davlan/xlm-roberta-base-ner-hrl` | 10 | 656 | 0.0629 | 0.0847 | 0.0694 | 21.2096 | 30.3053 | 2.7096 |

Model fetch time, local model load time, warmup, data acquisition, and steady inference are reported separately. Throughput excludes warmup and data loading. The GLiNER threshold is fixed at 0.5; no test-set tuning was done.

## Coverage and overlap

### `en_core_web_sm`

- Revision: `3.8.0`; model card: [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm-accuracy)
- Language coverage: English-only upstream reference model.
- Training data: spaCy English small model 3.8.0; its model metadata identifies OntoNotes 5 as its training source.
- WikiANN overlap: The model metadata does not report WikiANN training. Possible text overlap with Wikipedia was not audited.
- Location mapping: FAC, GPE, LOC -> WikiANN LOC
- Warmup: 8 sentences in 0.0293s; steady inference: 0.0155s
- Linear full-matrix inference estimate from this run: 19.3940s

| Language | Evaluation status | Coverage status | Sentences | Gold LOC | P | R | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `af` | not_evaluated_english_only | False | — | — | — | — | — |
| `am` | not_evaluated_english_only | False | — | — | — | — | — |
| `ar` | not_evaluated_english_only | False | — | — | — | — | — |
| `az` | not_evaluated_english_only | False | — | — | — | — | — |
| `be` | not_evaluated_english_only | False | — | — | — | — | — |
| `bg` | not_evaluated_english_only | False | — | — | — | — | — |
| `bn` | not_evaluated_english_only | False | — | — | — | — | — |
| `ca` | not_evaluated_english_only | False | — | — | — | — | — |
| `ceb` | not_evaluated_english_only | False | — | — | — | — | — |
| `cs` | not_evaluated_english_only | False | — | — | — | — | — |
| `cy` | not_evaluated_english_only | False | — | — | — | — | — |
| `da` | not_evaluated_english_only | False | — | — | — | — | — |
| `de` | not_evaluated_english_only | False | — | — | — | — | — |
| `el` | not_evaluated_english_only | False | — | — | — | — | — |
| `en` | evaluated | documented | 8 | 4 | 0.5000 | 0.5000 | 0.5000 |
| `eo` | not_evaluated_english_only | False | — | — | — | — | — |
| `es` | not_evaluated_english_only | False | — | — | — | — | — |
| `et` | not_evaluated_english_only | False | — | — | — | — | — |
| `eu` | not_evaluated_english_only | False | — | — | — | — | — |
| `fa` | not_evaluated_english_only | False | — | — | — | — | — |
| `fi` | not_evaluated_english_only | False | — | — | — | — | — |
| `fr` | not_evaluated_english_only | False | — | — | — | — | — |
| `fy` | not_evaluated_english_only | False | — | — | — | — | — |
| `ga` | not_evaluated_english_only | False | — | — | — | — | — |
| `gd` | not_evaluated_english_only | False | — | — | — | — | — |
| `gl` | not_evaluated_english_only | False | — | — | — | — | — |
| `gu` | not_evaluated_english_only | False | — | — | — | — | — |
| `he` | not_evaluated_english_only | False | — | — | — | — | — |
| `hi` | not_evaluated_english_only | False | — | — | — | — | — |
| `hu` | not_evaluated_english_only | False | — | — | — | — | — |
| `hy` | not_evaluated_english_only | False | — | — | — | — | — |
| `id` | not_evaluated_english_only | False | — | — | — | — | — |
| `ig` | not_evaluated_english_only | False | — | — | — | — | — |
| `is` | not_evaluated_english_only | False | — | — | — | — | — |
| `it` | not_evaluated_english_only | False | — | — | — | — | — |
| `ja` | not_evaluated_english_only | False | — | — | — | — | — |
| `jv` | not_evaluated_english_only | False | — | — | — | — | — |
| `ka` | not_evaluated_english_only | False | — | — | — | — | — |
| `kk` | not_evaluated_english_only | False | — | — | — | — | — |
| `km` | not_evaluated_english_only | False | — | — | — | — | — |
| `kn` | not_evaluated_english_only | False | — | — | — | — | — |
| `ko` | not_evaluated_english_only | False | — | — | — | — | — |
| `ku` | not_evaluated_english_only | False | — | — | — | — | — |
| `ky` | not_evaluated_english_only | False | — | — | — | — | — |
| `la` | not_evaluated_english_only | False | — | — | — | — | — |
| `lt` | not_evaluated_english_only | False | — | — | — | — | — |
| `lv` | not_evaluated_english_only | False | — | — | — | — | — |
| `mg` | not_evaluated_english_only | False | — | — | — | — | — |
| `mk` | not_evaluated_english_only | False | — | — | — | — | — |
| `ml` | not_evaluated_english_only | False | — | — | — | — | — |
| `mn` | not_evaluated_english_only | False | — | — | — | — | — |
| `mr` | not_evaluated_english_only | False | — | — | — | — | — |
| `ms` | not_evaluated_english_only | False | — | — | — | — | — |
| `mt` | not_evaluated_english_only | False | — | — | — | — | — |
| `my` | not_evaluated_english_only | False | — | — | — | — | — |
| `ne` | not_evaluated_english_only | False | — | — | — | — | — |
| `nl` | not_evaluated_english_only | False | — | — | — | — | — |
| `no` | not_evaluated_english_only | False | — | — | — | — | — |
| `pa` | not_evaluated_english_only | False | — | — | — | — | — |
| `pl` | not_evaluated_english_only | False | — | — | — | — | — |
| `ps` | not_evaluated_english_only | False | — | — | — | — | — |
| `pt` | not_evaluated_english_only | False | — | — | — | — | — |
| `ro` | not_evaluated_english_only | False | — | — | — | — | — |
| `ru` | not_evaluated_english_only | False | — | — | — | — | — |
| `si` | not_evaluated_english_only | False | — | — | — | — | — |
| `sk` | not_evaluated_english_only | False | — | — | — | — | — |
| `sl` | not_evaluated_english_only | False | — | — | — | — | — |
| `sq` | not_evaluated_english_only | False | — | — | — | — | — |
| `sr` | not_evaluated_english_only | False | — | — | — | — | — |
| `sv` | not_evaluated_english_only | False | — | — | — | — | — |
| `ta` | not_evaluated_english_only | False | — | — | — | — | — |
| `te` | not_evaluated_english_only | False | — | — | — | — | — |
| `tg` | not_evaluated_english_only | False | — | — | — | — | — |
| `th` | not_evaluated_english_only | False | — | — | — | — | — |
| `tr` | not_evaluated_english_only | False | — | — | — | — | — |
| `uk` | not_evaluated_english_only | False | — | — | — | — | — |
| `ur` | not_evaluated_english_only | False | — | — | — | — | — |
| `uz` | not_evaluated_english_only | False | — | — | — | — | — |
| `vi` | not_evaluated_english_only | False | — | — | — | — | — |
| `yi` | not_evaluated_english_only | False | — | — | — | — | — |
| `yo` | not_evaluated_english_only | False | — | — | — | — | — |
| `zh` | not_evaluated_english_only | False | — | — | — | — | — |

### `fastino/gliner2.5-multi-v1`

- Revision: `2ca71aafb3446d9014e1c55c7ff51c9bc7209c47`; model card: [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1)
- Language coverage: Model card describes the checkpoint as multilingual but does not publish an exhaustive language list.
- Training data: The pinned model card does not identify the training corpora.
- WikiANN overlap: Whether training included WikiANN/PAN-X or overlapping examples is unknown from the pinned model card.
- Location mapping: city, country, location -> WikiANN LOC; exact spans deduplicated
- Warmup: 8 sentences in 0.9560s; steady inference: 104.0755s
- Linear full-matrix inference estimate from this run: 67125.5113s

| Language | Evaluation status | Coverage status | Sentences | Gold LOC | P | R | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `af` | evaluated | evaluated_multilingual_claim | 8 | 12 | 0.5789 | 0.9167 | 0.7097 |
| `am` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.1250 | 0.3333 | 0.1818 |
| `ar` | evaluated | evaluated_multilingual_claim | 8 | 7 | 0.2222 | 0.2857 | 0.2500 |
| `az` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.5000 | 0.5000 | 0.5000 |
| `be` | evaluated | evaluated_multilingual_claim | 8 | 8 | 0.5833 | 0.8750 | 0.7000 |
| `bg` | evaluated | evaluated_multilingual_claim | 8 | 7 | 0.6000 | 0.8571 | 0.7059 |
| `bn` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `ca` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.3333 | 0.5000 | 0.4000 |
| `ceb` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.1429 | 0.3333 | 0.2000 |
| `cs` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0909 | 0.5000 | 0.1538 |
| `cy` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.5000 | 0.8000 | 0.6154 |
| `da` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `de` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.2500 | 0.6667 | 0.3636 |
| `el` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.5000 | 0.8333 | 0.6250 |
| `en` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.3333 | 0.7500 | 0.4615 |
| `eo` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.5000 | 1.0000 | 0.6667 |
| `es` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.4000 | 0.6667 | 0.5000 |
| `et` | evaluated | evaluated_multilingual_claim | 8 | 3 | 1.0000 | 1.0000 | 1.0000 |
| `eu` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.3333 | 0.5000 | 0.4000 |
| `fa` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `fi` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.7500 | 1.0000 | 0.8571 |
| `fr` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.2000 | 0.5000 | 0.2857 |
| `fy` | evaluated | evaluated_multilingual_claim | 8 | 11 | 0.7692 | 0.9091 | 0.8333 |
| `ga` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.2500 | 1.0000 | 0.4000 |
| `gd` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.6667 | 0.6667 | 0.6667 |
| `gl` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.3333 | 1.0000 | 0.5000 |
| `gu` | evaluated | evaluated_multilingual_claim | 8 | 2 | 1.0000 | 0.5000 | 0.6667 |
| `he` | evaluated | evaluated_multilingual_claim | 8 | 13 | 0.9091 | 0.7692 | 0.8333 |
| `hi` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `hu` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.3333 | 0.5000 | 0.4000 |
| `hy` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.3333 | 0.5000 | 0.4000 |
| `id` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.2000 | 0.3333 | 0.2500 |
| `ig` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.3333 | 0.6667 | 0.4444 |
| `is` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.4000 | 0.6667 | 0.5000 |
| `it` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.1250 | 0.5000 | 0.2000 |
| `ja` | evaluated | evaluated_multilingual_claim | 8 | 7 | 0.5000 | 0.8571 | 0.6316 |
| `jv` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.5000 | 0.8333 | 0.6250 |
| `ka` | evaluated | evaluated_multilingual_claim | 8 | 7 | 0.6250 | 0.7143 | 0.6667 |
| `kk` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.4000 | 0.5000 | 0.4444 |
| `km` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `kn` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.5714 | 0.6667 | 0.6154 |
| `ko` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.2500 | 0.2500 | 0.2500 |
| `ku` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.5000 | 1.0000 | 0.6667 |
| `ky` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `la` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.5000 | 1.0000 | 0.6667 |
| `lt` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.4444 | 0.8000 | 0.5714 |
| `lv` | evaluated | evaluated_multilingual_claim | 8 | 10 | 0.6667 | 0.6000 | 0.6316 |
| `mg` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.4167 | 1.0000 | 0.5882 |
| `mk` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.2000 | 0.6667 | 0.3077 |
| `ml` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.5000 | 0.2500 | 0.3333 |
| `mn` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.6250 | 1.0000 | 0.7692 |
| `mr` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.1667 | 1.0000 | 0.2857 |
| `ms` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `mt` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.5000 | 0.8000 | 0.6154 |
| `my` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `ne` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `nl` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.4167 | 1.0000 | 0.5882 |
| `no` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.5714 | 0.8000 | 0.6667 |
| `pa` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.2500 | 0.2500 | 0.2500 |
| `pl` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.6250 | 0.8333 | 0.7143 |
| `ps` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `pt` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.6667 | 1.0000 | 0.8000 |
| `ro` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `ru` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.2000 | 0.5000 | 0.2857 |
| `si` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `sk` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.2857 | 0.5000 | 0.3636 |
| `sl` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.6667 | 1.0000 | 0.8000 |
| `sq` | evaluated | evaluated_multilingual_claim | 8 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `sr` | evaluated | evaluated_multilingual_claim | 8 | 6 | 0.0000 | 0.0000 | 0.0000 |
| `sv` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.1667 | 0.3333 | 0.2222 |
| `ta` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `te` | evaluated | evaluated_multilingual_claim | 8 | 7 | 0.0000 | 0.0000 | 0.0000 |
| `tg` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `th` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `tr` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.1429 | 0.2500 | 0.1818 |
| `uk` | evaluated | evaluated_multilingual_claim | 8 | 1 | 0.2000 | 1.0000 | 0.3333 |
| `ur` | evaluated | evaluated_multilingual_claim | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `uz` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `vi` | evaluated | evaluated_multilingual_claim | 8 | 5 | 0.3333 | 0.4000 | 0.3636 |
| `yi` | evaluated | evaluated_multilingual_claim | 8 | 2 | 0.2000 | 0.5000 | 0.2857 |
| `yo` | evaluated | evaluated_multilingual_claim | 8 | 4 | 0.5000 | 0.2500 | 0.3333 |
| `zh` | evaluated | evaluated_multilingual_claim | 8 | 7 | 0.2143 | 0.4286 | 0.2857 |

### `Davlan/xlm-roberta-base-ner-hrl`

- Revision: `253f557bd8249b8515114cfd7f71974fe5fa4d2f`; model card: [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl)
- Language coverage: Fine-tuned for Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese. Other evaluated languages are cross-lingual transfer.
- Training data: Its pinned model card lists ANERcorp, CoNLL 2002/2003, Europeana Newspapers, Italian I-CAB, Latvian NER, Paramopama/Second HAREM, and MSRA for those ten languages.
- WikiANN overlap: The model card does not list WikiANN/PAN-X among its fine-tuning corpora. Pretraining/text-level overlap was not audited.
- Location mapping: B-LOC/I-LOC -> WikiANN LOC; token labels grouped to character spans
- Warmup: 8 sentences in 0.2507s; steady inference: 30.9294s
- Linear full-matrix inference estimate from this run: 19948.5276s

| Language | Evaluation status | Coverage status | Sentences | Gold LOC | P | R | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `af` | evaluated | cross_lingual_transfer | 8 | 12 | 0.0000 | 0.0000 | 0.0000 |
| `am` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `ar` | evaluated | fine_tuned_language | 8 | 7 | 0.1000 | 0.1429 | 0.1176 |
| `az` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `be` | evaluated | cross_lingual_transfer | 8 | 8 | 0.0000 | 0.0000 | 0.0000 |
| `bg` | evaluated | cross_lingual_transfer | 8 | 7 | 0.1429 | 0.1429 | 0.1429 |
| `bn` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `ca` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `ceb` | evaluated | cross_lingual_transfer | 8 | 6 | 0.0833 | 0.1667 | 0.1111 |
| `cs` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `cy` | evaluated | cross_lingual_transfer | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `da` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `de` | evaluated | fine_tuned_language | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `el` | evaluated | cross_lingual_transfer | 8 | 6 | 0.0000 | 0.0000 | 0.0000 |
| `en` | evaluated | fine_tuned_language | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `eo` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `es` | evaluated | fine_tuned_language | 8 | 3 | 0.3333 | 0.3333 | 0.3333 |
| `et` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `eu` | evaluated | cross_lingual_transfer | 8 | 4 | 0.3333 | 0.2500 | 0.2857 |
| `fa` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `fi` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `fr` | evaluated | fine_tuned_language | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `fy` | evaluated | cross_lingual_transfer | 8 | 11 | 0.0000 | 0.0000 | 0.0000 |
| `ga` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `gd` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `gl` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `gu` | evaluated | cross_lingual_transfer | 8 | 2 | 0.5000 | 0.5000 | 0.5000 |
| `he` | evaluated | cross_lingual_transfer | 8 | 13 | 0.2667 | 0.3077 | 0.2857 |
| `hi` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `hu` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `hy` | evaluated | cross_lingual_transfer | 8 | 6 | 0.1111 | 0.1667 | 0.1333 |
| `id` | evaluated | cross_lingual_transfer | 8 | 3 | 0.2000 | 0.3333 | 0.2500 |
| `ig` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `is` | evaluated | cross_lingual_transfer | 8 | 3 | 0.3333 | 0.3333 | 0.3333 |
| `it` | evaluated | fine_tuned_language | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `ja` | evaluated | cross_lingual_transfer | 8 | 7 | 0.2727 | 0.8571 | 0.4138 |
| `jv` | evaluated | cross_lingual_transfer | 8 | 6 | 0.0000 | 0.0000 | 0.0000 |
| `ka` | evaluated | cross_lingual_transfer | 8 | 7 | 0.1538 | 0.2857 | 0.2000 |
| `kk` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `km` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `kn` | evaluated | cross_lingual_transfer | 8 | 6 | 0.1667 | 0.1667 | 0.1667 |
| `ko` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `ku` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `ky` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `la` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `lt` | evaluated | cross_lingual_transfer | 8 | 5 | 0.2000 | 0.2000 | 0.2000 |
| `lv` | evaluated | fine_tuned_language | 8 | 10 | 0.0000 | 0.0000 | 0.0000 |
| `mg` | evaluated | cross_lingual_transfer | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `mk` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `ml` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `mn` | evaluated | cross_lingual_transfer | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `mr` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `ms` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `mt` | evaluated | cross_lingual_transfer | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `my` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `ne` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `nl` | evaluated | fine_tuned_language | 8 | 5 | 0.1429 | 0.2000 | 0.1667 |
| `no` | evaluated | cross_lingual_transfer | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `pa` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `pl` | evaluated | cross_lingual_transfer | 8 | 6 | 0.0000 | 0.0000 | 0.0000 |
| `ps` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `pt` | evaluated | fine_tuned_language | 8 | 2 | 0.5000 | 0.5000 | 0.5000 |
| `ro` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `ru` | evaluated | cross_lingual_transfer | 8 | 4 | 0.1667 | 0.2500 | 0.2000 |
| `si` | evaluated | cross_lingual_transfer | 8 | 5 | 0.0000 | 0.0000 | 0.0000 |
| `sk` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `sl` | evaluated | cross_lingual_transfer | 8 | 4 | 0.1667 | 0.2500 | 0.2000 |
| `sq` | evaluated | cross_lingual_transfer | 8 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `sr` | evaluated | cross_lingual_transfer | 8 | 6 | 0.0000 | 0.0000 | 0.0000 |
| `sv` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `ta` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `te` | evaluated | cross_lingual_transfer | 8 | 7 | 0.2000 | 0.2857 | 0.2353 |
| `tg` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `th` | evaluated | cross_lingual_transfer | 8 | 2 | 0.0000 | 0.0000 | 0.0000 |
| `tr` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `uk` | evaluated | cross_lingual_transfer | 8 | 1 | 0.0000 | 0.0000 | 0.0000 |
| `ur` | evaluated | cross_lingual_transfer | 8 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `uz` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `vi` | evaluated | cross_lingual_transfer | 8 | 5 | 0.2500 | 0.2000 | 0.2222 |
| `yi` | evaluated | cross_lingual_transfer | 8 | 2 | 0.3333 | 0.5000 | 0.4000 |
| `yo` | evaluated | cross_lingual_transfer | 8 | 4 | 0.0000 | 0.0000 | 0.0000 |
| `zh` | evaluated | fine_tuned_language | 8 | 7 | 0.2000 | 0.5714 | 0.2963 |

## Aggregate scores

Macro scores are unweighted means over evaluated languages; micro scores aggregate exact-span counts. Undefined precision/recall/F1 values use zero.

| Recognizer | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spacy_en` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| `gliner2_multi` | 0.3220 | 0.5099 | 0.3758 | 0.3496 | 0.5671 | 0.4326 |
| `xlmr_ner_hrl` | 0.0629 | 0.0847 | 0.0694 | 0.0773 | 0.1067 | 0.0896 |

Estimated total CPU inference for the complete matrix: 87093.4328s. Linear estimate from measured steady-state inference throughput; excludes model download/load and data acquisition. Small feasibility samples are not a quality result and may not predict every language.

The upstream spaCy model is scored on English only. Its other 81 available languages are explicitly marked as not evaluated; it is not treated as a multilingual competitor.
