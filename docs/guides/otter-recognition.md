# Compare the Otter recognizers

The optional `OtterRecognizer` adapter accepts an already prepared Otter model.
It supports the cross-encoder and bi-encoder interfaces reviewed for issue #161.
It does not change the default GLiNER2 recognizer or the existing benchmark arms.
No Otter quality or performance result is available from this change.

## Use a prepared model

Prepare the model in an approved compute environment before you use this adapter.
The model must contain its local tokenizer objects. Do not pass an object whose
first tokenizer access can fetch files.

```python
from geoparser.modules import OtterRecognizer

recognizer = OtterRecognizer(
    prepared_model,
    source="whoisjones/otter-cross-mmbert@8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d",
    entity_types=("city", "country", "location"),
    threshold=0.5,
    batch_size=8,
)
spans = recognizer.predict(["Paris and Paris", "Zürich", ""])
raw = recognizer.last_raw_predictions
```

`prepared_model` is an input to this example. The snippet does not acquire or
load a checkpoint. The adapter records `source` as caller-supplied provenance.
That string separates module identities. It does not prove which weights or
code the object contains.

The return value has one span list per input. Offsets are half-open Python
character positions in the original string. The adapter preserves Unicode and
repeated documents. Blank inputs produce empty lists. A score must be strictly
greater than the threshold. The default threshold comes from the supplied model
configuration and is recorded as a numeric value.

`predict_batch(texts)` returns sets of spans for the existing PAN-X predictor
interface. Save `last_raw_predictions` immediately after each call if you need
raw model scores and labels. The next call replaces that attribute. Do not share
one adapter instance between concurrent calls.

Malformed model output raises `PredictionOutputError`. Its `document_index` and
`raw_output` retain the failure evidence. Model execution errors propagate.
An overlength input raises `ValueError` before prediction. A benchmark runner
must retain the failed or unsupported document in its denominator. It must not
replace a failure with an ordinary empty prediction or omit the document.

## Prepare the pinned snapshot

Use an environment authorized for the run and check its resources. Model
construction executes custom Python code. Read the
[static source review](otter-source-review.md)
and the
[file manifest](../examples/otter-source-manifest.json)
before model construction.

1. Materialize the exact model revision from the arm plan in the approved
   environment. Keep the two model snapshots separate.
2. Verify the SHA-256 and size of every listed file. The recorded safetensors
   digests are Hub LFS metadata. This change did not download the weight files.
3. Check the code import closure against the review. Use the repository's locked
   dependencies. Keep the embedded encoder configurations.
4. Disable networking for construction and inference. Set `HF_HUB_OFFLINE=1`
   and `TRANSFORMERS_OFFLINE=1`. Use local paths and `local_files_only=True`.
5. Load the shipped tokenizers with custom tokenizer code disabled. Do not use
   an unpinned base tokenizer if loading fails.
6. Load the local model with the explicit, approved custom-code setting and
   safetensors. Install the prepared tokenizers in `_tokenizer` for the
   cross-encoder. For the bi-encoder, set `_token_tokenizer` and `_type_tokenizer`.
7. Set evaluation mode and the approved device. Construct the adapter with the
   verified snapshot identity and an explicit threshold.
8. Run the approved smoke fixture before the corpus. Check batch-size invariance,
   first-token spans, emoji, combining marks, mixed scripts, and repeated names.

The underscore attributes match the pinned upstream implementation. This is a
version-specific preparation contract. Re-review it when the source revision
changes. The adapter has no automatic loader because an injected object cannot
establish a verified file history.

The model weights alone occupy about 1.24 GB for cross and 1.91 GB for bi.
Tokenizers, dependencies, activations, and batch tensors require more storage or
memory. These file sizes are not a safe RAM or VRAM estimate. Establish the
actual peak on the approved device before the larger run.

## Bound the first smoke run

Start with at most 12 short synthetic texts, each below 128 tokens. Run one
model at a time with one CPU thread. Compare batch sizes of one and two. Use a
provisional target of 16 GiB host RAM and 8 GiB free disk. Check the actual
available resources first. These planning targets have not been measured.

Stop if a hash differs, the model attempts a network fallback, an output is
invalid, spans change with batch size, or memory pressure appears. Retain the
failure evidence. Measure peak memory before you expand to the public
validation split. A passing smoke run demonstrates adapter compatibility. It
does not establish comparative recognition quality.

## Use the explicit arm plan

[otter-recognition-arms.json](../examples/otter-recognition-arms.json) records both
pins, thresholds, labels, limits, and the screening procedure. It is an arm plan,
not a completed protocol manifest or an execution command. Dataset samples,
hardware, and results remain unmaterialized.

Use the same original texts and location-label mapping for both Otter arms and
the existing GLiNER2.5 and Davlan XLM-R baselines. Tune each Otter threshold on the
public development split. Freeze the selection before the common held-out test.
Retain the selected sample hashes, executing code revision, dependency lock,
raw outputs, failure counts, and source slices in the benchmark protocol.

Both models have a 1,024-token sequence limit and a 30-subword span limit. The
cross-encoder prefix consumes part of the sequence budget. The adapter counts
the complete prefixed input without truncation. The bi-encoder also checks the
label tokenizer's length limit. No implicit long-document chunking is present.

Label-embedding caching is `none` for both comparison arms. In the reviewed
bi-encoder code, `encode_labels` produces tokenizer inputs and `predict` runs
the label encoder for each batch. Do not report cached-embedding throughput.

Report per-language exact-span scores and paired uncertainty intervals. Separate
fetch, load, warm-up, and steady-state timings. Record peak process and device
memory. Include failed and unsupported examples in coverage reporting. A
batch-level failure needs per-document accounting before aggregation.

The pinned model cards declare Apache-2.0 and describe the models as
multilingual. They do not enumerate supported languages. The plan therefore
uses `unspecified` language support and `unknown` training overlap. Author
claims about accuracy and speed remain hypotheses until this project runs the
controlled comparison. Use public benchmark data only. This plan does not use
the owner's OSM text data or train either model.
