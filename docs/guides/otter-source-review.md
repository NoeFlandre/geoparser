# Otter static source review

This review covers the inference code at the two immutable Hub revisions in
[otter-source-manifest.json](../examples/otter-source-manifest.json). It does not certify arbitrary
injected model objects. It does not authorize a model run.

The review read the model cards, configuration, and inference import closure.
The closure contains `configuration_otter.py`, `modeling_otter.py`, `loss.py`,
`masks.py`, and `metrics.py`. Both revisions contain identical copies of these
five Python files. The manifest records each file's SHA-256 and size.
`collate_fn.py` is not imported by the inference path. Training is out of scope.

## Execution and acquisition boundaries

`AutoModel` requires `trust_remote_code=True` for these checkpoints. The custom
code constructs standard Transformers encoders from embedded configurations.
The reviewed import closure contains tensor operations and span decoding. No
shell commands, credential reads, or external reporting calls were found.
This static finding is limited to the pinned source and declared dependencies.
It does not replace dependency checks or a real-model smoke test.

Two fallback paths prevent a remote model revision alone from fixing all inputs:

- `_encoder_config` fetches an unpinned base configuration if an embedded encoder
  configuration is absent. Both reviewed checkpoint configurations contain the
  required encoder configurations. Keep those embedded values.
- `_load_tokenizer` tries `model.name_or_path` without a revision or a
  `local_files_only` constraint. It sets `trust_remote_code=True`, catches all
  exceptions, and then loads an unpinned base tokenizer. Load the shipped
  tokenizers from the verified local snapshot before constructing the adapter.
  Keep networking disabled during model construction and prediction.

`OtterRecognizer` does not download files, import this custom code, or construct a
model. It requires an already prepared model and inspects the tokenizer cache
fields without invoking the lazy properties. Its `source` is a caller-supplied
identity for configuration and cache separation. The adapter cannot verify the
origin of an in-memory Python object.

## Span and batch contract

`predict` accepts a string or a sequence, labels, a threshold, a batch size, and a
maximum sequence length. The adapter always passes a list. The result contains
one entity list per input in the same order. Each entity contains `text`,
`label`, `start`, `end`, and `score`.

Offsets index Python string characters. The start is inclusive and the end is
exclusive. The upstream decoder subtracts the cross-encoder prefix and trims
leading and trailing whitespace. The adapter checks the original slice. It does
not normalize text, search for repeated names, or adjust offsets.

`metrics.py` applies a sigmoid and keeps scores strictly greater than the
threshold. Its greedy decoder removes overlapping token spans. The adapter
validates output shape, bounds, surface text, labels, scores, and overlaps.
Malformed output raises an exception with the input index and raw result.

The upstream batch loop skips blank input and restores its previous training
mode. It uses `torch.no_grad()`. A real smoke test must check batch invariance,
first-token offsets, mixed scripts, combining marks, and repeated inputs.
Offline fake-model tests cannot establish the actual model's behavior.

## Limits and cache behavior

Both checkpoint configurations set `max_seq_length=1024` and
`max_span_length=30`. The span bound counts model subword tokens. The
cross-encoder label prefix and special tokens consume the sequence budget.
The bi-encoder type tokenizer has a 512-token limit.

The upstream prediction path truncates long input. The adapter first counts
untruncated tokens with the prepared tokenizer. It rejects a batch before model
prediction if any document or bi-encoder label exceeds its limit. It neither
chunks documents nor drops their tails.

The reviewed `encode_labels` returns tokenized label inputs. It does not return
label embeddings. The bi-encoder prediction path calls the type encoder for
every batch. These comparison arms therefore record label-embedding caching as
`none`. A cached-embedding optimization requires a separate reviewed adapter and
an output-equivalence test.

## License and language claims

Both pinned model cards declare Apache-2.0. The upstream repository at
`b7d53e4eb533296ac5940bf03efa0ae17cff8f6c` also has an Apache-2.0 license.
The model files are not redistributed here. Preserve required notices if a
future artifact redistributes them.

The pinned cards say multilingual but do not enumerate supported languages.
The upstream README discusses a 91-language FiNERweb training mixture. This
statement does not establish a supported-language list for either checkpoint.
Record per-language support as `unspecified` until checkpoint-specific evidence
supports a stronger label. Training overlap with the selected public evaluation
texts remains `unknown`. Do not call the held-out results contamination-free.

## Reviewed sources

- [Cross model card](https://huggingface.co/whoisjones/otter-cross-mmbert/blob/8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d/README.md)
- [Bi model card](https://huggingface.co/whoisjones/otter-bi-mmbert/blob/53e10a09bc71a2e45980a7a257233a28305a5777/README.md)
- [Inference implementation](https://huggingface.co/whoisjones/otter-cross-mmbert/blob/8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d/modeling_otter.py)
- [Span decoder](https://huggingface.co/whoisjones/otter-cross-mmbert/blob/8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d/metrics.py)
- [Span masks](https://huggingface.co/whoisjones/otter-cross-mmbert/blob/8729188e4f5fc7948d0e9dfd7d7e6d36c2e7270d/masks.py)
- [Upstream README](https://github.com/whoisjones/otter/blob/b7d53e4eb533296ac5940bf03efa0ae17cff8f6c/README.md)
- [Upstream license](https://github.com/whoisjones/otter/blob/b7d53e4eb533296ac5940bf03efa0ae17cff8f6c/LICENSE)

No model weights were downloaded. No candidate custom code, inference, or
training was run. The weight digests and sizes in the manifest come from the
Hub's immutable-revision LFS metadata. A later run must verify the actual bytes.
