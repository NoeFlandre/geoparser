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
