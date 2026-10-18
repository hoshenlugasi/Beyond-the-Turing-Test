# Structured internal outputs

These are privacy-reviewed, field-filtered records from the same 50 study games. They are separate from the four-file submitted observational dataset.

The existing JSONL event structure records parsed scheduler decisions, generated candidates, explicit short reviewer assessments and selections, deterministic guards, delivery outcomes, and selected repetition/behavior/round-summary updates. Not all shared-memory fields or worker results are included.

`v6_delivery_sent` is a recorded delivery. `v6_reviewer_selected` and `v4_pipeline_complete` are not proof of delivery. Missing observations were not filled in. An omitted event must not be interpreted as a zero or a WAIT. Candidate text, including rejected or suppressed text, is not part of the public chat and must not enter the 2,414 delivered-message denominator.

The export contains no full prompts, full memory snapshots, raw API envelopes, billing metadata, participant registry, or private identifier maps. All 5,385 distinct text values in the initial projection received a separate manual review. Flagged and context-ambiguous values were withheld in full, including repeated copies. Redactions can prevent exact text comparison; candidate indices, event types and delivery statuses are preserved. This review reduces disclosure risk; it is not a guarantee of anonymity, consent determination or authorization for re-identification.
