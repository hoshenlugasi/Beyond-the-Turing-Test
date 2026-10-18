# Analysis input provenance

The dataset core comes from the submitted supplementary dataset: each released
game has the four submitted files (transcript, metadata, votes and surveys).
Their layout and content are preserved separately from auxiliary analysis inputs.

The numeric auxiliary inputs support the reported calculations, not figure
generation. Message-level labels, counts, validation decisions and cached judge
results are derived inputs; they do not add original conversation text or raw
model logs. They are not substitutes for the submitted transcripts. Cached
annotations/judgments are observations from the recorded model runs, not a promise
of identical results from a fresh paid API call.

The separate paper-examples JSON is the explicit text exception: five exchanges
from the final message-label table and four positive worker chains from the final
worker table. Only fixed, reviewed quotations and narrow summaries are exported.
Worker output summaries are paraphrases; fields named worker_output_quote contain
literal substrings of recorded worker outputs. Candidate and selected/delivered
messages are literal excerpts. Generator membership, Reviewer selection,
personal-chat delivery events and public transcript appearance were checked
privately. Observed information flow does not establish a worker's causal effect.
Source identifiers, run fragments, paths and audit mappings remain private.
No full worker responses, Reviewer reasoning, hidden chain of thought or raw logs
are included in these examples.

The demographic summary contains only verified paper-reported aggregates for 226
known demographic records. It is not individual demographic data and cannot
reproduce participant linkage or deduplication. Age statistics retain source
precision (31.88 and 9.66), rounded to 31.9 and 9.7 in the paper. Only the three
paper-reported country counts are included, not rare-country cells.

Prompts reproduced in the paper are normalized presentations for readability;
they are not byte-identical runtime prompt/configuration snapshots. The released
code and configuration specify executable behavior. Neither paper prompt wording
nor these illustrative examples replace the recorded study configuration.