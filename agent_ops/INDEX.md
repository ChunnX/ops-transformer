# Operator Performance Ledgers

This directory contains lightweight per-operator performance ledgers. It is a summary layer, not a raw profiling dump store.

Current ledgers:

| operator | status | summary |
| --- | --- | --- |
| `lightning_indexer` | `optimized_kept` | Host `blockDim` capped by shape-implied base-block count. |
| `recurrent_gated_delta_rule` | `optimized_kept` | Host `vectorCoreNum` capped by `b * nv`. |

Raw `msprof` output, long logs, generated binaries, `.o`, `.bin`, and full CSV dumps should stay under `artifacts/` or build outputs. Ledger files should only record commands, hashes, matrix summaries, decisions, and follow-up hypotheses.
