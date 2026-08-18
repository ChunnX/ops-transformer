# CatCCOS UDMA overlay

The build uses the official CatCCOS repository at revision
`878c2e0a504be8c49c03b5e1157a7c714a628e29`.

The headers under `include/catccos` are the minimal all-to-all-v UDMA delta
required by this operator and are placed before the official CatCCOS include
directory. They preserve the validated UDMA peer-serial scheduling and
layout-aware two-dimensional copy path while all other CatCCOS, CATLASS, and
SHMEM sources continue to come from the official third-party checkout.

The overlay also carries the follow-up UDMA MoE delta from CatCCOS commit
`1d96ae736f1054a28b0ddf3cbb96cd0c64c81a21`: the optional AllToAllV export
variant and the GroupedMatmul-AllToAllV non-communication-core deadlock fix.
CatCCOS commit `2444c7ee5acee145c8a296e3462b4bcf487571b8` further pipelines
the optional export: AIV subcore 0 exports the previous staging buffer while
subcore 1 submits the current local MTE and remote UDMA communication.
The GroupedMatMul-AllToAllV overlay additionally carries separate input and
output row extents so unbalanced routes can consume `sum(global_tokens)` rows
and return `sum(local_tokens)` rows without requiring those totals to match.
It also carries the target-order AIC/AIV schedulers from CatCCOS commit
`f03b24518e25d9965977b3d70813238f799da2bb`. Complete N-width row ranges are
staged in destination order, reducing each source-rank/round transfer to one
contiguous UDMA request; the final scheduler coordinate-type fixes are included.
The overlaid problem-shape ABI reads both route-count tables from contiguous
signed 64-bit storage, matching the public PyTorch `torch.int64` interface.
