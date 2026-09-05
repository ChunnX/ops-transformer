# Fmm016StreamingTopKBlockedV4

Long-sequence static router variant.  It is selected by the TorchAir
converter for padded leaf counts above 1024 and uses the same fixed-shape
router contract as V2.  The split keeps long scans isolated from the
short-sequence tiling cache.

Build and package it with the standard `ops-transformer` CANN build; do not
invoke the legacy `Fmm016StreamingTopKBlocked` symbol.
