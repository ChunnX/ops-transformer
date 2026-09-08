# Fmm016StreamingTopKBlockedV2

Short/medium-sequence static router variant.  It is selected by the TorchAir
converter for padded leaf counts up to 1024 and shares the fixed-shape
`selected_index`/`selected_valid` contract documented in the parent directory.

Build and package it with the standard `ops-transformer` CANN build; do not
invoke the legacy `Fmm016StreamingTopKBlocked` symbol.
