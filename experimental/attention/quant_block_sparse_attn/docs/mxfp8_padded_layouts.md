# MXFP8 padded query layouts

`quant_mode=2` supports matching `layout_q` / `layout_out` values TND, BSND and BNSD.
K/V remain segmented PA_BNBD. This does not add continuous BSND/BNSD KV support.

| Layout | Q | Q descale (E8M0) | attention_out (BF16) | LSE (FP32) |
| --- | --- | --- | --- | --- |
| TND | [T,Nq,D] | [T,Nq,ceil(D/64),2] | [T,Nq,Dv] | [T,Nq] |
| BSND | [B,Sq,Nq,D] | [B,Sq,Nq,ceil(D/64),2] | [B,Sq,Nq,Dv] | [B,Nq,Sq] |
| BNSD | [B,Nq,Sq,D] | [B,Nq,Sq,ceil(D/64),2] | [B,Nq,Sq,Dv] | [B,Nq,Sq] |

Q and Q descale must be contiguous. Pass `layout_out` explicitly to match `layout_q`.
Existing head dimension, sparse block, mask and PA constraints still apply.

`cu_seqlens_q` remains required: int32 [B+1], starting at zero, with nonnegative
differences no greater than physical Sq. Differences give effective Q lengths.
For equal lengths use [0,Sq,2*Sq,...]. `seqused_q` is not consumed by this MX path.
The GM batch stride uses physical Sq, never the cumulative effective lengths.
Only effective tokens are written; padded output/LSE elements are unspecified.
Sparse tensors retain [B,Nq,Qb,Kb] / [B,Nq,Qb]; Qb must cover ceil(Sq/q_block_size).
Rows beyond each batch's effective Q length should have zero sparse counts.
The existing metadata scheduler is driven by these sparse rows and requires no
new layout-specific scheduling format. Regenerate metadata when sparse rows change.

Layout is encoded in the existing LAYOUT_T tiling key and selected by constexpr
branches. Physical Sq is passed in MX tiling data. Rebuild host and device together:
the MX tiling data struct now contains qSeqSize. Existing FP8 template selections
and TND/TN output conventions are retained.

Validation: the existing InferShape CSV runner includes BSND/BNSD cases, and
test_mx_query_layout.cpp checks the production address helpers on CPU.
On an Ascend device, set `QBSA_CHECK_PADDED_LAYOUTS=1` when running the existing
MXFP8 golden in eager mode. It reuses each TND case's Q/K/V/scales/sparse metadata,
repackages Q and Q scale, and compares both padded outputs and BNS LSE exactly.
Include B>1, GQA, unequal Q lengths, S1 tails, empty sparse rows and causal cases;
enable LSE to exercise its output. Performance must be checked on device separately.

## STC

Both `tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_stc.csv` and the matching
`tests/ttk` CSV contain 12 added layout cases (six per layout). Pytest cases include
unequal batch lengths and normalize valid output tokens to TND/TN for the existing
CPU comparison. TTK cases use equal Q lengths because its comparator checks all
output elements, including the full physical sequence dimension.

Run the new pytest STC cases on the Ascend test machine, for example:

```bash
python qbsa_mxfp8_golden.py --csv --case_files qbsa_mxfp8_stc.csv --case_name <case_name_with_BSND_or_BNSD_suffix> --mode all
```

The graph path uses the same public-layout packing and valid-token comparison.
The CPU reference stays in packed TND/TN; the test adapters convert only at the
operator boundary. Existing STC cases are retained.
