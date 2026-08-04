# Session: sparse_lightning_indexer_grad_kl_loss 507015 fix
Date: 2026-08-04

## Root cause

The tail-padding code in `Init()` (added by commit `fe1a4aee2`) distributes
padding work across AIV cores. When `t1PadLen < totalAivNum` (or `t2PadLen < totalAivNum`),
integer division gives `splitPadLen = 0`, and only `remainPadLen` cores get 1 unit.
All other cores get `currentCorePadLen = 0`.

For cores with 0 work:
- `t1PadLenSum = aivIdx * 0 + Min(aivIdx, remainPadLen)`
- When `aivIdx >= remainPadLen`, `t1PadLenSum = remainPadLen = t1PadLen`
- This makes `dQOutputOffset = (t1PadLen + actualLen) * ... = t1Size * ...`
- The pointer `dQueryIndex + dQOutputOffset` is **one-past-the-end** of the output buffer

The code then calls `SetGlobalBuffer` with this pointer (and size 0), followed by
`Fill` with count 0. On real hardware (950), the runtime validates the pointer and
triggers 507015 (AICore exception / SVM page fault).

The cannsim simulator does NOT validate this and silently ignores it,
which is why the test passed in simulation.

## Fix

`op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_kernel_base.h`

Added `if (currentCoreT1PadLen > 0)` guard around the Fill operations for
dQueryIndex/dWeight, and `if (currentCoreT2PadLen > 0)` for dKeyIndex.
Cores with zero padding work now skip the GlobalTensor creation and Fill.

## Verification

- Build: `bash build.sh --pkg --soc=ascend950 --ops=sparse_lightning_indexer_grad_kl_loss` → PASSED
- cannsim: default example test → PASSED (no tail-padding, not affected)
- Functionality: tail-padding still works for cores that DO have work (currentCorePadLen > 0)

## Files changed

1 file: `op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_kernel_base.h`
- Lines 147-169: Added `if (currentCoreT1PadLen > 0)` guard for t1 padding
- Lines 184-196: Added `if (currentCoreT2PadLen > 0)` guard for t2 padding
