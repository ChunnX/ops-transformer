# ============================================================================
# Result verification for stem_oam_prep_varlen_q (direct-invoke pathway)
# ============================================================================
#
# Compares kernel output.bin vs golden.bin with BF16 tolerance.
# ============================================================================

import numpy as np
import sys

# Verification parameters
dtype = np.bfloat16    # Output dtype, matches kernel output
rtol = 1e-2            # Relative tolerance (BF16 range)
atol = 1e-2            # Absolute tolerance (BF16 range)

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    # Cast to float32 for comparison (bfloat16 numpy comparison is limited)
    output_f = output.astype(np.float32)
    golden_f = golden.astype(np.float32)

    if output_f.shape != golden_f.shape:
        print(f"Shape mismatch: output {output_f.shape} vs golden {golden_f.shape}")
        return False

    if np.allclose(output_f, golden_f, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output_f.shape}")
        print(f"Max diff: {np.max(np.abs(output_f - golden_f))}")
        return True
    else:
        diff = np.abs(output_f - golden_f)
        print(f"Verification FAILED!")
        print(f"Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden_f))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden_f)}")
        if len(mismatches) > 0 and len(mismatches) <= 20:
            for idx in mismatches:
                print(f"  idx={idx}: output={output_f[idx]:.6f}, golden={golden_f[idx]:.6f}, diff={diff[idx]:.6f}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)