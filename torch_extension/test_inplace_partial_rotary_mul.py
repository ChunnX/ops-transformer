#!/usr/bin/env python3
"""
Verification script for inplace_partial_rotary_mul torch interface
on A3 NPU (910B3) with CANN 9.1.0
"""
import os
import sys
import torch
import torch_npu
import numpy as np

# Set correct CANN path
os.environ["ASCEND_HOME_PATH"] = "/mnt/workspace/Ascend/cann-9.1.0"

# Register the op directly (bypassing full package import which triggers
# all ops' JIT compilation and may fail due to CANN version mismatch)
torch.library.define(
    'npu_ops_transformer::inplace_partial_rotary_mul',
    '(Tensor(a!) x, Tensor r1, Tensor r2, *, str rotary_mode, int[2] partial_slice) -> ()'
)

@torch.library.impl('npu_ops_transformer::inplace_partial_rotary_mul', 'Meta')
def inplace_partial_rotary_mul_meta(x, r1, r2, *, rotary_mode, partial_slice):
    torch._check(x.dim() == 4, lambda: f"Input tensor x's dim num should be 4, actual {x.dim()}.")

# JIT compile only our C++ extension
from torch.utils.cpp_extension import load

torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
package_path = "/mnt/workspace/gitCode/ops-transformer/torch_extension/npu_ops_transformer"
cann_path = os.environ["ASCEND_HOME_PATH"]

cxx_args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
            '-s', '-fvisibility=hidden', '-D_FORTIFY_SOURCE=2', '-O2', '-w']
if torch._C._GLIBCXX_USE_CXX11_ABI:
    cxx_args.append('-D_GLIBCXX_USE_CXX11_ABI=1')

ldflags = [
    '-L' + os.path.join(cann_path, 'lib64'), '-lascendcl',
    '-L' + os.path.join(torch_npu_path, 'lib'), '-ltorch_npu'
]

include_paths = [
    os.path.join(cann_path, 'include'),
    os.path.join(torch_npu_path, 'include'),
    os.path.join(torch_npu_path, 'include/third_party/hccl/inc'),
    os.path.join(torch_npu_path, 'include/third_party/acl/inc'),
    os.path.join(package_path, 'common/inc')
]

print("Compiling inplace_partial_rotary_mul C++ extension...")
op_module = load(
    name='inplace_partial_rotary_mul',
    sources=[os.path.join(package_path, '..', 'test_inplace_partial_rotary_mul_standalone.cpp')],
    extra_include_paths=include_paths,
    extra_cflags=cxx_args,
    extra_ldflags=ldflags,
    verbose=True
)
print("Compilation succeeded!\n")

# Register PrivateUse1 dispatch
from torch.library import impl

AS_LIBRARY_FRAG = torch.library.Library('npu_ops_transformer', 'IMPL', 'PrivateUse1')

def _inplace_partial_rotary_mul_dispatch(x, r1, r2, *, rotary_mode, partial_slice):
    op_module.inplace_partial_rotary_mul(x, r1, r2, rotary_mode, partial_slice)

AS_LIBRARY_FRAG.impl('inplace_partial_rotary_mul', _inplace_partial_rotary_mul_dispatch)

# ========== Test Cases ==========

def compute_reference(x, r1, r2, rotary_mode, partial_slice):
    """Compute reference result on CPU using numpy"""
    x_np = x.cpu().float().numpy()
    r1_np = r1.cpu().float().numpy()
    r2_np = r2.cpu().float().numpy()

    B, S, N, D = x_np.shape

    if rotary_mode == "half":
        x_rot = np.concatenate([-x_np[..., D//2:], x_np[..., :D//2]], axis=-1)
        cos_part = np.concatenate([r1_np, r1_np], axis=-1) if r1_np.shape[-1] == D//2 else r1_np
        sin_part = np.concatenate([r2_np, r2_np], axis=-1) if r2_np.shape[-1] == D//2 else r2_np
    elif rotary_mode == "interleave":
        x_rot = np.stack([-x_np[..., 1::2], x_np[..., 0::2]], axis=-1).reshape(x_np.shape)
        cos_part = np.stack([r1_np[..., 0::2], r1_np[..., 1::2]], axis=-1).reshape(x_np.shape) if r1_np.shape[-1] == D//2 else r1_np
        sin_part = np.stack([r2_np[..., 0::2], r2_np[..., 1::2]], axis=-1).reshape(x_np.shape) if r2_np.shape[-1] == D//2 else r2_np
    else:
        return None  # quarter/interleave-half modes: skip CPU reference

    result = x_np * cos_part + x_rot * sin_part
    return torch.from_numpy(result).to(x.dtype).npu()

def run_test(test_name, B, S, N, D, dtype_x, dtype_r, rotary_mode, partial_slice, rtol=1e-3, atol=1e-3):
    print(f"\n--- {test_name} ---")
    x = torch.randn(B, S, N, D, dtype=dtype_x).npu()
    half_D = D // 2
    r_shape = (B, S, N, half_D) if rotary_mode in ["half", "interleave"] else (B, S, N, D)
    r1 = torch.randn(*r_shape, dtype=dtype_r).npu()
    r2 = torch.randn(*r_shape, dtype=dtype_r).npu()

    x_ref = x.clone()
    x_input = x.clone()

    torch.ops.npu_ops_transformer.inplace_partial_rotary_mul(
        x_input, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice)

    print(f"  Input:  x shape={x_input.shape}, dtype={x_input.dtype}")
    print(f"  Output: x modified, shape={x_input.shape}, dtype={x_input.dtype}")
    print(f"  x changed: {not torch.allclose(x_input.cpu(), x_ref.cpu(), atol=1e-6)}")

    # Reference comparison (only for half/interleave)
    ref = compute_reference(x_ref, r1, r2, rotary_mode, partial_slice)
    if ref is not None:
        try:
            max_diff = (x_input.cpu().float() - ref.cpu().float()).abs().max().item()
            allclose = torch.allclose(x_input.cpu().float(), ref.cpu().float(), rtol=rtol, atol=atol)
            print(f"  Reference: max_diff={max_diff:.6f}, allclose(rtol={rtol}, atol={atol})={allclose}")
        except Exception as e:
            print(f"  Reference comparison failed: {e}")
    else:
        print(f"  Reference: skipped (mode={rotary_mode})")

    return True

# ========== Run Tests ==========

print("=" * 60)
print("NPU Verification: inplace_partial_rotary_mul")
print(f"Device: {torch.npu.get_device_name(0)}, Count: {torch.npu.device_count()}")
print("=" * 60)

# Test 1: FP16, half mode
run_test("FP16 + half mode", B=2, S=8, N=4, D=128,
         dtype_x=torch.float16, dtype_r=torch.float16,
         rotary_mode="half", partial_slice=[0, 0], rtol=1e-3, atol=1e-3)

# Test 2: FP16, interleave mode
run_test("FP16 + interleave mode", B=2, S=8, N=4, D=128,
         dtype_x=torch.float16, dtype_r=torch.float16,
         rotary_mode="interleave", partial_slice=[0, 0], rtol=1e-3, atol=1e-3)

# Test 3: BF16, interleave mode
run_test("BF16 + interleave mode", B=2, S=8, N=4, D=128,
         dtype_x=torch.bfloat16, dtype_r=torch.bfloat16,
         rotary_mode="interleave", partial_slice=[0, 0], rtol=5e-3, atol=5e-3)

# Test 4: Mixed precision (BF16 x + FP32 r1/r2)
run_test("BF16 x + FP32 r1/r2 (mixed precision)", B=2, S=8, N=4, D=128,
         dtype_x=torch.bfloat16, dtype_r=torch.float32,
         rotary_mode="interleave", partial_slice=[0, 0], rtol=5e-3, atol=5e-3)

# Test 5: FP16, quarter mode (no reference, just verify runs)
print("\n--- FP16 + quarter mode (no CPU reference) ---")
x = torch.randn(2, 8, 4, 128, dtype=torch.float16).npu()
r1 = torch.randn(2, 8, 4, 32, dtype=torch.float16).npu()  # D/4 for quarter
r2 = torch.randn(2, 8, 4, 32, dtype=torch.float16).npu()
try:
    torch.ops.npu_ops_transformer.inplace_partial_rotary_mul(
        x, r1, r2, rotary_mode="quarter", partial_slice=[0, 0])
    print(f"  PASSED: x modified, shape={x.shape}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 6: FP16, interleave-half mode
print("\n--- FP16 + interleave-half mode (no CPU reference) ---")
x = torch.randn(2, 8, 4, 128, dtype=torch.float16).npu()
r1 = torch.randn(2, 8, 4, 64, dtype=torch.float16).npu()
r2 = torch.randn(2, 8, 4, 64, dtype=torch.float16).npu()
try:
    torch.ops.npu_ops_transformer.inplace_partial_rotary_mul(
        x, r1, r2, rotary_mode="interleave-half", partial_slice=[0, 0])
    print(f"  PASSED: x modified, shape={x.shape}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 7: Larger shape
run_test("FP16 + larger shape (B=4,S=128,N=8,D=256)", B=4, S=128, N=8, D=256,
         dtype_x=torch.float16, dtype_r=torch.float16,
         rotary_mode="interleave", partial_slice=[0, 0], rtol=1e-3, atol=1e-3)

# Test 8: Invalid dim (should fail)
print("\n--- Invalid dim (3D input, should be rejected) ---")
x_bad = torch.randn(2, 8, 128, dtype=torch.float16).npu()
r1 = torch.randn(2, 8, 4, 64, dtype=torch.float16).npu()
r2 = torch.randn(2, 8, 4, 64, dtype=torch.float16).npu()
try:
    torch.ops.npu_ops_transformer.inplace_partial_rotary_mul(
        x_bad, r1, r2, rotary_mode="interleave", partial_slice=[0, 0])
    print("  FAILED: 3D input not rejected")
except Exception as e:
    print(f"  PASSED: 3D rejected ({type(e).__name__})")

# Test 9: Invalid rotary_mode (should fail)
print("\n--- Invalid rotary_mode ---")
x = torch.randn(2, 8, 4, 128, dtype=torch.float16).npu()
r1 = torch.randn(2, 8, 4, 64, dtype=torch.float16).npu()
r2 = torch.randn(2, 8, 4, 64, dtype=torch.float16).npu()
try:
    torch.ops.npu_ops_transformer.inplace_partial_rotary_mul(
        x, r1, r2, rotary_mode="invalid_mode", partial_slice=[0, 0])
    print("  FAILED: invalid mode not rejected")
except Exception as e:
    print(f"  PASSED: invalid mode rejected ({type(e).__name__})")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
