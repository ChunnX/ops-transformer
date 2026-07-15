"""
最小复现：gemm_v0 transpose_A=True 的 nZ/zN 分形布局不兼容 bug

现象：transpose_A=True 时 gemm_v0 产出严重错误结果，
      手工预转置 + transpose_A=False 则结果正确。

根因：copy_l1_to_l0a 在 transpose=true 时以 nZ 布局解读 L1 buffer，
      但 L1 buffer 由 alloc_L1 以 zN 分配，分形 tile 边界不兼容。

前置条件：需要 NPU 环境 (torch_npu)
用法：    python repro_transpose_a_bug.py
"""

import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()


def _make_gemm_kernel(M=64, K_gemm=64, N=128):
    """
    构造最小 gemm kernel：
      Buggy:   C[M,N] = A[M,K]^T @ B[K,N]   (transpose_A=True)
      Correct: C[M,N] = At[K,M]   @ B[K,N]   (transpose_A=False, At 已预转置)
    """

    @tilelang.jit(out_idx=[-2, -1])
    def kernel(
        M: int, K_gemm: int, N: int,
    ):
        @T.prim_func
        def main(
            A: T.Tensor((M, K_gemm), "float16"),
            At: T.Tensor((K_gemm, M), "float16"),
            B: T.Tensor((K_gemm, N), "float16"),
            C_buggy: T.Tensor((M, N), "float32"),
            C_correct: T.Tensor((M, N), "float32"),
        ):
            with T.Scope("C"):
                A_L1 = T.alloc_L1((M, K_gemm), "float16")
                At_L1 = T.alloc_L1((K_gemm, M), "float16")
                B_L1 = T.alloc_L1((K_gemm, N), "float16")
                C0_L0C = T.alloc_L0C((M, N), "float32")
                C1_L0C = T.alloc_L0C((M, N), "float32")

                T.copy(A, A_L1)
                T.copy(At, At_L1)
                T.copy(B, B_L1)
                T.set_flag("mte2", "m", 0)
                T.wait_flag("mte2", "m", 0)

                # === Buggy: transpose_A=True ===
                T.gemm_v0(A_L1, B_L1, C0_L0C,
                          transpose_A=True, transpose_B=False, init=True)

                # === Correct: transpose_A=False + pre-transposed At ===
                T.gemm_v0(At_L1, B_L1, C1_L0C,
                          transpose_A=False, transpose_B=False, init=True)

                T.set_flag("m", "fix", 0)
                T.wait_flag("m", "fix", 0)

                T.copy(C0_L0C, C_buggy)
                T.copy(C1_L0C, C_correct)

    return kernel(M, K_gemm, N)


def test_transpose_a_bug():
    M, K_gemm, N = 64, 64, 128
    dtype = torch.float16

    torch.manual_seed(42)
    A = torch.randn(M, K_gemm, dtype=dtype).npu()
    At = A.T.contiguous().clone().npu()          # 预转置: (K, M)
    B = torch.randn(K_gemm, N, dtype=dtype).npu()
    C_buggy = torch.zeros(M, N, dtype=torch.float32).npu()
    C_correct = torch.zeros(M, N, dtype=torch.float32).npu()

    func = _make_gemm_kernel(M, K_gemm, N)
    func(A, At, B, C_buggy, C_correct)
    torch.npu.synchronize()

    # PyTorch reference
    ref = (A.T.to(torch.float32) @ B.to(torch.float32))

    buggy_cpu = C_buggy.cpu()
    correct_cpu = C_correct.cpu()
    ref_cpu = ref.cpu()

    # ── 打印 ──────────────────────────────────────────────────
    print("=" * 72)
    print("gemm_v0 transpose_A=True bug 最小复现")
    print("=" * 72)
    print(f"矩阵: C[{M},{N}] = A[{M},{K_gemm}]^T @ B[{K_gemm},{N}]")
    print()

    buggy_diff = (buggy_cpu - ref_cpu).abs()
    correct_diff = (correct_cpu - ref_cpu).abs()
    ref_max = ref_cpu.abs().max().item()

    # 1. 全局统计
    print("┌─ 全局统计 ──────────────────────────────────────────┐")
    print(f"│ Reference 值域:   [{ref_cpu.min().item():+.4f}, {ref_cpu.max().item():+.4f}]")
    print(f"│ Buggy     值域:   [{buggy_cpu.min().item():+.4f}, {buggy_cpu.max().item():+.4f}]")
    print(f"│ Correct   值域:   [{correct_cpu.min().item():+.4f}, {correct_cpu.max().item():+.4f}]")
    print("│")
    print("│              max|error|    mean|error|   rel(max)")
    print(f"│ Buggy   :   {buggy_diff.max().item():.4f}       {buggy_diff.mean().item():.4f}       {buggy_diff.max().item() / max(ref_max, 1e-10):.2e}")
    print(f"│ Correct :   {correct_diff.max().item():.4f}       {correct_diff.mean().item():.4f}       {correct_diff.max().item() / max(ref_max, 1e-10):.2e}")
    print("└──────────────────────────────────────────────────────┘")
    print()

    # 2. 逐行误差
    buggy_row_err = (buggy_cpu - ref_cpu).abs().max(dim=1).values
    correct_row_err = (correct_cpu - ref_cpu).abs().max(dim=1).values
    print("┌─ 逐行 max|error| 对比 (前16行) ─────────────────────┐")
    print(f"│ {'Row':<4} {'Buggy':>10} {'Correct':>10}   {'Status'}")
    for i in range(min(16, M)):
        status = "✗✗✗" if buggy_row_err[i] > ref_max * 0.1 else "  ✓ "
        print(f"│ {i:<4} {buggy_row_err[i]:>10.4f} {correct_row_err[i]:>10.6f}   {status}")
    print("└──────────────────────────────────────────────────────┘")
    print()

    # 3. 左上角 8×8 并排
    K_PRINT = 8
    print("┌─ 左上角 8×8 子矩阵并排 ─────────────────────────────┐")
    print("│  Reference              Buggy                   Correct")
    print("│  ─────────              ─────                   ───────")
    for i in range(K_PRINT):
        ref_row = " ".join(f"{ref_cpu[i, j]:+6.2f}" for j in range(K_PRINT))
        bug_row = " ".join(f"{buggy_cpu[i, j]:+6.2f}" for j in range(K_PRINT))
        cor_row = " ".join(f"{correct_cpu[i, j]:+6.2f}" for j in range(K_PRINT))
        print(f"│  {ref_row}   {bug_row}   {cor_row}")
    print("└──────────────────────────────────────────────────────┘")
    print()

    # 4. 结论
    buggy_rel = buggy_diff.max().item() / max(ref_max, 1e-10)
    correct_rel = correct_diff.max().item() / max(ref_max, 1e-10)

    print("┌─ 结论 ──────────────────────────────────────────────┐")
    if correct_rel < 1e-4:
        print("│ ✓ pre-transpose + transpose_A=False  结果正确       │")
    else:
        print("│ ✗ pre-transpose 路径异常！                          │")

    if buggy_rel > 0.1:
        print(f"│ ✗ transpose_A=True  结果严重错误  (rel={buggy_rel:.2e})      │")
        print("│                                                    │")
        print("│ 根因:                                              │")
        print("│   L1 buffer 由 alloc_L1 以 zN 分形布局分配，        │")
        print("│   copy_l1_to_l0a(transpose=true) 以 nZ 布局解读，   │")
        print("│   两种分形 tile 边界不兼容 → 读到错位数据。          │")
        print("│                                                    │")
        print("│ 影响范围:                                          │")
        print("│   所有 gemm_v0(transpose_A=True) 调用，              │")
        print("│   包括 kL0split=1 的场景。                           │")
        print("│                                                    │")
        print("│ >>> BUG CONFIRMED: transpose_A=True 不可用 <<<      │")
    elif buggy_rel < 1e-4:
        print("│ ✓ transpose_A=True 结果正确（当前维度未复现）        │")
    else:
        print(f"│ ? transpose_A=True 部分偏离 (rel={buggy_rel:.2e})  │")
    print("└──────────────────────────────────────────────────────┘")

    return buggy_rel > 0.1


if __name__ == "__main__":
    test_transpose_a_bug()
