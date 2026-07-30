#!/usr/bin/env python3
"""
mhc_pre_backward 算子内存占用计算脚本
支持确定性场景和非确定性场景
支持 BSN!(N!+2N) 和 BSNN(N²+2N) 两种 fusionSize 公式
"""

import math
import argparse
from dataclasses import dataclass, field
from typing import List


# ========== 常量定义 (来自 kernel 代码) ==========
BYTE_SIZE_PER_BLOCK = 32
ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK // 4  # 8 (float32)
BYTE_SIZE_PER_REPEAT = 256
ELEMENTS_SIZE_PER_REPEAT = 256 // 4  # 64
PRE_POST_NUM = 2
INNER_SPILT_NUM = 8
C0 = 256  # Vector Length (VL), 硬编码

# A2 平台典型值
A2_AIV_NUM = 48
A2_AIC_NUM = 24
A2_UB_SIZE = 192 * 1024  # 192 KB per core
A2_SYSTEM_WORKSPACE = 16 * 1024 * 1024  # 16 MB


def factorial(n):
    if n <= 1:
        return 1
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def ceil_div(a, b):
    return (a + b - 1) // b


@dataclass
class MemItem:
    name: str
    size: int
    desc: str = ""


@dataclass
class MemSummary:
    category: str
    items: List[MemItem] = field(default_factory=list)

    @property
    def total(self):
        return sum(item.size for item in self.items)


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def compute_ub_memory(N, D, fusionSize, tile=32, dtype_bytes=2):
    """
    计算 UB (Unified Buffer) 每核内存占用
    dtype_bytes: 输入数据类型大小 (bf16=2, fp16=2, fp32=4)
    """
    c0 = C0
    onceTask = tile // INNER_SPILT_NUM  # 32/8 = 4
    f4 = 4  # sizeof(float)

    ub = MemSummary(category="UB (Unified Buffer) — 每核占用")

    # 1. fusedGradHPre2AndGradHPost2Buf_
    size = tile * N * 2 * f4
    ub.items.append(
        MemItem(
            "fusedGradHPre2AndGradHPost2Buf_",
            size,
            f"tile({tile}) * N({N}) * 2 * sizeof(float)",
        )
    )

    # 2. gradRsqrtBuf_
    size = tile * fusionSize * f4 * 2
    ub.items.append(
        MemItem(
            "gradRsqrtBuf_",
            size,
            f"tile({tile}) * fusionSize({fusionSize}) * sizeof(float) * 2",
        )
    )

    # 3. gradBiasBuf_
    size = 2 * 2 * tile * fusionSize * f4
    ub.items.append(
        MemItem(
            "gradBiasBuf_",
            size,
            f"2 * 2 * tile({tile}) * fusionSize({fusionSize}) * sizeof(float)",
        )
    )

    # 4. onesBuf_
    size = tile * N * 2 * f4
    ub.items.append(
        MemItem("onesBuf_", size, f"tile({tile}) * N({N}) * 2 * sizeof(float)")
    )

    # 5. hcBaseBuf_
    size = fusionSize * f4
    ub.items.append(
        MemItem("hcBaseBuf_", size, f"fusionSize({fusionSize}) * sizeof(float)")
    )

    # 6. ScaleBuf_
    size = BYTE_SIZE_PER_BLOCK * 2
    ub.items.append(
        MemItem("ScaleBuf_", size, f"BYTE_SIZE_PER_BLOCK({BYTE_SIZE_PER_BLOCK}) * 2")
    )

    # 7. inputXInQueue (double buffer, T is bf16/fp16)
    # InitBuffer(inputXInQueue, 2, tile * N * c0 * sizeof(float) / 4)
    # /4 because T is 2 bytes, float is 4 bytes → T_buf = float_buf / 2, but double buffer → *2
    # Actually: sizeof(float)/4 = 1, so it's tile * N * c0 bytes per buffer? No...
    # The code: tileCoreBS_ * n_ * c0_ * sizeof(float) / 4
    # This means: tile * N * c0 * 4 / 4 = tile * N * c0 bytes per buffer
    # But T is bf16 (2 bytes), so tile * N * c0 elements of T = tile * N * c0 * 2 bytes
    # The /4 comes from sizeof(float)=4, so it's tile * N * c0 * 1 = tile * N * c0 bytes
    # This is the byte count passed to InitBuffer
    per_buf = tile * N * c0 * f4 // 4  # = tile * N * c0 bytes
    size = 2 * per_buf  # double buffer
    ub.items.append(
        MemItem(
            "inputXInQueue",
            size,
            f"2 * tile({tile}) * N({N}) * c0({c0}) * sizeof(float) / 4 (double buffer, T={dtype_bytes}B)",
        )
    )

    # 8. inputGradQueue (single buffer)
    per_buf = tile * N * ELEMENTS_SIZE_PER_BLOCK * f4
    size = 1 * per_buf
    ub.items.append(
        MemItem(
            "inputGradQueue",
            size,
            f"1 * tile({tile}) * N({N}) * ELEM_PER_BLOCK({ELEMENTS_SIZE_PER_BLOCK}) * sizeof(float)",
        )
    )

    # 9. OutQueue (double buffer, output is T)
    per_buf = tile * N * c0 * f4 // 8
    size = 2 * per_buf
    ub.items.append(
        MemItem(
            "OutQueue",
            size,
            f"2 * tile({tile}) * N({N}) * c0({c0}) * sizeof(float) / 8 (double buffer, T={dtype_bytes}B)",
        )
    )

    # 10. tempBuf_ (= ubSizeRemain)
    # ubSizeRemain = CeilDiv(tile, 8) * 64 * 4      (dRsqrtBrcb)
    #              + CeilDiv(tile*N, 8) * 64 * 4     (preBrcb)
    #              + onceTask * N * c0 * 4 * 2       (gradXCast + xCast)
    #              + onceTask * c0 * 4               (gradYCast)
    term1 = ceil_div(tile, ELEMENTS_SIZE_PER_BLOCK) * ELEMENTS_SIZE_PER_REPEAT * f4
    term2 = ceil_div(tile * N, ELEMENTS_SIZE_PER_BLOCK) * ELEMENTS_SIZE_PER_REPEAT * f4
    term3 = onceTask * N * c0 * f4 * 2
    term4 = onceTask * c0 * f4
    ub_size_remain = term1 + term2 + term3 + term4

    # dhat 系列缓冲区 (offset=0 重置, 与上面的缓冲区重叠使用)
    dhat_local = tile * fusionSize * 2 * f4
    rsqrtbrcb = (
        ceil_div(tile, ELEMENTS_SIZE_PER_BLOCK)
        * ELEMENTS_SIZE_PER_BLOCK
        * fusionSize
        * f4
    )
    hat2scale = tile * fusionSize * f4
    dhat_before_norm = tile * fusionSize * f4
    hat_local = tile * fusionSize * f4
    rsqrt_temp = tile * fusionSize * f4
    dhat_total = (
        dhat_local + rsqrtbrcb + hat2scale + dhat_before_norm + hat_local + rsqrt_temp
    )

    temp_buf_size = max(ub_size_remain, dhat_total)
    ub.items.append(
        MemItem(
            "tempBuf_",
            temp_buf_size,
            f"max(ubSizeRemain={ub_size_remain}, dhatTotal={dhat_total})",
        )
    )

    return ub


def compute_workspace(
    B, S, N, D, fusionSize, deterministic=True, aivNum=A2_AIV_NUM, aicNum=A2_AIC_NUM
):
    """
    计算 GM Workspace 内存占用
    """
    f4 = 4  # sizeof(float)
    ws = MemSummary(
        category=f"GM Workspace ({'确定性' if deterministic else '非确定性'})"
    )

    # xCastWorkspace = B * S * N * D * sizeof(float) * 2
    x_cast_ws = B * S * N * D * f4 * 2
    ws.items.append(
        MemItem(
            "xCastWorkspace",
            x_cast_ws,
            f"B({B}) * S({S}) * N({N}) * D({D}) * sizeof(float) * 2",
        )
    )

    # gradHat2Workspace = B * S * fusionSize * sizeof(float)
    grad_hat2_ws = B * S * fusionSize * f4
    ws.items.append(
        MemItem(
            "gradHat2Workspace",
            grad_hat2_ws,
            f"B({B}) * S({S}) * fusionSize({fusionSize}) * sizeof(float)",
        )
    )

    # systemWorkspace
    ws.items.append(
        MemItem(
            "systemWorkspace",
            A2_SYSTEM_WORKSPACE,
            "GetLibApiWorkSpaceSize() (A2 典型值)",
        )
    )

    if deterministic:
        # 确定性场景: 额外 workspace
        extra_aic = aicNum * fusionSize * (N * D) * f4
        ws.items.append(
            MemItem(
                "determ_extra_aic",
                extra_aic,
                f"aicNum({aicNum}) * fusionSize({fusionSize}) * (N*D={N * D}) * sizeof(float)",
            )
        )

        extra_aiv = aivNum * (fusionSize + ELEMENTS_SIZE_PER_BLOCK) * f4
        ws.items.append(
            MemItem(
                "determ_extra_aiv",
                extra_aiv,
                f"aivNum({aivNum}) * (fusionSize+8={fusionSize + ELEMENTS_SIZE_PER_BLOCK}) * sizeof(float)",
            )
        )

    return ws


def compute_io_memory(B, S, N, D, fusionSize, factN, dtype_bytes=2):
    """
    计算输入输出张量内存占用 (GM)
    """
    f4 = 4  # sizeof(float)
    io = MemSummary(category="输入输出张量 (GM)")

    # --- Inputs ---
    # x: (B, S, N, D) - T type
    io.items.append(
        MemItem(
            "x (in)", B * S * N * D * dtype_bytes, f"({B},{S},{N},{D}) * {dtype_bytes}B"
        )
    )

    # phi: (fusionSize, N*D) - float32
    io.items.append(
        MemItem(
            "phi (in)", fusionSize * N * D * f4, f"({fusionSize},{N * D}) * float32"
        )
    )

    # alpha: (3,) - float32
    io.items.append(MemItem("alpha (in)", 3 * f4, "(3,) * float32"))

    # gradHin: (B, S, D) - T type
    io.items.append(
        MemItem(
            "grad_hin (in)", B * S * D * dtype_bytes, f"({B},{S},{D}) * {dtype_bytes}B"
        )
    )

    # gradHPost: (B, S, N) - float32
    io.items.append(
        MemItem("grad_h_post (in)", B * S * N * f4, f"({B},{S},{N}) * float32")
    )

    # gradHRes: (B, S, factN) BSN! or (B, S, N, N) BSNN - float32
    io.items.append(
        MemItem("grad_h_res (in)", B * S * factN * f4, f"({B},{S},{factN}) * float32")
    )

    # invRms: (B, S, 1) - float32
    io.items.append(MemItem("inv_rms (in)", B * S * 1 * f4, f"({B},{S},1) * float32"))

    # hMix: (B, S, fusionSize) - float32
    io.items.append(
        MemItem(
            "h_mix (in)", B * S * fusionSize * f4, f"({B},{S},{fusionSize}) * float32"
        )
    )

    # hPre: (B, S, N) - float32
    io.items.append(MemItem("h_pre (in)", B * S * N * f4, f"({B},{S},{N}) * float32"))

    # hPost: (B, S, N) - float32
    io.items.append(MemItem("h_post (in)", B * S * N * f4, f"({B},{S},{N}) * float32"))

    # gamma: (N, D) - float32
    io.items.append(MemItem("gamma (in)", N * D * f4, f"({N},{D}) * float32"))

    # gradXPost: (B, S, N, D) - T type
    io.items.append(
        MemItem(
            "grad_x_post (in)",
            B * S * N * D * dtype_bytes,
            f"({B},{S},{N},{D}) * {dtype_bytes}B",
        )
    )

    # --- Outputs ---
    # gradX: (B, S, N, D) - T type
    io.items.append(
        MemItem(
            "grad_x (out)",
            B * S * N * D * dtype_bytes,
            f"({B},{S},{N},{D}) * {dtype_bytes}B",
        )
    )

    # gradHCfn (phi_grad): (fusionSize, N*D) - float32
    io.items.append(
        MemItem(
            "grad_phi (out)",
            fusionSize * N * D * f4,
            f"({fusionSize},{N * D}) * float32",
        )
    )

    # gradHCscale: (3,) - float32
    io.items.append(MemItem("grad_alpha (out)", 3 * f4, "(3,) * float32"))

    # gradHCbase: (fusionSize,) - float32
    io.items.append(
        MemItem("grad_bias (out)", fusionSize * f4, f"({fusionSize},) * float32")
    )

    # gradGamma: (N, D) - float32
    io.items.append(MemItem("grad_gamma (out)", N * D * f4, f"({N},{D}) * float32"))

    return io


def print_summary(summary, indent=2):
    prefix = " " * indent
    print(f"\n{'=' * 70}")
    print(f"  {summary.category}")
    print(f"{'=' * 70}")
    for item in summary.items:
        print(f"{prefix}{item.name:40s} {format_size(item.size):>12s}   # {item.desc}")
    print(f"{'-' * 70}")
    print(f"{prefix}{'小计':40s} {format_size(summary.total):>12s}")


def main():
    parser = argparse.ArgumentParser(description="mhc_pre_backward 算子内存占用计算")
    parser.add_argument("-B", type=int, default=4, help="Batch size (默认: 4)")
    parser.add_argument("-S", type=int, default=4, help="Sequence length (默认: 4)")
    parser.add_argument("-N", type=int, default=4, help="残差流维度 N (默认: 4)")
    parser.add_argument(
        "-D", type=int, default=512, help="D 维度, 必须被128整除 (默认: 512)"
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="输入数据类型 (默认: bf16)",
    )
    parser.add_argument(
        "--formula",
        choices=["factorial", "square", "both"],
        default="both",
        help="fusionSize 公式: factorial=N!+2N, square=N²+2N, both=两者都算 (默认: both)",
    )
    parser.add_argument(
        "--aiv", type=int, default=A2_AIV_NUM, help=f"AIV 核数 (默认: {A2_AIV_NUM})"
    )
    parser.add_argument(
        "--aic", type=int, default=A2_AIC_NUM, help=f"AIC 核数 (默认: {A2_AIC_NUM})"
    )
    args = parser.parse_args()

    dtype_bytes = {"bf16": 2, "fp16": 2, "fp32": 4}[args.dtype]

    if args.D % 128 != 0:
        print(f"错误: D={args.D} 必须被 128 整除")
        return

    formulas = []
    if args.formula in ("factorial", "both"):
        formulas.append(
            ("N!+2N (BSN!)", factorial(args.N) + 2 * args.N, factorial(args.N))
        )
    if args.formula in ("square", "both"):
        formulas.append(("N²+2N (BSNN)", args.N * args.N + 2 * args.N, args.N * args.N))

    print(f"\n{'#' * 70}")
    print("#  mhc_pre_backward 算子内存占用计算")
    print(
        f"#  参数: B={args.B}, S={args.S}, N={args.N}, D={args.D}, dtype={args.dtype}"
    )
    print(
        f"#  平台: A2 (aivNum={args.aiv}, aicNum={args.aic}, UB={format_size(A2_UB_SIZE)})"
    )
    print(f"{'#' * 70}")

    for formula_name, fusionSize, factN in formulas:
        print(f"\n\n{'*' * 70}")
        print(f"*  公式: {formula_name}, fusionSize={fusionSize}, factN={factN}")
        print(f"{'*' * 70}")

        # 1. UB 内存
        ub = compute_ub_memory(args.N, args.D, fusionSize, dtype_bytes=dtype_bytes)
        print_summary(ub)
        ub_usage_pct = ub.total / A2_UB_SIZE * 100
        print(
            f"    UB 使用率: {ub_usage_pct:.1f}% ({format_size(ub.total)} / {format_size(A2_UB_SIZE)})"
        )

        # 2. 输入输出张量
        io = compute_io_memory(
            args.B, args.S, args.N, args.D, fusionSize, factN, dtype_bytes
        )
        print_summary(io)

        # 3. Workspace - 确定性
        ws_det = compute_workspace(
            args.B,
            args.S,
            args.N,
            args.D,
            fusionSize,
            deterministic=True,
            aivNum=args.aiv,
            aicNum=args.aic,
        )
        print_summary(ws_det)

        # 4. Workspace - 非确定性
        ws_nondet = compute_workspace(
            args.B,
            args.S,
            args.N,
            args.D,
            fusionSize,
            deterministic=False,
            aivNum=args.aiv,
            aicNum=args.aic,
        )
        print_summary(ws_nondet)

        # 5. 总计
        print(f"\n{'=' * 70}")
        print(f"  总计 ({formula_name})")
        print(f"{'=' * 70}")
        total_io = io.total
        total_ub = ub.total * args.aiv  # 所有 AIV 核
        total_ws_det = ws_det.total
        total_ws_nondet = ws_nondet.total

        print(f"  输入输出张量 (GM):          {format_size(total_io):>12s}")
        print(
            f"  UB 总量 (所有AIV核):        {format_size(total_ub):>12s}  ({format_size(ub.total)}/core * {args.aiv} cores)"
        )
        print(f"  Workspace (确定性):         {format_size(total_ws_det):>12s}")
        print(f"  Workspace (非确定性):       {format_size(total_ws_nondet):>12s}")
        print(f"  {'-' * 50}")
        print(
            f"  GM 总占用 (确定性):         {format_size(total_io + total_ws_det):>12s}"
        )
        print(
            f"  GM 总占用 (非确定性):       {format_size(total_io + total_ws_nondet):>12s}"
        )

        # 确定性 vs 非确定性 差异
        diff = total_ws_det - total_ws_nondet
        print(
            f"\n  确定性额外开销:             {format_size(diff):>12s}  ({diff / total_ws_nondet * 100:.1f}% 增加)"
        )


if __name__ == "__main__":
    main()
