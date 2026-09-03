#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立编译 kernel .o 文件，用 readelf 分析 .aicore_binary section 大小。
用于对比不同代码版本的 NPU 指令体积。

使用方法：
  python3 tests/add/compile_kernel.py

输出：
  各版本的 .o 文件大小和 .aicore_binary section 大小
"""

import os
import subprocess
import tempfile

CANN_PATH = os.environ.get("CANN_PATH", "/usr/local/Ascend/cann-9.1.0")


def ccec_compile(src_content, tag, arch="dav-3510"):
    """用 ccec 编译 kernel-only 源码，返回 .o 路径"""
    ccec = f"{CANN_PATH}/compiler/ccec_compiler/bin/ccec"
    includes = [
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/interface",
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/interface/ascendc",
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/impl",
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/impl/ascendc",
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/impl/ascendc/memory",
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/impl/ascendc/operation",
        f"{CANN_PATH}/compiler/tikcpp/tikcfw/impl/ascendc/tensor",
    ]
    inc_args = " ".join([f"-I{p}" for p in includes])

    workdir = tempfile.mkdtemp(prefix=f"kernel_{tag}_")
    src_file = os.path.join(workdir, f"{tag}.cpp")
    out_o = os.path.join(workdir, f"{tag}.o")

    with open(src_file, "w") as f:
        f.write(src_content)

    cmd = (f"{ccec} {src_file} -o {out_o} "
           f"--ccec-aicore-arch={arch} "
           f"-O2 "
           f"--ccec-aicore-only "
           f"-std=c++17 {inc_args} "
           f"-D__DAV_3510__ "
           f"2>&1")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None, r.stderr[-300:] if r.stderr else r.stdout[-300:]
    return out_o, None


def get_section_sizes(out_o):
    """用 readelf 获取各 section 大小"""
    r = subprocess.run(f"readelf -S {out_o}", shell=True, capture_output=True, text=True)
    sections = {}
    for line in r.stdout.split("\n"):
        if ".text" in line or ".aicore" in line or ".data" in line or ".rodata" in line:
            parts = line.split()
            if len(parts) >= 6:
                name = parts[1] if len(parts) > 2 else ""
                try:
                    size = int(parts[5], 16)
                    sections[name] = size
                except:
                    pass
    # 也获取总文件大小
    sections["_file_size"] = os.path.getsize(out_o)
    return sections


def generate_kernel_sources():
    """生成各测试版本的 kernel-only 源码（仅 kernel 函数，不含 host 端）"""

    common_header = '''
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
__global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength,
                                       int64_t blockLength, uint32_t tileSize)
{
'''

    # K1: 纯 prologue（注释版）
    k1 = common_header + '''
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr static int64_t PIPELINE_DEPTH = 2;
    TPipe pipe;
    GlobalTensor<T> xGm, yGm, zGm;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * GetBlockIdx());
    int64_t currentBlockLength = totalLength - GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;
}
'''

    # K2: 双return（for体内return + tail内return，无DMA无Add）
    k2 = common_header + '''
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr static int64_t PIPELINE_DEPTH = 2;
    TPipe pipe;
    GlobalTensor<T> xGm, yGm, zGm;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * GetBlockIdx());
    int64_t currentBlockLength = totalLength - GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;
    for (int64_t i = 0; i < tileNum; ++i) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        return;
    }
    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        return;
    }
}
'''

    # K3: return死代码 + tail完整执行
    k3 = common_header + '''
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr static int64_t PIPELINE_DEPTH = 2;
    TPipe pipe;
    GlobalTensor<T> xGm, yGm, zGm;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * GetBlockIdx());
    int64_t currentBlockLength = totalLength - GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elementNumPerTile * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        return;
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
    }
    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tailTileElementNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        Add(zLocal, xLocal, yLocal, tailTileElementNum);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        zLocal = outQueueZ.DeQue<T>();
        DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
}
'''

    # K4: 完整 kernel（无 return）
    k4 = common_header + '''
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr static int64_t PIPELINE_DEPTH = 2;
    TPipe pipe;
    GlobalTensor<T> xGm, yGm, zGm;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * GetBlockIdx());
    int64_t currentBlockLength = totalLength - GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elementNumPerTile * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        Add(zLocal, xLocal, yLocal, elementNumPerTile);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        zLocal = outQueueZ.DeQue<T>();
        DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tailTileElementNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        Add(zLocal, xLocal, yLocal, tailTileElementNum);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        zLocal = outQueueZ.DeQue<T>();
        DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
}
'''

    return [
        ("K1_prologue", k1, "纯prologue(注释版)"),
        ("K2_dual_ret", k2, "双return(无DMA无Add)"),
        ("K3_dead_ret_tail", k3, "return死代码+tail完整执行"),
        ("K4_full", k4, "完整kernel(无return)"),
    ]


def main():
    versions = generate_kernel_sources()

    print(f"{'Version':<25} {'Description':<35} {'Compile':>8} {'.o Size':>10} {'.aicore':>10}")
    print("-" * 95)
    for tag, src, desc in versions:
        out_o, err = ccec_compile(src, tag)
        if out_o is None:
            print(f"{tag:<25} {desc:<35} {'FAIL':>8}  {err[:80]}")
            continue
        sections = get_section_sizes(out_o)
        file_size = sections.get("_file_size", 0)
        aicore = sections.get(".aicore_binary", 0)
        print(f"{tag:<25} {desc:<35} {'OK':>8} {file_size:>8} B {aicore:>8} B")

    print(f"\n.aicore_binary = 真正的 NPU 执行指令段")
    print(f"return 本身几乎零开销：K1→K2 仅增加 AllocTensor 指令")
    print(f"tail 块 DMA+Add 贡献主要指令体积：K2→K3 增量")
    print(f"for 体内死代码也被编译：K3→K4 增量")


if __name__ == "__main__":
    main()
