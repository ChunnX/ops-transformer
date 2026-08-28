#!/usr/bin/env python3
"""Insert BSA_MXFP8_DUMP DumpTensor probes. Preserves CRLF."""
from pathlib import Path

ROOT = Path("/home/npu_user5/xuxuepeng/ops-transformer_0703")


def patch_file(rel, replacements):
    p = ROOT / rel
    raw = p.read_bytes()
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    text = raw.replace(b"\r\n", b"\n").decode("utf-8")
    for i, (old, new) in enumerate(replacements):
        if old not in text:
            raise SystemExit(f"{rel}: replacement {i} not found:\n{old[:240]!r}")
        text = text.replace(old, new, 1)
    out = text.encode("utf-8")
    if nl == b"\r\n":
        out = out.replace(b"\n", b"\r\n")
    p.write_bytes(out)
    print("patched", rel)


DUMP_HELPER = r'''
#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 1
#endif
#if BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP_CORE() (AscendC::GetBlockIdx() == 0)
#endif
'''

patch_file(
    "attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h",
    [
        (
            "namespace BsaKernelArch35 {\n",
            "namespace BsaKernelArch35 {\n" + DUMP_HELPER,
        ),
        (
            """                        // AscendC::DumpTensor(ubOTmpTensor, 130, 128);
                        ComputeVec2(delay12TileInfo);""",
            """#if BSA_MXFP8_DUMP
                        if (BSA_MXFP8_DUMP_CORE() && delay12CoreTaskId == 0 && delay12TaskTileIdx == 0) {
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DumpTensor(ubOTmpTensor, 220, 32);
                            AscendC::printf("[MXFP8-DUMP] OTmp desc=220 otmp0=%f lastKvs=%d kvsAct=%u qsAct=%u\\n",
                                static_cast<float>(ubOTmpTensor.GetValue(0)),
                                static_cast<int>(delay12TileInfo.isLastKvsTile),
                                delay12TileInfo.kvsActBaseTile,
                                delay12TaskInfo.qsActBaseTile);
                        }
#endif
                        ComputeVec2(delay12TileInfo);""",
        ),
        (
            """                        // AscendC::DumpTensor(ubSTensor[0], 7771, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128], 7772, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 2], 7773, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 3], 7774, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 4], 7775, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 5], 7776, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 128], 7777, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 129], 7778, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 130], 7779, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 131], 7780, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 256], 7781, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 257], 7782, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 258], 7783, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 259], 7784, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 260], 7785, 128);
                        // AscendC::DumpTensor(ubSTensor[0][128 * 261], 7786, 128);
                        epilogueOnlineSoftmax(""",
            """#if BSA_MXFP8_DUMP
                        if (BSA_MXFP8_DUMP_CORE() && coreTaskId == 0 && curTaskTileIdx == 0) {
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DumpTensor(ubSTensor[0], 200, 32);
                            AscendC::printf("[MXFP8-DUMP] S desc=200 s0=%f s1=%f loop=%u kvsAct=%u qsAct=%u taskTiles=%u\\n",
                                static_cast<float>(ubSTensor[0].GetValue(0)),
                                static_cast<float>(ubSTensor[0].GetValue(1)),
                                static_cast<uint32_t>(coreTileLoop),
                                curTileInfo.kvsActBaseTile,
                                curTaskInfo.qsActBaseTile,
                                curTaskInfo.taskTileNum);
                        }
#endif
                        epilogueOnlineSoftmax(""",
        ),
    ],
)

patch_file(
    "attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_low_prec_fp16_mxfp8.hpp",
    [
        (
            """        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
        //     AscendC::PRINTF("mm1Res \\n");
        // AscendC::DumpTensor(sUBTensor[spBuffIdx][0], 127, 128);
        // AscendC::DumpTensor(sUBTensor[spBuffIdx][256], 127, 128);
        //         AscendC::DumpTensor(sUBTensor[spBuffIdx][512], 127, 128);
        // AscendC::DumpTensor(sUBTensor[spBuffIdx][768], 127, 128);""",
            """        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 1
#endif
#if BSA_MXFP8_DUMP
        if (AscendC::GetBlockIdx() == 0 && tileInfo.loop == 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DumpTensor(sUBTensor[spBuffIdx], 201, 32);
            AscendC::printf("[MXFP8-DUMP] S-epilogue desc=201 s0=%f actKvs=%u actQs=%u\\n",
                static_cast<float>(sUBTensor[spBuffIdx].GetValue(0)), actKvsTile, actQsTile);
        }
#endif""",
        ),
        (
            """        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);

        // DataCopy UB -> L1：NZ 物理视图按整基块 16KB 固定（M=64×N=512，half-N 拷 16KB）""",
            """        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
#if BSA_MXFP8_DUMP
        if (AscendC::GetBlockIdx() == 0 && tileInfo.loop == 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DumpTensor(pUBTensor[spBuffIdx], 210, 32);
            AscendC::DumpTensor(localGroupMaxUBTensor[groupMaxIdx], 211, 32);
            AscendC::printf("[MXFP8-DUMP] P desc=210 p0=%u gmax0=%f\\n",
                static_cast<uint32_t>(pUBTensor[spBuffIdx].GetValue(0)),
                static_cast<float>(localGroupMaxUBTensor[groupMaxIdx].GetValue(0)));
        }
#endif

        // DataCopy UB -> L1：NZ 物理视图按整基块 16KB 固定（M=64×N=512，half-N 拷 16KB）""",
        ),
    ],
)

patch_file(
    "attention/block_sparse_attention/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_arch35_reg_high_prec_mxfp8.hpp",
    [
        (
            """        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {""",
            """        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];
#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 1
#endif
#if BSA_MXFP8_DUMP
        if (AscendC::GetBlockIdx() == 0 && tileInfo.isLastKvsTile) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DumpTensor(oTmpUBTensor, 221, 32);
            AscendC::DumpTensor(localRowSumUBTensor, 230, 32);
            AscendC::DumpTensor(globalRowSumUBTensor, 231, 32);
            AscendC::printf("[MXFP8-DUMP] rescale desc=221/230/231 otmp0=%f lsum0=%f gsum0=%f last=%d\\n",
                static_cast<float>(oTmpUBTensor.GetValue(0)),
                static_cast<float>(localRowSumUBTensor.GetValue(0)),
                static_cast<float>(globalRowSumUBTensor.GetValue(0)),
                static_cast<int>(tileInfo.isLastKvsTile));
        }
#endif

        if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {""",
        ),
        (
            """                Mxfp8VF::processOut<MXFP8::QS_BASE_SIZE, ElementO>(outTensorFp32Ub, oUBTensor, globalRowSumUBTensor);
                AscendC::PipeBarrier<PIPE_V>();""",
            """                Mxfp8VF::processOut<MXFP8::QS_BASE_SIZE, ElementO>(outTensorFp32Ub, oUBTensor, globalRowSumUBTensor);
                AscendC::PipeBarrier<PIPE_V>();
#if BSA_MXFP8_DUMP
                if (AscendC::GetBlockIdx() == 0) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DumpTensor(outTensorFp32Ub, 300, 32);
                    AscendC::printf("[MXFP8-DUMP] O desc=300 o0=%f\\n",
                        static_cast<float>(outTensorFp32Ub.GetValue(0)));
                }
#endif""",
        ),
    ],
)

print("ok")
