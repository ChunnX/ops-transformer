# QLI v2 FD/LD Dump Guide

当前调试版本额外覆盖长S2场景下的FD分片和LD合并。目标行必须同时满足kernel编译期宏与
Python环境变量一致：

    QLIV2_DEBUG_TARGET_B_IDX   <-> QLIV2_DEBUG_B_IDX
    QLIV2_DEBUG_TARGET_N2_IDX  <-> QLIV2_DEBUG_N2_IDX
    QLIV2_DEBUG_TARGET_S1_IDX  <-> QLIV2_DEBUG_S1_IDX

重新编译算子后运行单测，将完整日志交给解析脚本：

    python3 attention/quant_lightning_indexer_v2/tests/pytest/parse_qliv2_debug_dump.py \
      mxfp4_case.log -o qliv2_npu_dump

## Descriptor

    590000                 完整QLI/QLD metadata
    6100000 + partial_id   LD分片写入元数据
    6200000 + partial_id   写workspace前的局部TopK score key
    6300000 + partial_id   写workspace前的局部TopK index
    6400000 + round        ProcessLD每轮读取元数据
    6500000 + round        ProcessLD每轮读取并完成padding后的score key
    6600000 + round        ProcessLD每轮读取并完成padding后的index
    6700000 + round        每轮LdTopK合并后的score key
    6800000 + round        每轮LdTopK合并后的index
    600000/600001/600002   LD最终index、score key和BF16 value

partial_id = saveWorkspaceIdx * 4 + rowIdx。LD目标绝对行号按
mIdx * s1BaseSize + mStart + j计算，不能只使用mStart + j。

## Metadata

分片元数据的12个uint32依次为：

    version, aivBlockId, bIdx, n2Idx, absoluteS1Idx, curS2StartIdx,
    validS2Len, saveWorkspaceIdx, rowIdx, workspaceElementOffset,
    topkCount, topkCountAlign16

每轮LD读取元数据的12个uint32依次为：

    version, aivBlockId, bIdx, n2Idx, absoluteS1Idx, workspaceBegin,
    workspaceNum, roundIdx, previousTopkOffset, topkCountAlign16,
    validReadLen, alignedReadLen

## Automatic Verification

解析器还会将LD最终BF16 value保存为`npu_ld_topk_value_kernel.npy`，用于和
`npu_topk_score_kernel.npy`对照，判断异常发生在LD合并阶段还是score key还原阶段。

解析器会自动输出[LD verify]结果：

- score_equal/index_equal检查每轮读取内容是否等于各核写入前的局部TopK。
- score_zero/index_neg1检查对齐尾部是否正确刷成0/-1。
- ld_round_*_merged_*用于定位LdTopK在哪一轮首次出现异常。
- 没有LD partial records表示目标行没有进入FD/LD路径，需检查metadata或目标行配置。
