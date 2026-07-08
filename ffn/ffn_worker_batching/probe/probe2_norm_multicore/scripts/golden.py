# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代一 穿刺2 (probe2_norm_multicore) — CPU golden
#
# 语义 (全 token 有效，无 mask；聚焦多核 sort -> MrgSort 4 路归并 -> 核间 gather/group_listing 分工):
#   gather_idx        = 稳定升序 argsort(expert_ids)   (排序键=专家 ID，稳定序=原始行序 -> 确定性契约)
#   sorted_expert_ids = expert_ids[gather_idx]
#   y                 = token_data[gather_idx]  (原样字节 gather, fp16)
#   session_ids       = session_src[gather_idx]
#   micro_batch_ids   = micro_batch_src[gather_idx]
#   token_ids         = token_src[gather_idx]
#   group_count[e]    = 该专家 token 数 (bincount)      (group_listing 计数)
#   actual_token_num  = Y                              (全有效)
# 全部输出与 NPU 结果二进制一致 (np.array_equal, rtol=atol=0)。
# ----------------------------------------------------------------------------------------------------------
import numpy as np


def compute_golden(expert_ids, token_data, session_src, micro_batch_src, token_src, expert_num):
    Y = expert_ids.shape[0]
    # 稳定升序排序：键=expert_id，等键保持原始行序 (== NPU Sort/MrgSort 稳定序确定性契约)
    gather_idx = np.argsort(expert_ids, kind="stable").astype(np.int32)

    sorted_expert_ids = expert_ids[gather_idx].astype(np.int32)
    y = token_data[gather_idx].astype(np.float16)                     # [Y, H] 原样字节 gather
    session_ids = session_src[gather_idx].astype(np.int32)
    micro_batch_ids = micro_batch_src[gather_idx].astype(np.int32)
    token_ids = token_src[gather_idx].astype(np.int32)
    group_count = np.bincount(expert_ids, minlength=expert_num).astype(np.int32)
    actual_token_num = np.array([Y], dtype=np.int32)

    return {
        "sorted_expert_ids": sorted_expert_ids,
        "gather_idx": gather_idx,
        "y": y,
        "session_ids": session_ids,
        "micro_batch_ids": micro_batch_ids,
        "token_ids": token_ids,
        "group_count": group_count,
        "actual_token_num": actual_token_num,
    }
