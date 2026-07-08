# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代二 穿刺2 (probe_iter2_boundary_maxshape) — CPU golden
#
# 语义 (全 token 有效 k=N，边界上界 shape；聚焦 UB 收敛不溢出 + 多核 sort→MrgSort→分工):
#   gather_idx        = 稳定升序 argsort(expert_ids)   (排序键=专家 ID，稳定序=原始行序 -> 确定性契约)
#   sorted_expert_ids = expert_ids[gather_idx]
#   y                 = token_data[gather_idx]         (原样字节 gather, fp16)
#   session_ids       = id_pack[gather_idx, 0]         (打包 id 第 0 列)
#   micro_batch_ids   = id_pack[gather_idx, 1]         (打包 id 第 1 列)
#   token_ids         = id_pack[gather_idx, 2]         (打包 id 第 2 列)
#   group_count[e]    = 该专家 token 数 (bincount)     (group_listing 计数)
#   actual_token_num  = Y                              (全有效)
# 全部输出与 NPU 结果二进制一致 (np.array_equal, rtol=atol=0)。
# ----------------------------------------------------------------------------------------------------------
import numpy as np

ID_PACK_WIDTH = 8
ID_SESSION_COL = 0
ID_MICRO_COL = 1
ID_TOKEN_COL = 2


def compute_golden(expert_ids, token_data, id_pack, expert_num):
    Y = expert_ids.shape[0]
    # 稳定升序排序：键=expert_id，等键保持原始行序 (== NPU Sort/MrgSort 稳定序确定性契约)
    gather_idx = np.argsort(expert_ids, kind="stable").astype(np.int32)

    sorted_expert_ids = expert_ids[gather_idx].astype(np.int32)
    y = token_data[gather_idx].astype(np.float16)                     # [Y, H] 原样字节 gather
    session_ids = id_pack[gather_idx, ID_SESSION_COL].astype(np.int32)
    micro_batch_ids = id_pack[gather_idx, ID_MICRO_COL].astype(np.int32)
    token_ids = id_pack[gather_idx, ID_TOKEN_COL].astype(np.int32)
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
