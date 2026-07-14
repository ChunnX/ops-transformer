#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import math
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

def check_valid_param(params):
    if params.get('layout_q') not in ["BSND", "TND"]:
        raise ValueError(f"不支持的Q shape: {params.get('layout_q')}")

    if params.get('layout_kv') not in ["BSND", "PA_BBND", "TND"]:
        raise ValueError(f"不支持的KV shape: {params.get('layout_kv')}")

    B = params.get('B')
    cmp_ratio = params.get('cmp_ratio')
    K = params.get('K')
    template_run_mode = params.get('template_run_mode')

    is_swa = template_run_mode == "SWA" or (cmp_ratio is None and (K is None or K == ['None']))

    if params.get('layout_q') == "TND":
        cu_seqlens_q = params.get('cu_seqlens_q')
        T1 = params.get('T1')
        if cu_seqlens_q is None:
            raise ValueError("layout_q为TND时，cu_seqlens_q不能为None")
        if T1 is None:
            raise ValueError("layout_q为TND时，T1不能为None")
        if cu_seqlens_q[-1] != T1:
            raise ValueError(f"cu_seqlens_q最后一维({cu_seqlens_q[-1]})与T1({T1})不一致")
        if len(cu_seqlens_q) != B + 1:
            raise ValueError(f"cu_seqlens_q长度({len(cu_seqlens_q)})应等于B+1({B + 1})")

    if params.get('layout_kv') == "TND":
        cu_seqlens_ori_kv = params.get('cu_seqlens_ori_kv')
        if cu_seqlens_ori_kv is None:
            raise ValueError("layout_kv为TND时，cu_seqlens_ori_kv不能为None")
        T2 = cu_seqlens_ori_kv[-1]
        if len(cu_seqlens_ori_kv) != B + 1:
            raise ValueError(f"cu_seqlens_ori_kv长度({len(cu_seqlens_ori_kv)})应等于B+1({B + 1})")
        if not all(cu_seqlens_ori_kv[i] <= cu_seqlens_ori_kv[i + 1] for i in range(len(cu_seqlens_ori_kv) - 1)):
            raise ValueError(f"cu_seqlens_ori_kv必须单调递增: {cu_seqlens_ori_kv}")

        if not is_swa:
            cu_seqlens_cmp_kv = params.get('cu_seqlens_cmp_kv')
            seqused_cmp_kv = params.get('seqused_cmp_kv')
            if cu_seqlens_cmp_kv is None and seqused_cmp_kv is None:
                raise ValueError("layout_kv为TND且非SWA时，cu_seqlens_cmp_kv或seqused_cmp_kv至少传一个")
            if cu_seqlens_cmp_kv is not None:
                if len(cu_seqlens_cmp_kv) != B + 1:
                    raise ValueError(f"cu_seqlens_cmp_kv长度({len(cu_seqlens_cmp_kv)})应等于B+1({B + 1})")
                if not all(cu_seqlens_cmp_kv[i] <= cu_seqlens_cmp_kv[i + 1] for i in range(len(cu_seqlens_cmp_kv) - 1)):
                    raise ValueError(f"cu_seqlens_cmp_kv必须单调递增: {cu_seqlens_cmp_kv}")

    if params.get('layout_kv') == "PA_BBND":
        seqused_ori_kv = params.get('seqused_ori_kv')
        if seqused_ori_kv is None:
            raise ValueError("layout_kv为PA_BBND时，seqused_ori_kv不能为None")

        if not is_swa:
            seqused_cmp_kv = params.get('seqused_cmp_kv')
            if seqused_cmp_kv is None:
                raise ValueError("layout_kv为PA_BBND且非SWA时，seqused_cmp_kv不能为None")
