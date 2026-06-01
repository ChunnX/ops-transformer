#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

TestCases = {
    'BASE_04': {
        'B': [1],
        'N1': [19],
        'N2': [19],
        'S1': [1024],
        'S2': [1024],
        'D': [128],
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'mask_mode': [0],
        'seqused_q': [[640]],
        'seqused_kv': [[1024]],
        'q_range': [(-10, 10)],
        'k_range': [(-10, 10)],
        'v_range': [(-10, 10)],
    },
}
