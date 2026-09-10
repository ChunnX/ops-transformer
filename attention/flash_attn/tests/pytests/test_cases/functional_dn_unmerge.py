# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# DN 不合轴(G对齐切块)专项用例: 无mask, BNSD(GS布局)块不跨G边界
# 覆盖: 整G块/混合步进/g=1回归/尾块小G

BASE = {
    "B": [2],
    "N1": [8],
    "N2": [1],
    "S1": [128],
    "S2": [1024],
    "D": [128],
    "DV": [128],
    "Dtype": ["fp16"],
    "layout_q": ["BNSD"],
    "layout_kv": ["BNSD"],
    "layout_out": ["BNSD"],
    "block_size": [-1],
    "return_softmax_lse": [False],
    "enable_learnable_sink": [False],
    "cu_seqlens_q": [[None]],
    "cu_seqlens_kv": [[None]],
    "seqused_q": [[None]],
    "seqused_kv": [[None]],
    "mask_mode": [0],
    "win_left": [-1],
    "win_right": [-1],
    "q_range": [(-5.0, 5.0)],
    "k_range": [(-5.0, 5.0)],
    "v_range": [(-5.0, 5.0)],
}

# fmt: off
TestCases = {
    # g=8, s1Size=118: 每块恰一整G(step=118, <128), 原128块会跨G
    "UNM_01": {**BASE,
               "N2": [1], "S1": [118], "S2": [512]},
    # g=2, s1Size=200: 混合步进(128 + 72), 尾块72>16 走正常对半
    "UNM_02": {**BASE,
               "N1": [4], "N2": [2], "S1": [200], "S2": [512]},
    # g=4, s1Size=90: 每G一步(90), align32=96, AIV0=48/AIV1=42
    "UNM_03": {**BASE,
               "N1": [8], "N2": [2], "S1": [90], "S2": [512]},
    # g=1, s1Size=100: 单G, 等价上游尾块行为(回归)
    "UNM_04": {**BASE,
               "N1": [8], "N2": [8], "S1": [100], "S2": [512]},
    # g=2, s1Size=8: 极小G, 块=8<=16 触发上游小块特例(AIV1空转)
    "UNM_05": {**BASE,
               "N1": [4], "N2": [2], "S1": [8], "S2": [512]},
    # g=2, s1Size=256: G边界恰在128处, step恒128(无变化, 纯回归)
    "UNM_06": {**BASE,
               "N1": [4], "N2": [2], "S1": [256], "S2": [512]},
    # TND(SG布局): 不做G对齐, 行为与上游一致(回归对照)
    "UNM_07": {**BASE,
               "B": [4], "N1": [8], "N2": [4],
               "S1": [None], "S2": [None],
               "cu_seqlens_q": [[0, 90, 218, 346, 512]],
               "cu_seqlens_kv": [[0, 300, 500, 700, 900]],
               "seqused_q": [[90, 128, 128, 166]],
               "seqused_kv": [[300, 200, 200, 200]],
               "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"]},
    # D=64(config 0)
    "UNM_08": {**BASE,
               "D": [64], "DV": [64], "N2": [1], "S1": [118], "S2": [512]},
}
# fmt: on
