SCALE_DEFAULT = 0.0883883476483184

PARAM_NAMES = [
    "B", "N1", "N2", "S1", "S2", "D", "DV",
    "layout_q", "layout_kv", "layout_out",
    "dtype", "mask_mode", "return_softmax_lse",
]

DEFAULTS = {
    "DV": [128],
    "N2": [1],
    "S2": [1024],
    "layout_kv": ["BNSD"],
    "layout_out": ["BNSD"],
    "mask_mode": [0],
    "return_softmax_lse": [False],
    "seed": [42],
    "graph_path": [0],
    "device_id": [0],
    "block_size": [-1],
    "scale": [0.0],
    "win_left": [-1],
    "win_right": [-1],
    "cu_seqlens_q": [None],
    "cu_seqlens_kv": [None],
    "seqused_q": [None],
    "seqused_kv": [None],
    "block_table_shape": [None],
    "q_range": [(-5.0, 5.0)],
    "k_range": [(-5.0, 5.0)],
    "v_range": [(-5.0, 5.0)],
    "enable_learnable_sink": [False],
}

CI_BASE = {
    "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
    "D": [128], "DV": [128],
    "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
    "dtype": ["fp16"],
}

CASE_SETS = {
    "ci": {
        "FP16_BNSD_01": {
            **CI_BASE,
        },
        "BF16_BNSD_02": {
            **CI_BASE, "dtype": ["bf16"],
        },
        "FP16_BSND_03": {
            "B": [1], "N1": [1], "N2": [1], "S1": [64], "S2": [256],
            "D": [128], "DV": [128],
            "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
            "dtype": ["fp16"],
        },
        "BF16_BSND_04": {
            "B": [1], "N1": [1], "N2": [1], "S1": [64], "S2": [256],
            "D": [128], "DV": [128],
            "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
            "dtype": ["bf16"],
        },
        "FP16_TND_05": {
            "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
            "D": [128], "DV": [128],
            "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"],
            "dtype": ["fp16"],
            "cu_seqlens_q": [0, 64, 128, 192, 256, 320, 384, 448, 512],
            "cu_seqlens_kv": [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
        },
        "FP16_TND_seqused_06": {
            "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
            "D": [128], "DV": [128],
            "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"],
            "dtype": ["fp16"],
            "cu_seqlens_q": [0, 64, 128, 192, 256, 320, 384, 448, 512],
            "cu_seqlens_kv": [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
            "seqused_q": [60] * 8,
            "seqused_kv": [1000] * 8,
        },
        "FP16_PA_BBND_07": {
            "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
            "dtype": ["fp16"],
            "block_size": 128,
        },
        "FP16_PA_BNBD_08": {
            "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
            "dtype": ["fp16"],
            "block_size": 128,
        },
        "FP16_mask3_09": {
            "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
            "dtype": ["fp16"], "mask_mode": [3],
        },
        "FP16_mask3_S1gtS2_10": {
            "B": [2], "N1": [8], "N2": [1], "S1": [1024], "S2": [64],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
            "dtype": ["fp16"], "mask_mode": [3],
        },
        "FP16_LSE_11": {
            "B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
            "dtype": ["fp16"], "return_softmax_lse": [True],
        },
        "BF16_GQA_12": {
            "B": [1], "N1": [4], "N2": [1], "S1": [128], "S2": [512],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
            "dtype": ["bf16"], "mask_mode": [3],
        },
    },

    "debug": {
        "FP16_BNSD_min": {
            "B": [1], "N1": [1], "N2": [1], "S1": [64], "S2": [64],
            "D": [128], "DV": [128],
            "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
            "dtype": ["fp16"],
        },
        "BF16_BSND_min": {
            "B": [1], "N1": [1], "N2": [1], "S1": [64], "S2": [64],
            "D": [128], "DV": [128],
            "layout_q": ["BSND"], "layout_kv": ["BSND"], "layout_out": ["BSND"],
            "dtype": ["bf16"],
        },
    },
}

SKIP_CASES = {
    "ci": set(),
    "debug": set(),
}
