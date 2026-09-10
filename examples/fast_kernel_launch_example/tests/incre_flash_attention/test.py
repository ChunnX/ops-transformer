# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import csv
import logging
import os
from pathlib import Path

import torch
import torch_npu
import torch.nn as nn
import ascend_ops

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

HEAD_TIMING_MAGIC = 0x1FA20268
HEAD_TIMING_ENTRY_COMMIT = 0xE17E2026
HEAD_TIMING_COMMIT = 0xC0DE2026
HEAD_TIMING_SLOT_COUNT = 8
CORE_TYPE_VECTOR = 1
MAX_CUBE_CORES_950 = 32
MAX_VECTOR_CORES_950 = 64
EXPECTED_CUBE_CORES = 0
EXPECTED_VECTOR_CORES = 0
HEAD_TIMING_LAUNCHES = int(os.environ.get("HEAD_TIMING_LAUNCHES", "101"))
HEAD_TIMING_SHOTS = int(os.environ.get("HEAD_TIMING_SHOTS", "5"))
HEAD_TIMING_OUTPUT = Path(
    os.environ.get("HEAD_TIMING_OUTPUT", "ifa_head_timing_raw.csv")
)
NPU_DEVICE = int(os.environ.get("NPU_DEVICE", "0"))
HEAD_TIMING_POINT = os.environ.get("HEAD_TIMING_POINT", "mte2")
if HEAD_TIMING_POINT not in {"entry", "mte2"}:
    raise ValueError("HEAD_TIMING_POINT must be 'entry' or 'mte2'")

# =============================================
# 🔧 CONFIGURATION CENTER (可配置化)
# =============================================


class AttentionConfig:
    def __init__(self):
        # Attention Dimensions
        self.batch_size = EXPECTED_CUBE_CORES
        self.q_head_num = 64
        self.kv_head_num = 1
        self.q_seq = 1
        self.block_size = 128
        self.head_dim = 128

        # KV Cache Dimensions
        self.kv_seq_length = 8192
        self.max_block_num_per_batch = (self.kv_seq_length // self.block_size) + 1
        self.block_num = self.batch_size * self.max_block_num_per_batch

        # Attention Scale
        self.softmax_scale = 1.0 / (self.head_dim**0.5)


# =============================================
# 🧠 MODEL DEFINITION (模块化：模型清晰分离)
# =============================================


class FusedAttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, param: dict):
        # Step 1: Compute metadata
        # metadata = torch.ops.ascend_ops.npu_fused_infer_attention_score_metadata(
        #     **param["metaParam"]
        # )
        # param["faParam"]["metadata"] = metadata

        # Step 2: Call fused attention
        return torch.ops.ascend_ops.npu_fused_infer_attention_score(**param["faParam"])


# =============================================
# 📥 INPUT GENERATION (输入构造独立函数)
# =============================================


def generate_inputs(config: AttentionConfig):
    # Query
    q = (
        torch.randn(config.batch_size, config.q_head_num, config.q_seq, config.head_dim)
        .to(dtype=torch.bfloat16)
        .npu()
    )

    # KV Cache (INT8 quantized)
    key_cache = (
        torch.randint(
            0,
            100,
            (
                config.block_num,
                config.kv_head_num,
                config.head_dim // 32,
                config.block_size,
                32,
            ),
        )
        .to(dtype=torch.int8)
        .npu()
    )

    value_cache = (
        torch.randint(
            0,
            100,
            (
                config.block_num,
                config.kv_head_num,
                config.head_dim // 32,
                config.block_size,
                32,
            ),
        )
        .to(dtype=torch.int8)
        .npu()
    )

    # Block Table
    block_table = (
        torch.arange(
            config.batch_size * config.max_block_num_per_batch, dtype=torch.int32
        )
        .view(config.batch_size, config.max_block_num_per_batch)
        .npu()
    )

    # Sequence lengths
    actual_seq_kvlen = torch.tensor(
        [config.kv_seq_length] * config.batch_size, dtype=torch.int64
    ).npu()

    # Dequantization scales
    dequant_scale_key = (
        torch.randn(config.kv_head_num, 1, config.head_dim)
        .to(dtype=torch.bfloat16)
        .npu()
    )

    dequant_scale_value = (
        torch.randn(config.kv_head_num, 1, config.head_dim)
        .to(dtype=torch.bfloat16)
        .npu()
    )

    # FA Param
    fa_param = {
        "query": q,
        "key": key_cache,
        "value": value_cache,
        "actual_seq_kvlen": actual_seq_kvlen,
        "block_table": block_table,
        "dequant_scale_key": dequant_scale_key,
        "dequant_scale_value": dequant_scale_value,
        "num_query_heads": config.q_head_num,
        "num_key_value_heads": config.kv_head_num,
        "softmax_scale": config.softmax_scale,
        "block_size": config.block_size,
        "input_layout": "BNSD",
        "sparse_mode": 0,
        "inner_precise": 1,
        "key_quant_mode": 0,
        "value_quant_mode": 0,
    }
    # Meta Param
    meta_param = {
        "batch_size": config.batch_size,
        "query_seq_size": config.q_seq,
        "query_head_num": config.q_head_num,
        "key_head_num": config.kv_head_num,
        "head_dim": config.head_dim,
        "block_size": config.block_size,
        "max_block_num_per_batch": config.max_block_num_per_batch,
        "actual_seq_lengths_kv": actual_seq_kvlen.to(dtype=torch.int64),
        "layout_query": "BNSD",
    }

    return {"metaParam": meta_param, "faParam": fa_param}


def read_cycle_shot(phase: str, shot: int):
    """回读持久 cycle tensor (每拍 replay 后立即读, 记录 = 本拍最后一次 launch 的 48 核打点)"""
    cycle = torch.ops.ascend_ops.npu_fused_ifa_cycle_tensor()
    raw = cycle.detach().cpu().contiguous().flatten()
    record_count = raw.numel() // HEAD_TIMING_SLOT_COUNT
    records = raw[: record_count * HEAD_TIMING_SLOT_COUNT].view(
        record_count, HEAD_TIMING_SLOT_COUNT
    )
    valid = (
        (records[:, 0] == HEAD_TIMING_MAGIC)
        & (records[:, 1] >= 0)
        & (records[:, 1] < EXPECTED_VECTOR_CORES)
        & (records[:, 2] == CORE_TYPE_VECTOR)
        & (records[:, 3] > 0)
        & (records[:, 5] == HEAD_TIMING_COMMIT)
    )
    invalid_count = int((~valid).sum().item())
    records = records[valid]
    if records.numel() == 0:
        raise RuntimeError("No IFA head timing records were written by the kernel")
    records = records[records[:, 1].argsort()]
    if records.shape[0] != EXPECTED_VECTOR_CORES:
        raise RuntimeError(
            f"Expected {EXPECTED_VECTOR_CORES} Vector timing records, "
            f"got {records.shape[0]} (invalid={invalid_count})"
        )
    result = []
    for record in records.tolist():
        result.append(
            {
                "measurement_point": HEAD_TIMING_POINT,
                "phase": phase,
                "launch_index": shot,
                "vector_block": record[1],
                "ai_core": record[6],
                "kernel_entry_cycle": record[3],  # T0: kernel 第一行
                "before_first_mte2_cycle": record[4],  # T1: 首个 MTE2 前
                "kernel_end_cycle": record[7] if record[7] > 0 else "",  # T2: kernel 末
            }
        )
    return result


def write_head_timing(rows):
    HEAD_TIMING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with HEAD_TIMING_OUTPUT.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Raw head timing written to %s", HEAD_TIMING_OUTPUT.resolve())


# =============================================
# 🏁 MAIN EXECUTION (主入口函数)
# =============================================


def main():
    global EXPECTED_CUBE_CORES, EXPECTED_VECTOR_CORES

    # 1. Select the device and use all cores available on this 950 SKU.
    logger.info("🔄 Resetting TorchDynamo...")
    torch._dynamo.reset()
    torch.npu.set_device(NPU_DEVICE)
    properties = torch.npu.get_device_properties(NPU_DEVICE)
    EXPECTED_CUBE_CORES = int(
        os.environ.get("HEAD_TIMING_CUBE_CORES", properties.cube_core_num)
    )
    EXPECTED_VECTOR_CORES = int(
        os.environ.get("HEAD_TIMING_VECTOR_CORES", properties.vector_core_num)
    )
    if not 0 < EXPECTED_CUBE_CORES <= MAX_CUBE_CORES_950:
        raise ValueError(
            f"950 Cube core count must be in [1, {MAX_CUBE_CORES_950}], "
            f"got {EXPECTED_CUBE_CORES}"
        )
    if not 0 < EXPECTED_VECTOR_CORES <= MAX_VECTOR_CORES_950:
        raise ValueError(
            f"950 Vector core count must be in [1, {MAX_VECTOR_CORES_950}], "
            f"got {EXPECTED_VECTOR_CORES}"
        )
    if EXPECTED_VECTOR_CORES != 2 * EXPECTED_CUBE_CORES:
        raise ValueError(
            "This IFA mix-kernel test requires a 1:2 Cube/Vector core ratio, "
            f"got {EXPECTED_CUBE_CORES}/{EXPECTED_VECTOR_CORES}"
        )
    logger.info(
        "Using %d Cube cores and %d Vector cores on %s",
        EXPECTED_CUBE_CORES,
        EXPECTED_VECTOR_CORES,
        properties.name,
    )

    # 2. Load config
    config = AttentionConfig()

    # 3. Build model
    logger.info("🧠 Building FusedAttentionNetwork...")
    model = FusedAttentionNetwork().npu()

    # 4. Generate inputs
    logger.info("📥 Generating input tensors...")
    param = generate_inputs(config)

    # 5. Run inference
    logger.info("🚀 Running inference...")
    try:
        # warmup (capture 外): 触发 op 首次调用链 (含持久 cycle tensor 的 at::zeros
        # 分配) —— capture 内的动态分配/memset 会导致 replay 死锁, 对齐 add/gmm 做法
        logger.info("Warming up once outside graph capture...")
        model(param)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        timing_outputs = []
        logger.info("Capturing %d IFA nodes...", HEAD_TIMING_LAUNCHES)
        with torch.npu.graph(graph):
            for _ in range(HEAD_TIMING_LAUNCHES):
                output, _ = model(param)
                timing_outputs.append(output)
        del timing_outputs

        logger.info("Synchronizing graph capture...")
        torch.npu.synchronize()

        # 对齐 add/gmm 多拍模式: 每拍独立完整 replay, 拍后立即回读 cycle tensor
        timing_rows = []
        for shot in range(HEAD_TIMING_SHOTS):
            logger.info("Replaying shot %d/%d...", shot + 1, HEAD_TIMING_SHOTS)
            graph.replay()
            torch.npu.synchronize()
            launch_rows = read_cycle_shot("replay", shot)
            timing_rows.extend(launch_rows)
            reached_count = sum(
                row["before_first_mte2_cycle"] != "" for row in launch_rows
            )
            logger.info(
                "Shot %d: point=%s vector_entries=%d active_vectors=%d",
                shot,
                HEAD_TIMING_POINT,
                len(launch_rows),
                reached_count,
            )
        write_head_timing(timing_rows)

    except Exception as e:
        logger.info(f"❌ Inference failed: {e}")
        raise


# =============================================
# 🚀 ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()
