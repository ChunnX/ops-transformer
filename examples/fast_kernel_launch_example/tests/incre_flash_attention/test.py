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
EXPECTED_VECTOR_CORES = 48
HEAD_TIMING_LAUNCHES = int(os.environ.get("HEAD_TIMING_LAUNCHES", "101"))
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
        self.batch_size = 24
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


def read_head_timing(timing_output: torch.Tensor, phase: str, launch_index: int):
    raw = timing_output.detach().cpu().contiguous().view(torch.int64).flatten()
    record_count = raw.numel() // HEAD_TIMING_SLOT_COUNT
    records = raw[: record_count * HEAD_TIMING_SLOT_COUNT].view(
        record_count, HEAD_TIMING_SLOT_COUNT
    )
    records = records[records[:, 0] == HEAD_TIMING_MAGIC]
    if records.numel() == 0:
        raise RuntimeError("No IFA head timing records were written by the kernel")

    entry_valid = (
        (records[:, 1] >= 0)
        & (records[:, 1] < EXPECTED_VECTOR_CORES)
        & (records[:, 2] == CORE_TYPE_VECTOR)
        & (records[:, 3] > 0)
        & (
            (records[:, 5] == HEAD_TIMING_ENTRY_COMMIT)
            | (records[:, 5] == HEAD_TIMING_COMMIT)
        )
    )
    invalid_count = int((~entry_valid).sum().item())
    records = records[entry_valid]
    if records.numel() == 0:
        raise RuntimeError("All IFA head timing records were incomplete")
    if invalid_count:
        logger.warning("Discarded %d incomplete timing records", invalid_count)

    records = records[records[:, 1].argsort()]
    if records.shape[0] != EXPECTED_VECTOR_CORES:
        raise RuntimeError(
            f"Expected {EXPECTED_VECTOR_CORES} Vector entry records, "
            f"got {records.shape[0]}"
        )

    result = []
    for record in records.tolist():
        reached_first_mte2 = (
            record[4] > 0 and record[5] == HEAD_TIMING_COMMIT
        )
        result.append(
            {
                "measurement_point": HEAD_TIMING_POINT,
                "phase": phase,
                "launch_index": launch_index,
                "vector_block": record[1],
                "ai_core": record[6],
                "kernel_entry_cycle": record[3],
                "before_first_mte2_cycle": record[4] if reached_first_mte2 else "",
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
    # 1. Load config
    config = AttentionConfig()

    # 2. Reset Dynamo (for clean compile)
    logger.info("🔄 Resetting TorchDynamo...")
    torch._dynamo.reset()
    torch.npu.set_device(NPU_DEVICE)

    # 3. Build model
    logger.info("🧠 Building FusedAttentionNetwork...")
    model = FusedAttentionNetwork().npu()

    # 4. Generate inputs
    logger.info("📥 Generating input tensors...")
    param = generate_inputs(config)

    # 5. Run inference
    logger.info("🚀 Running inference...")
    try:
        graph = torch.npu.NPUGraph()
        timing_outputs = []
        logger.info("Capturing %d IFA nodes...", HEAD_TIMING_LAUNCHES)
        with torch.npu.graph(graph):
            for _ in range(HEAD_TIMING_LAUNCHES):
                output, _ = model(param)
                timing_outputs.append(output)

        logger.info("Synchronizing graph capture...")
        torch.npu.synchronize()
        logger.info("Replaying the graph once...")
        graph.replay()
        torch.npu.synchronize()

        timing_rows = []
        for launch_index, timing_output in enumerate(timing_outputs):
            launch_rows = read_head_timing(timing_output, "replay", launch_index)
            timing_rows.extend(launch_rows)
            reached_count = sum(
                row["before_first_mte2_cycle"] != "" for row in launch_rows
            )
            expected_reached_count = (
                0 if HEAD_TIMING_POINT == "entry" else EXPECTED_VECTOR_CORES
            )
            if reached_count != expected_reached_count:
                raise RuntimeError(
                    f"{HEAD_TIMING_POINT} mode expected "
                    f"{expected_reached_count} T2 records, got {reached_count}"
                )
            logger.info(
                "Launch %d: point=%s vector_entries=%d active_vectors=%d",
                launch_index,
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
