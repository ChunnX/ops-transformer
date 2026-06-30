#!/usr/bin/env python3
"""
NPU Profiler 性能分析 - 0629.py TileLang 单核版本

用法:
  python profile_0629.py                           # 默认配置
  PROFILE_PATH=./prof_out python profile_0629.py   # 指定输出目录
  PROFILE_LEVEL=Level2 python profile_0629.py      # 更详细的 Level2 profiling

输出:
  - {PROFILE_PATH}/       TensorBoard trace 文件 (可用 MindStudio Insight 打开)
  - 控制台输出各算子耗时 Top-20 排名

环境变量:
  PROFILE_PATH:   profiling 输出目录, 默认 ./profiling_output_0629
  PROFILE_LEVEL:  ProfilerLevel (Level0/Level1/Level2), 默认 Level1
  LOOP_TIME:      总循环次数, 默认 55
  WARM_TIME:      warmup 次数, 默认 5
"""

import torch
import torch_npu
import math
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
mod = import_module("0629")
sparse_lightning_indexer_grad_kl_loss = mod.sparse_lightning_indexer_grad_kl_loss
mod._clear_caches()

# ==============================================================================
# 配置
# ==============================================================================
PROF_PATH = os.environ.get("PROFILE_PATH", "./profiling_output_0629")

_level_map = {
    "Level0": torch_npu.profiler.ProfilerLevel.Level0,
    "Level1": torch_npu.profiler.ProfilerLevel.Level1,
    "Level2": torch_npu.profiler.ProfilerLevel.Level2,
}
PROFILE_LEVEL = _level_map.get(
    os.environ.get("PROFILE_LEVEL", "Level1"),
    torch_npu.profiler.ProfilerLevel.Level1,
)

LOOP_TIME = int(os.environ.get("LOOP_TIME", "55"))
WARM_TIME = int(os.environ.get("WARM_TIME", "5"))
REPEAT_TIME = 1

CASE_CONFIG = {
    "B": 1, "S1": 1, "S2": 2048, "N1": 128, "N2": 1,
    "Nidx1": 64, "Nidx2": 1, "DQuery": 512, "DQueryIndex": 128,
    "d_rope": 64, "K": 2048, "layout": "BSND",
}


# ==============================================================================
# 输入生成 (与 test_0629.py 完全一致)
# ==============================================================================
def gen_inputs(seed=42):
    B = CASE_CONFIG["B"]; S1 = CASE_CONFIG["S1"]; S2 = CASE_CONFIG["S2"]
    N1 = CASE_CONFIG["N1"]; N2 = CASE_CONFIG["N2"]
    Nidx1 = CASE_CONFIG["Nidx1"]; Nidx2 = CASE_CONFIG["Nidx2"]
    DQuery = CASE_CONFIG["DQuery"]; DQueryIndex = CASE_CONFIG["DQueryIndex"]
    d_rope = CASE_CONFIG["d_rope"]; K = CASE_CONFIG["K"]
    G = N1 // N2
    scale_value = 1.0 / math.sqrt(DQuery)

    torch.manual_seed(seed)
    dtype = torch.float16

    query = torch.randn(B, S1, N1, DQuery, dtype=dtype).npu()
    key = torch.randn(B, S2, N2, DQuery, dtype=dtype).npu()
    query_index = torch.randn(B, S1, Nidx1, DQueryIndex, dtype=dtype).npu()
    key_index = torch.randn(B, S2, Nidx2, DQueryIndex, dtype=dtype).npu()
    query_rope = torch.randn(B, S1, N1, d_rope, dtype=dtype).npu()
    key_rope = torch.randn(B, S2, N2, d_rope, dtype=dtype).npu()
    weights = torch.randn(B, S1, Nidx1, dtype=dtype).npu()
    sparse_indices = torch.zeros(B, S1, Nidx2, K, dtype=torch.int32).npu()

    s2_real_len = min(S2, K)
    indices = torch.randperm(S2)[:s2_real_len]
    for s1_idx in range(S1):
        sparse_indices[0, s1_idx, 0, :s2_real_len] = indices.to(torch.int32).npu()
        if s2_real_len < K:
            sparse_indices[0, s1_idx, 0, s2_real_len:] = -1
    torch.npu.synchronize()

    with torch.no_grad():
        k_mqa = key.squeeze(2)
        k_rope_mqa = key_rope.squeeze(2)
        q_full = torch.cat([query, query_rope], dim=-1)
        k_full = torch.cat([k_mqa, k_rope_mqa], dim=-1)
        k_full_t = k_full.transpose(-1, -2).unsqueeze(1)
        sm_max_list, sm_sum_list = [], []
        for s1_idx in range(S1):
            row = torch.matmul(q_full[:, s1_idx:s1_idx+1], k_full_t)
            row_scaled = row.float() * scale_value
            row_max = row_scaled.max(dim=-1).values
            row_sum = torch.exp(row_scaled - row_max.unsqueeze(-1)).sum(dim=-1)
            sm_max_list.append(row_max)
            sm_sum_list.append(row_sum)
        sm_max_val = torch.cat(sm_max_list, dim=1)
        sm_sum_val = torch.cat(sm_sum_list, dim=1)
        softmax_max = sm_max_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()
        softmax_sum = sm_sum_val.reshape(B, S1, N2, G).permute(0, 2, 1, 3).contiguous().float()

    return {
        "query": query, "key": key,
        "query_index": query_index, "key_index": key_index,
        "weights": weights, "sparse_indices": sparse_indices,
        "softmax_max": softmax_max, "softmax_sum": softmax_sum,
        "scale_value": scale_value,
        "query_rope": query_rope, "key_rope": key_rope,
    }


def call_tl(inputs):
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        layout_query="BSND",
        actual_seq_lengths_query=[CASE_CONFIG["S1"]] * CASE_CONFIG["B"],
        actual_seq_lengths_key=[CASE_CONFIG["S2"]] * CASE_CONFIG["B"],
        sparse_mode=3,
    )


# ==============================================================================
# Profiler 配置
# ==============================================================================
def get_profiler(loop, repeat, warmup, path):
    active = loop // repeat - warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=PROFILE_LEVEL,
        l2_cache=False,
        data_simplification=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=active, repeat=repeat, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
    return prof


def parse_performance(profiling_path):
    """解析 op_summary.csv, 按算子类型统计耗时"""
    columns = ["Device", "AvgTime_us", "MinTime_us", "OpType"]
    time_consumption = pd.DataFrame(columns=columns)

    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                device_id = op_summary["Device_id"].iloc[0]

                for op_type in op_summary["OP Type"].unique():
                    op_info = op_summary[op_summary["OP Type"] == op_type]
                    op_execute_time = op_info["Task Duration(us)"].tail(50)
                    avg_time = op_execute_time.mean()
                    min_time = op_execute_time.min()
                    time_consumption.loc[time_consumption.shape[0]] = [
                        device_id, avg_time, min_time, op_type,
                    ]

    time_consumption["Device"] = time_consumption["Device"].astype("int32")
    time_consumption.sort_values(by="AvgTime_us", ascending=False, inplace=True)
    time_consumption.reset_index(drop=True, inplace=True)
    return time_consumption


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 65)
    print("NPU Profiler - 0629.py TileLang-Ascend 单核版本")
    print("=" * 65)
    print(f"配置: B={CASE_CONFIG['B']}, S1={CASE_CONFIG['S1']}, S2={CASE_CONFIG['S2']}, "
          f"K={CASE_CONFIG['K']}, N1={CASE_CONFIG['N1']}, Nidx1={CASE_CONFIG['Nidx1']}")
    print(f"Profile Level: {PROFILE_LEVEL}, Loop={LOOP_TIME}, Warmup={WARM_TIME}")
    print(f"输出目录: {PROF_PATH}")

    inputs = gen_inputs()

    # 额外预热: 让 JIT 编译完成, 排除编译时间干扰
    print("\n[0] JIT 预热 (3 次)...")
    for _ in range(3):
        call_tl(inputs)
        torch.npu.synchronize()
    print("    预热完成")

    # 填充 GPU 占用, 避免 profiler 采到空闲段
    filler = torch.ones(10000, 10000, dtype=torch.float16).npu()

    print(f"\n[1] Profiling (共 {LOOP_TIME} 轮)...")
    with get_profiler(LOOP_TIME, REPEAT_TIME, WARM_TIME, PROF_PATH) as prof:
        for i in range(LOOP_TIME):
            call_tl(inputs)

            if i == 0:
                # 首轮后做额外计算填充, 拉开 trace 间隔
                for _ in range(100):
                    torch.exp(filler)
            prof.step()

    torch.npu.synchronize()
    print("    Profiling 完成")

    # 解析
    print("\n[2] 解析 profiling 数据...")
    perf = parse_performance(PROF_PATH)
    if perf.empty:
        print("    警告: 未找到 op_summary.csv, 请检查输出目录")
    else:
        print(f"\n算子耗时 Top 20 (按平均耗时降序):")
        print(perf.head(20).to_string(index=False))

        total_row = perf[perf["OpType"] == "total"]
        if not total_row.empty:
            print(f"\n总耗时: avg={total_row.iloc[0]['AvgTime_us']:.1f}us, "
                  f"min={total_row.iloc[0]['MinTime_us']:.1f}us")

    print(f"\n[3] Trace 文件已保存到: {PROF_PATH}")
    print("    用 MindStudio Insight → Import 打开查看流水时序图")


if __name__ == "__main__":
    main()
