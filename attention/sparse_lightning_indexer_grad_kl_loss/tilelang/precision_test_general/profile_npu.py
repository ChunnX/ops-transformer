#!/usr/bin/env python3
"""
torch_npu.profiler 性能分析脚本 - SparseLightningIndexerGradKLLoss TileLang

用法:
  python profile_npu.py                           # 默认 Case 1
  PROFILE_CASE=2 python profile_npu.py            # 指定 Case
  PROFILE_CASE=1 PROFILE_PATH=./my_prof python profile_npu.py

可选环境变量:
  PROFILE_CASE: 测试用例编号 (0-4), 默认 1
  PROFILE_PATH: profiling 输出目录, 默认 ./profiling_output
"""

import torch
import torch_npu
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss
from precision_test_general import gen_inputs_corrected, TEST_CASES

CASE_IDX = int(os.environ.get("PROFILE_CASE", "1"))
PROF_PATH = os.environ.get("PROFILE_PATH", "./profiling_output")

LOOP_TIME = 55
REPEAT_TIME = 1
WARM_TIME = 5


def _call_op(inputs, case):
    return sparse_lightning_indexer_grad_kl_loss(
        inputs["query"], inputs["key"],
        inputs["query_index"], inputs["key_index"],
        inputs["weights"], inputs["sparse_indices"],
        inputs["softmax_max"], inputs["softmax_sum"],
        inputs["scale_value"],
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
        layout_query=case["layout"],
        actual_seq_lengths_query=[case["S1"]] * case["B"] if case["layout"] == "BSND" else [case["S1"]],
        actual_seq_lengths_key=[case["S2"]] * case["B"] if case["layout"] == "BSND" else [case["S2"]],
        sparse_mode=3,
    )


def get_profiling(loop, repeat, warmup, path):
    active = loop // repeat - warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
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


def get_performance(profiling_path):
    columns = ["rank_id", "avg_time_us", "min_time_us", "op_type"]
    time_consumption = pd.DataFrame(columns=columns)

    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                local_rank_id = op_summary["Device_id"].iloc[0]

                for op_type in op_summary["OP Type"].unique():
                    op_info = op_summary[op_summary["OP Type"] == op_type]
                    op_execute_time = op_info["Task Duration(us)"].tail(50)
                    avg_time = op_execute_time.mean()
                    min_time = op_execute_time.min()
                    time_consumption.loc[time_consumption.shape[0]] = [
                        local_rank_id, avg_time, min_time, op_type,
                    ]

    time_consumption["rank_id"] = time_consumption["rank_id"].astype("int32")
    time_consumption.sort_values(by="avg_time_us", ascending=False, inplace=True)
    time_consumption.reset_index(drop=True, inplace=True)
    return time_consumption


def main():
    case = TEST_CASES[CASE_IDX]
    print(f"Profiling Case {CASE_IDX}: {case['name']}")
    print(f"配置: B={case['B']}, S1={case['S1']}, S2={case['S2']}, K={case['K']}, "
          f"N1={case['N1']}, Nidx1={case['Nidx1']}, layout={case['layout']}")
    print(f"Profiling 输出: {PROF_PATH}")

    inputs = gen_inputs_corrected(case)

    tensor = torch.ones(10000, 10000).to(torch.float16).npu()

    with get_profiling(LOOP_TIME, REPEAT_TIME, WARM_TIME, PROF_PATH) as prof:
        for i in range(LOOP_TIME):
            _call_op(inputs, case)

            if i == 0:
                for _ in range(100):
                    torch.exp(tensor)
            prof.step()

    torch.npu.synchronize()

    print("\n解析 profiling 数据...")
    perf = get_performance(PROF_PATH)
    print("\n算子耗时 Top 20 (按平均耗时降序):")
    print(perf.head(20).to_string(index=False))

    total_row = perf[perf["op_type"] == "total"]
    if not total_row.empty:
        print(f"\n总耗时: avg={total_row.iloc[0]['avg_time_us']:.1f}us, "
              f"min={total_row.iloc[0]['min_time_us']:.1f}us")

    print(f"\n完整数据已保存到: {PROF_PATH}")
    print("可用 MindStudio Insight 导入查看流水详情")


if __name__ == "__main__":
    main()