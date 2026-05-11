# 新对话 Prompt

## 算子位置

两个 MC2 量化通信算子，路径均在 `~/ops-transformer` 仓库 `perf_write_status_leader_v2` 分支：

```
mc2/quant_all_reduce/          — QuantAllReduce (QAR)
  op_kernel/quant_all_reduce_mte_one_shot.h
  op_kernel/quant_all_reduce_tiling_data.h
  op_host/op_tiling/quant_all_reduce_tiling.cpp

mc2/quant_reduce_scatter/      — QuantReduceScatter (QRS)
  op_kernel/quant_reduce_scatter_mte.h
  op_kernel/quant_reduce_scatter_tiling_data.h
  op_host/op_tiling/quant_reduce_scatter_tiling.cpp
  op_kernel/mte_comm.h          — 公共 MTE 通信层（两算子共用）
  op_kernel/vec_comp.h          — 公共 Vector 计算层（两算子共用）
  op_kernel/utils.h
```

## 修改后编译验证

编译测试流程见 `mc2/quant_all_reduce/BUILD_AND_TEST.md`，核心步骤：

```bash
# 编译
source ~/env.sh
cd ~/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=quant_all_reduce,quant_reduce_scatter

# 安装
cd build_out
bash cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=$HOME/SUB
source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash

# 跑前查卡
npu-smi info | tail -12   # 确认 4 卡全空闲

# 精度（one_hccl_npu）
cd ~/mc2_test_0416
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D

# 性能（one_hccl，跑前清 csv）
rm -f aclnnQuantReduceScatter_performance_result.csv aclnnQuantReducescatter_result.csv mc2_aclnnQuantReduceScatter_pro_result.csv
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 --num 1 24 --scene one_hccl --world_size 4 --soc 910D

rm -f aclnnQuantAllReduce_performance_result.csv aclnnQuantAllreduce_result.csv mc2_aclnnQuantAllReduce_pro_result.csv
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 --num 1 24 --scene one_hccl --world_size 4 --soc 910D
```

## 环境

- 硬件：4×Ascend950PR (aivNum=56, world_size=4)
- CANN：9.0.0
- 测试脚本：`~/mc2_test_0416/mc2_test.py`
- 性能基线：QRS ~242us, QAR ~662us (24 用例合计)
- 精度基线：192/192 PASS

## 关键设计要点

- QAR 是 allgather 型（compute-friendly），QRS 是 all2all 型（comm-bound）
- xPerBlock 由 host tiling 的 TARGET_ITER=3 + MIN_BLOCK=2048 公式计算，kernel 端以 host 推荐值为上限
- 对齐粒度 `alignBlock` 在 tiling data 中设置（当前 1024），kernel 从 tiling data 读取
- `ALIGN_BLOCK` 在 tiling cpp 中定义，`alignBlock` 通过 tiling data 传递给 kernel
- mte_comm.h 是两算子共用的 MTE 通信层，修改需同时验证两个算子
- NPU 是多人共享，跑前必查 `npu-smi info`，有占用 sudo kill
- mc2_test csv 是 append 模式，跑性能前必清
