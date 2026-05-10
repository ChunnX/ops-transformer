# Stage6: Execute 读端 DoubleBuffer 优化 — 开发 prompt

## 背景

当前 QAR/QRS 算子的最优配置在分支 `stage6_read_db`（commit 551d53f2）。
配置：ALIGN=1024, MIN=2048, TARGET_ITER=3, X_BLOCK_ALIGN_NUM=1024。

CopyDataToWin（写端）已有 DoubleBuffer（TQueBind VECIN→VECOUT），但 Execute（读端）没有。
ReadDataBlockReduceSum 内 Win→UB 数据搬运与向量计算完全串行，MTE 和 Vector 单元互斥。

## 目标

将 Execute 阶段的 `ReadDataBlockReduceSum` 拆分为异步预取 + 同步计算，
利用 TQueBind VECIN→VECOUT 实现相邻 rank 的 Win→UB 搬运与向量计算重叠。

## 详细设计

完整设计方案见：`mc2/quant_all_reduce/docs/perf/stage6/DESIGN.md`

核心改动：
1. `xInQueue_` 从 `TQue<VECIN, 1>` 改为 `TQueBind<VECIN, VECOUT, 1>`
2. 拆分 ReadDataBlockReduceSum 为 PreFetchBlock + WaitAndCompute
3. ExecuteAllReduce/ExecuteReduceScatter 的 rank 循环改为预取流水模式

## 涉及文件

- `mc2/quant_all_reduce/op_kernel/quant_all_reduce_mte_one_shot.h` — QAR 改动
- `mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h` — QRS 改动
- host tiling / mte_comm.h / vec_comp.h 均不改动

## 关键参考文件（优先阅读）

1. 设计文档：`mc2/quant_all_reduce/docs/perf/stage6/DESIGN.md`
2. Stage5 证伪完整记录：`mc2/quant_all_reduce/docs/perf/MC2_QUANT_STAGES_HANDOFF.md`（§7.6）
3. QAR kernel（当前实现）：`mc2/quant_all_reduce/op_kernel/quant_all_reduce_mte_one_shot.h`
4. QRS kernel（当前实现）：`mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h`
5. MTE 公共通信层：`mc2/quant_reduce_scatter/op_kernel/mte_comm.h`
6. 向量计算层：`mc2/quant_reduce_scatter/op_kernel/vec_comp.h`

## 开发流程

### Step 1: 阅读代码
- 先读 DESIGN.md 理解方案
- 再读 stage5 handoff 文档了解踩过的坑
- 最后读 4 个 kernel 文件理解当前实现

### Step 2: 实现
- 先改 QAR（quant_all_reduce_mte_one_shot.h），QRS 同理跟进
- 改 Queue 类型声明 + Init 中的 InitBuffer
- 新增 PreFetchBlock / WaitAndCompute 两个函数
- 重构 ExecuteAllReduce 的 rank 循环
- 不改 host tiling，不改 mte_comm.h，不改 vec_comp.h

### Step 3: 编译验证
```bash
source ~/env.sh
cd ~/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=quant_all_reduce,quant_reduce_scatter
cd build_out
bash cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=$HOME/SUB
source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash
```

### Step 4: 精度验证
```bash
cd ~/mc2_test_0416
# 先跑精度（one_hccl_npu）
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D
```

### Step 5: 性能测试（精度通过后）
```bash
# 跑前必查 NPU
npu-smi info | tail -12
# 有占用先 sudo kill，确认 4 卡全空闲再跑

# QRS 24 用例
rm -f aclnnQuantReduceScatter_performance_result.csv aclnnQuantReducescatter_result.csv \
      mc2_aclnnQuantReduceScatter_pro_result.csv
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl --world_size 4 --soc 910D

# QAR 24 用例
rm -f aclnnQuantAllReduce_performance_result.csv aclnnQuantAllreduce_result.csv \
      mc2_aclnnQuantAllReduce_pro_result.csv
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl --world_size 4 --soc 910D
```

### Step 6: 对比基线
基线文件：
- QRS: `mc2/quant_reduce_scatter/docs/perf/stage3_perf.csv`（total: 241.98 us）
- QAR: `mc2/quant_all_reduce/docs/perf/stage3_perf.csv`（total: 661.94 us）

对比脚本模板见 stage5 实验日志。

## 已知陷阱

1. **NPU 共享冲突**：多人共用服务器，跑前必查 `npu-smi info`，错误码 561000/EADDRINUSE 多为冲突
2. **mc2_test csv 是 append 模式**：每次跑前必须 `rm` 清空
3. **profiler 状态残留**：批跑偶有单点 mc2_prof 异常飙升（ID0013/ID0014 等），需单独重测确认
4. **Stage5 已证伪的方案**：不要降 ALIGN（1024→256），不要改 MIN（2048），不要换公式。只改读端 DB。
5. **QRS 和 QAR 的 xPerBlock 不同**：QRS=(DSV3 2048, qwen 2048)，QAR=(DSV3 4096, qwen 2048)
6. **scale 数据量极小**：scaleInQue 改为 TQueBind 收益几乎为零，可考虑保持 TQue 不变，降低复杂度
7. **TQueBind 替代 TQue 时 InitBuffer 参数不变**：`tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xPerBlock_ * sizeof(XType))` 对 TQueBind 同样有效
8. **每次跑测试前 source 环境**：`source ~/env.sh && source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash`

## 环境信息

- 硬件：4×Ascend950PR (aivNum=56, AIV:AIC=2:1)
- CANN：9.0.0
- world_size：4
- 测试脚本：`~/mc2_test_0416/mc2_test.py`
- 编译：`~/ops-transformer/build.sh --pkg --soc=ascend950 --ops=...`

## 验收标准

- 精度：QRS 24/24 PASS + QAR 24/24 PASS
- 性能：QAR 总耗时 vs stage3 基线改善 ≥5%（低于此值需 profiling 分析瓶颈是否在别处）
- 代码：仅改动 kernel 文件，不涉及 host tiling / mte_comm.h / vec_comp.h
