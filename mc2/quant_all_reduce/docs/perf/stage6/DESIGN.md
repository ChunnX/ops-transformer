# Stage6: Execute 读端 DoubleBuffer 优化设计

**日期**：2026-05-10
**分支**：stage6_read_db
**基线 commit**：551d53f2（stage4 最优配置）
**目标算子**：QAR + QRS（共用 mte_comm.h / vec_comp.h，同步修改）

---

## 1. 问题描述

### 1.1 当前流水线

```
Process()
  ├─ CopyDataToWin       ← DB已激活（TQueBind VECIN→VECOUT, BUFFER_NUM=2）
  ├─ WriteStatusToWin
  ├─ ReadStatus
  └─ Execute             ← 无DB，串行
       └─ for curBlock:
            ClearSumTensor
            └─ for rank in 0..3:           ← 4次串行
                 ReadDataBlockReduceSum     ← [读x → 等 → 读scale → 等] → 算
            CopyResultToOutput
```

### 1.2 瓶颈

`ReadDataBlockReduceSum`（`quant_all_reduce_mte_one_shot.h:204` / `quant_reduce_scatter_mte.h:215`）完全串行：

```cpp
// 步骤1：读 x，阻塞等待
DataCopy(xTmpTensor, remoteWinXTensor_[...], xNum);
xInQueue_.EnQue(xTmpTensor);
xTmpTensor = xInQueue_.DeQue<XType>();       // 等 MTE 完成 ← 向量单元空闲

// 步骤2：读 scale，阻塞等待
DataCopyPad(scaleTmpTensor, ...);
scaleInQue.EnQue(scaleTmpTensor);
scaleTmpTensor = scaleInQue.DeQue<ScalesType>(); // 等 MTE 完成 ← 向量单元空闲

// 步骤3：计算
vecComp_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_);
// ← MTE 搬运单元空闲
```

在 rank 循环中（4 次 × 3~4 blocks = 12~16 次调用），**MTE 搬运和向量计算始终互斥**：读数据时向量单元空闲，计算时 MTE 搬运单元空闲。

### 1.3 影响

QAR qwen（xPerBlock=2048, iter=4, rankDim=4）：Execute 阶段串行执行 16 次 Win→UB 搬运 × 每次约 4KB。MTE 搬运延迟 + 向量计算延迟完全叠加。

---

## 2. 优化方案

### 2.1 核心思路

**拆分 `ReadDataBlockReduceSum` 为异步读 + 同步算，重叠相邻 rank 的 Win→UB 搬运和向量计算。**

```
当前（rank 0,1,2,3 串行）：
  read(0) → compute(0) → read(1) → compute(1) → read(2) → compute(2) → read(3) → compute(3)

优化后（相邻 rank 读算重叠）：
  read(0) → [compute(0) || read(1)] → [compute(1) || read(2)] → [compute(2) || read(3)] → compute(3)
```

### 2.2 实现方式

将 xInQueue_ 和 scaleInQue 从 `TQue<VECIN, 1>` 改为 `TQueBind<VECIN, VECOUT, 1>`，利用 VECIN/VECOUT 分离实现异步流水：

- **VECIN 端**：发起 DataCopy（异步，不阻塞）
- **VECOUT 端**：DeQue 等待数据到达后开始计算

函数拆分：

```cpp
// 新函数1：异步发起数据搬运（不等待完成）
void PreFetchBlock(uint64_t curXOffset, uint64_t curScaleOffset, uint32_t xNum, uint32_t scaleNum)
{
    // x: Win → UB（VECIN，异步）
    LocalTensor<XType> xTmp = xInQueue_.AllocTensor<XType>();
    DataCopy(xTmp, remoteWinXTensor_[curXOffset], xNum);
    xInQueue_.EnQue(xTmp);  // 不等待，立即返回

    // scale: Win → UB（VECIN，异步）
    LocalTensor<ScalesType> scaleTmp = scaleInQue.AllocTensor<ScalesType>();
    DataCopyPad(scaleTmp, remoteWinScaleTensor_[curScaleOffset], ...);
    scaleInQue.EnQue(scaleTmp);  // 不等待，立即返回
}

// 新函数2：等待数据并计算（阻塞）
void WaitAndCompute()
{
    // 等待 x 数据到达（VECOUT 端）
    LocalTensor<XType> xTmp = xInQueue_.DeQue<XType>();
    // 等待 scale 数据到达（VECOUT 端）
    LocalTensor<ScalesType> scaleTmp = scaleInQue.DeQue<ScalesType>();

    // 计算
    vecComp_.DequantReduceSum(xTmp, scaleTmp, sumTensor_);

    xInQueue_.FreeTensor(xTmp);
    scaleInQue.FreeTensor(scaleTmp);
}
```

Execute 中的 rank 循环改为：

```cpp
// rank 0: 预取第一个 rank 的数据
PreFetchBlock(curXOffset_rank0, curScaleOffset_rank0, curXNum, curScaleNum);

for (uint32_t i = 1; i < rankDim; ++i) {
    // 发起下一个 rank 的预取
    PreFetchBlock(curXOffset_rankI, curScaleOffset_rankI, curXNum, curScaleNum);
    // 同时等待上一个 rank 的数据并计算
    WaitAndCompute();
}

// 最后一个 rank：只需等待和计算
WaitAndCompute();
```

### 2.3 Queue 类型变更

| 组件 | 当前类型 | 新类型 | 影响 |
|------|---------|--------|------|
| xInQueue_ | `TQue<VECIN, 1>` | `TQueBind<VECIN, VECOUT, 1>` | UB 分配不变，仍为 BUFFER_NUM × xPerBlock × sizeof(XType) |
| scaleInQue | `TQue<VECIN, 1>` | `TQueBind<VECIN, VECOUT, 1>` | UB 分配不变 |

`TQueBind` 的 VECIN/VECOUT 分离允许 EnQue（VECIN）和 DeQue（VECOUT）在不同的硬件流水线上执行，实现真正的搬运/计算重叠。

---

## 3. 涉及文件

| 文件 | 改动内容 |
|------|---------|
| `mc2/quant_all_reduce/op_kernel/quant_all_reduce_mte_one_shot.h` | xInQueue_/scaleInQue 类型变更 + ReadDataBlockReduceSum 拆分 + ExecuteAllReduce rank 循环重构 |
| `mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h` | 同上（QRS 的 ExecuteReduceScatter） |
| `mc2/quant_reduce_scatter/op_kernel/mte_comm.h` | 无需改动 |
| `mc2/quant_reduce_scatter/op_kernel/vec_comp.h` | 无需改动 |
| host tiling | 无需改动（公式不变） |

**不改 UB 空间计算**：buffer 大小由 `xPerBlock_` 和 `scaleNumsPerBlock_` 决定，queue 类型变更不影响分配大小。

---

## 4. 预期收益

| 算子 | 形状 | 当前耗时 | 预期改善 | 依据 |
|------|------|---------|---------|------|
| QAR qwen | 96×4096 | ~276 us | **−5~−15%** | 16 次串行搬运→8 次可重叠，MTE/Vector 并行度翻倍 |
| QAR DSV3 | 96×7168 | ~386 us | **−5~−15%** | 同上，12 次串行搬运 |
| QRS | 48×7168 | ~127 us | **−2~−5%** | 仅 4 次 rank 迭代，重叠收益有限 |

QRS 收益较小因为只有 1 block × 4 ranks = 4 次调用，重叠窗口少。

---

## 5. 测试计划

### 5.1 编译安装
```bash
source ~/env.sh && cd ~/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=quant_all_reduce,quant_reduce_scatter
cd build_out && bash cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=$HOME/SUB
source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash
```

### 5.2 精度验证（先跑）
```bash
cd ~/mc2_test_0416
# QRS 精度
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D
# QAR 精度
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl_npu --world_size 4 --soc 910D
```

### 5.3 性能测试
```bash
# 跑前必查 npu-smi，有占用先 kill
npu-smi info | tail -12

# QRS 全量（清 csv → 跑 → 对比 stage3_perf.csv）
rm -f aclnnQuantReduceScatter_performance_result.csv ...
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl --world_size 4 --soc 910D

# QAR 全量
rm -f aclnnQuantAllReduce_performance_result.csv ...
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl --world_size 4 --soc 910D
```

### 5.4 基线对比
对比文件：
- QRS: `mc2/quant_reduce_scatter/docs/perf/stage3_perf.csv`
- QAR: `mc2/quant_all_reduce/docs/perf/stage3_perf.csv`

### 5.5 验收标准
- 精度：24/24 PASS
- 性能：QAR 总耗时改善 ≥5%（低于此值说明读端DB收益被其他瓶颈掩盖，需 profiling 分析）

---

## 6. 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| TQueBind 替换 TQue 后 EnQue/DeQue 行为变化 | 编译/运行错误 | 先在单算子上验证 |
| BUFFER_NUM=2 不够支撑预取+计算重叠 | 流水线 stall | 必要时升级为 BUFFER_NUM=3 |
| scale 数据量极小，scaleInQue 改为 TQueBind 收益接近零 | 代码复杂度增加 | scaleInQue 可保持不变 |
| 硬件 MTE/Vector 调度器实际重叠率低于预期 | 收益不达标 | 用 msprof 采集确认 |
