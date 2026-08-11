# apace_quant_matmul_reduce_scatter 算子

## 基本信息

- **算子名称**：apace_quant_matmul_reduce_scatter
- **算子类别**：MC2 通算融合（Quant MatMul + ReduceScatter 通信）
- **路线**：APACE MC2（UDMA PUT 直调）
- **支持数据类型**：A/B = fp8_e4m3fn_t / fp8_e5m2_t（4 种组合），C = bfloat16_t
- **支持芯片**：Ascend950PR（DAV_3510）
- **CANN 版本**：9.2.0
- **多卡拓扑**：4 卡 ReduceScatter（AllToAll PUT + AtomicAdd）
- **开发状态**：✅ 功能验证通过（ST 6/6 PASS）

## 数学公式

```
输入:
  rank i 持有:
    A_i  - shape [M, K_local],  dtype = fp8_e4m3fn_t / fp8_e5m2_t   (K_local = K / rankSize)
    ScaleA_i - shape [M, CeilDiv(K_local,64)*2], dtype = fp8_e8m0_t  (MXFP8 scale)
    B_full - shape [K, N],      dtype = fp8_e4m3fn_t / fp8_e5m2_t   (全量复制)
    ScaleB_full - shape [CeilDiv(K,64)*2, N], dtype = fp8_e8m0_t    (全量复制)

输出:
  rank r 获得:
    C_r - shape [M/rankSize, N], dtype = bfloat16_t                 (ReduceScatter 输出)

数学公式:
  C_r = Σ_{i=0}^{rankSize-1} dequant(A_i[r*M/R : (r+1)*M/R, :]) × dequant(B_i[:, :])
```

## 目录结构

```
apace_quant_matmul_reduce_scatter/
├── kernel/apace_quant_matmul_reduce_scatter/   # [MODIFY] 3 个头文件
│   ├── apace_quant_matmul_reduce_scatter_udma_impl.h      # Impl 类 + PipeBarrier 补丁
│   ├── quant_matmul_mx_kernel.h                           # Blaze matmul 调度
│   └── apace_quant_matmul_reduce_scatter_tiling_data.h    # tiling 结构体 + CommContext
├── block/                                      # [REUSE] 共享层（aiv_comm + blaze_ext）
├── tiling/                                     # [REUSE] 通信+计算 tiling
├── basic/                                      # [REUSE] fragment_tensor
├── utils/                                      # [REUSE] comm_channel_builder + constant
├── apace -> .                                  # 符号链接（使 apace/ 前缀 include 解析）
├── st/apace_quant_matmul_reduce_scatter/       # [MODIFY] ST 工程
│   ├── CMakeLists.txt
│   ├── run.sh
│   ├── cases.csv
│   ├── src/
│   │   ├── main.cpp                           # Host 入口（ReduceScatter tiling 推导）
│   │   └── kernel_launcher.h                  # 4 个 __global__ 入口函数
│   └── scripts/
│       ├── gen_data.py                        # ReduceScatter golden 生成
│       └── verify_result.py                   # 精度验证（输出 [M/R, N]）
├── st/utils/                                   # [REUSE] root_info_exchanger + utils
├── st/CMakeLists.txt                           # 顶层构建
├── cmake/third_party/ops-tensor.cmake          # ops-tensor FetchContent
└── docs/                                       # 设计文档
    ├── DESIGN.md
    ├── PLAN.md
    ├── WALKTHROUGH.md
    └── environment.md
```

## 关键设计决策

1. **通信模式**：AllToAll PUT + AtomicAdd（UDMA 引擎）。ReduceScatter 语义通过 AllToAll PUT 发送不同 M 段给不同 rank + AtomicAdd 累加部分和实现。
2. **localMatmul=1**：LOCAL（A_local × B_local → C 首次写入）+ REMOTE（AtomicAdd 累加）。通算并行是 MC2 核心价值，BF16 AtomicAdd 精度损失 ~0.79% 在 FP8 量化容差内。
3. **PipeBarrier<PIPE_ALL>() 补丁**：localMatmul=1 模式下 RunLocalMatmul 和 RunMatmul 之间必须插入 PipeBarrier，确保 LOCAL fixpipe 排空后再 REMOTE AtomicAdd（否则 MTE 异常 507015）。
4. **ProblemShape.K = ka（per-rank K）**：PUT 模式硬性不变量，ka = K/rankSize（非全量 K）。
5. **SetAtomicAdd 由 kernel 内部自动调用**：不显式调用，由 QuantMatmulMxKernel::Run() 内部根据 isAtomicAdd_ 自动调用。

## 编译与运行

### 前置条件

```bash
# 设置 CANN 9.2.0 环境
export ASCEND_HOME_PATH=/home/l00947526/Ascend806/cann-9.2.0
source $ASCEND_HOME_PATH/set_env.sh

# NPU 设备可用（4 × Ascend950PR）
```

### 编译

```bash
cd operators/apace_quant_matmul_reduce_scatter
cmake -S st -B st/build
cmake --build st/build --target apace_quant_matmul_reduce_scatter --parallel 4
```

### 运行测试

```bash
cd st/apace_quant_matmul_reduce_scatter

# 运行全部用例
bash run.sh

# 运行指定用例（行号）
bash run.sh 1 3

# 命令行模式
bash run.sh --cli 2048 4096 2048 4

# 性能模式
bash run.sh --perf 1
```

### 精度标准

- 输出 dtype：bfloat16_t
- 容差：rtol=atol=1e-2（BF16 输出，FP8 量化）
- 预期 max_rel_diff：≈ 0.79%（localMatmul=1 的 BF16 AtomicAdd 精度损失）

## 测试结果

| 编号 | M | K | N | rankNum | 结果 | rel_err max |
|-----|-----|-----|-----|---------|------|-------------|
| T1 | 2048 | 4096 | 2048 | 4 | ✅ PASS | ≈0.78% |
| T2 | 512 | 1024 | 512 | 4 | ✅ PASS | ≈0.78% |
| T3 | 8192 | 8192 | 4096 | 4 | ✅ PASS | ≈0.78% |
| T4 | 1120 | 2048 | 1024 | 4 | ✅ PASS | ≈0.78% |
| T7 | 2048 | 4096 | 2048 | 2 | ✅ PASS | ≈0.78% |
| T8 | 2048 | 4608 | 2048 | 4 | ✅ PASS | ≈0.78% |

## APACE 路线约束（红线）

- ✅ 无 `__schedmode__(1)` / `[[bisheng::core_ratio(1,1)]]`
- ✅ 有 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)`（4 个入口函数）
- ✅ 无 `Hccl::` 高阶 API
- ✅ 无 `AscendC::Matmul`（Matmul 走 Blaze 模板）
- ✅ `block/` `tiling/` `basic/` `utils/` 未修改（与官网一致）
- ✅ localMatmul=1 时有 `PipeBarrier<PIPE_ALL>()`

## 参考资料

- 设计文档：[docs/DESIGN.md](docs/DESIGN.md)
- 开发计划：[docs/PLAN.md](docs/PLAN.md)
- 设计串讲：[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)
- MC2 最佳实践：`/ascendc-mc2-best-practice` skill
- Blaze 开发指南：`/ascendc-blaze-best-practice` skill
