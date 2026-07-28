# AllGatherQuantMatmul（Prefill）

## 概述

AllGatherQuantMatmul 是通算解耦的 MX 量化矩阵乘算子，用于 prefill 阶段。AIV 负责 AllGather 通信，AIC 负责量化 Matmul 计算，两者通过 cross-core flag 同步，实现通信与计算 overlap。

数学定义：

$$C = \text{AllGather}(A) \times B$$

其中 A、B 为 fp8_e4m3fn_t 量化矩阵，带 per-group fp8_e8m0_t scale，C 为 bfloat16_t。

## 架构

```
AIV (向量核)                    AIC (矩阵核)
    │                               │
    ├─ 预触发 dependId=0             │
    ├─ round 0: AllGather ──flag──▶  ├─ HEAD 段（自身数据，无需等待）
    ├─ round 1: AllGather ──flag──▶  ├─ MAIN round 0（等待 dependId=1）
    ├─ ...                           ├─ MAIN round 1（等待 dependId=2）
    ├─ tail: AllGather ──flag──▶     ├─ TAIL 段（等待 dependId=commTurn）
    │                               │
    ▼                               ▼
```

- **Win buffer 布局**：rank-major，每个 rank 占 M 行，data + scale 两段连续存放
- **logical M 调度**：AIC 将 head + main 段 + tail 拼接为统一 logical M，由 BlockScheduler 按 mPos 分发到对应 region

## 关键优化

| 优化 | 说明 |
|------|------|
| **FragmentTensor 抽象** | 将多个离散 GM fragment 封装为统一 Tensor，支持 Slice/Copy/Scatter |
| **延迟 FT 构建** | HEAD 的 fragment tensor 立即构建，MAIN/TAIL 延迟到首次命中时构建，构建开销被 comm wait 掩盖 |
| **UpdateAddrList** | MAIN round 切换时仅更新地址列表，不重建整个 fragment tensor |
| **EnsureWinRankBasesReady** | 各 rank 的 win buffer 基地址仅计算一次，后续 round 复用 |

## 数据类型

| 角色 | 类型 |
|------|------|
| A | fp8_e4m3fn_t |
| B | fp8_e4m3fn_t |
| A scale / B scale | fp8_e8m0_t |
| C | bfloat16_t |

## 文件

| 文件 | 说明 |
|------|------|
| `all_gather_matmul_impl.h` | 算子实现：Init → AllGatherProcess / MatmulProcess |
| `all_gather_matmul_tiling_data.h` | Tiling 数据结构，含 fragment tiling 预计算 |

## 测试

```bash
cd apace/tests/st/all_gather_quant_matmul_prefill
bash run.sh          # 运行 cases.csv 全部用例
bash run.sh 1        # 运行第 1 行
bash run.sh --cli M K N rankNum  # 命令行模式
```

测试用例（4 ranks，fp8 in / bf16 out，bit-exact 验证）：

| # | M_per_rank | K | N |
|---|-----------|------|------|
| 1 | 1010 | 4096 | 6656 |
| 2 | 2304 | 4096 | 6656 |
| 3 | 4040 | 4096 | 6656 |

## 依赖

- `apace/basic/fragment_tensor/` — FragmentTensor 模板类
- `apace/block/blaze_ext/gemm/block/block_mmad_qbmm_mx_fragcout.h` — FragmentTensor 输出的 BlockMmad
- `apace/block/aiv_comm/` — AIV 通信库（CollectiveComm、barrier）
- `blaze/gemm/` — QBMM MX 矩阵乘 Kernel + BlockScheduler
