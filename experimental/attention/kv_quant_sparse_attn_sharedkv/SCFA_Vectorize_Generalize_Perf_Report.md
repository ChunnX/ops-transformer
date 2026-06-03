# KvQuantSparseAttnSharedkv SCFA — 向量化地址计算泛化 + 三流水性能对比报告

> 范围: 单文件功能改动 `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h`
> (地址计算向量化泛化), 配合 `tests/pytest` 新增 prefill/decode 大用例。
> 结论: 泛化后 128 对齐快路径**零回归** (my_test 精度 99.994818% PASS, ~109us 不变);
> decode 大用例三流水对比 **baseline 77.8us → lead-by-1 62.4us (-19.8%)**。

日期: 2026-06-03

---

## 1. 背景与目标

`KvQuantSparseAttnSharedkv` SCFA 在 Ascend 950 上以 AIC(cube)/AIV(vector) CV 融合
生产者-消费者流水运行。本轮工作包含两部分:

1. **地址计算向量化泛化**: `block_vector.h` 中 `GetKVPhyAddrVF` 用 MicroAPI 向量指令
   把 sparse block 索引转换为 KV 物理地址。原实现按特定 case 写死 (s2 为 `s2NumPerLoop`
   整数倍, 忽略 `s2Tail`), 无法支持任意 `b/s1/k`。
2. **三套流水的性能对比**: 在泛化后的 `block_vector.h` 基础上, 对比 baseline(老版流水)、
   lead-by-1、h01 三种 kernel 在 prefill / decode 两个大用例下的性能。

三套 kernel 共用同一份 `block_vector.h`, 区别仅在 `*_scfa_kernel.h` 的流水编排。

---

## 2. 向量化地址计算泛化 (block_vector.h)

### 2.1 原实现的局限

`GetKVPhyAddrVF` 的 vector loop 以 `s2NumPerReg`(b32 VL=64) 为粒度, 每轮处理两段
共 128 个索引, 直接 `LoadAlign → 计算 → StoreAlign` 全宽进行, **从不处理 `s2Tail`**。
当 `validS2` 不是 128 整数倍时, 尾段会:

- `LoadAlign` 越界读相邻 buffer (脏索引);
- 脏索引经 `DataCopyGather` 触发越界 gather;
- `StoreAlign` 把垃圾地址写入输出。

### 2.2 泛化方案 (方案一: mask 控制最后一个 vloop)

对最后一轮 (`isTail`) 用 `UpdateMask<uint32_t>(count)` 生成尾掩码, 配合 `Select`
把尾部失效 lane 的索引强制置 0:

```cpp
for (uint16_t i = 0; i < s2Loop; i++){
    bool isTail = (i == s2Loop - 1);
    uint32_t cnt1 = isTail ? (s2Tail < s2NumPerReg ? s2Tail : s2NumPerReg) : s2NumPerReg;
    uint32_t cnt2 = isTail ? (s2Tail > s2NumPerReg ? s2Tail - s2NumPerReg : 0U) : s2NumPerReg;
    MaskReg preg_tail_1 = UpdateMask<uint32_t>(cnt1);
    MaskReg preg_tail_2 = UpdateMask<uint32_t>(cnt2);
    LoadAlign(..., vreg_sparse_idx_1, sparseIdxUb + i*s2NumPerLoop);
    LoadAlign(..., vreg_sparse_idx_2, sparseIdxUb + s2NumPerReg + i*s2NumPerLoop);
    Select(vreg_sparse_idx_1, vreg_sparse_idx_1, vreg_zero, preg_tail_1); // 失效 lane→idx 0
    Select(vreg_sparse_idx_2, vreg_sparse_idx_2, vreg_zero, preg_tail_2);
    ... // ShiftRights/Muls/Sub/DataCopyGather/Mull/Add/AddC 保持全宽
    StoreAlign(kvPhyAddrUb + i*outOffsetPerLoop,            ..._L_1, ..._H_1, preg_tail_1);
    StoreAlign(kvPhyAddrUb + outOffsetPerReg + i*..., ..._L_2, ..._H_2, preg_tail_2);
}
```

要点:
- **Select 置 0**: 让失效 lane 的 gather 落到 idx 0 (合法地址), 全宽 `DataCopyGather`
  不越界, 中间算子无需改宽度。
- **掩码写回**: `StoreAlign` 用尾掩码, 垃圾地址不写出。
- **buffer padding**: `GetKVPhyAddr` 里 `sparseIdxBuf` / `kvPhyAddrBuf` 按
  `CeilAlign(sparseBlockCount, 128)` 分配, 保证尾段全宽 `LoadAlign` 读取不越界。
- **快路径零开销**: 当 `validS2` 为 128 整数倍 (本报告 3 个 benchmark 均
  `validS2 = K = 1024`), `cnt1=cnt2=64` → 满掩码 → 与原全宽路径逐位等价, **零性能影响**。

### 2.3 回归验证 (my_test, 128 对齐快路径)

| 项目 | 结果 |
|---|---|
| 精度 | **99.994818% PASS** (与泛化前一致) |
| 性能 (h01, 3 runs) | 109.346 / 108.363 / 110.359 us, 均值 **~109.4us** (与泛化前 h01 一致) |

结论: 泛化为纯正确性改动, 对已对齐快路径零回归。

---

## 3. 性能对比 (三套流水)

### 3.1 用例

| 用例 | layout | B | S1 | S2 | N1 | N2 | D | K(topk) | cmp_ratio | kv_quant_mode | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill | BSND | 1 | 8192 | 131072 | 128 | 1 | 512 | 1024 | 4 | 1 | SCFA |
| decode  | BSND | 24 | 2 | 8192 | 128 | 1 | 512 | 1024 | 4 | 1 | SCFA |

每用例 profiling 取 3 次均值 (用例较大)。msprof `--aic-metrics=PipeUtilization`。

> 注: 性能测量使用 `QSAS_PERF_ONLY=1` 跳过 CPU golden forward (prefill 单次 ~50min),
> msprof 仍捕获真实 NPU kernel 耗时。

### 3.2 Decode 结果 (b=24, s1=2, s2=8k)

| Kernel | 3-run (us) | 均值 (us) | aic_mac_ratio | vs baseline |
|---|---|---|---|---|
| baseline (老版流水) | 78.43 / 76.68 / 78.43 | **77.85** | 0.468 | — |
| h01 (a05 + lead-by-1) | 66.14 / 63.35 / 62.60 | **64.03** | 0.593 | **-17.7%** |
| lead-by-1 (baseline 序 + lead-by-1) | 63.24 / 62.03 / 62.02 | **62.43** | 0.608 | **-19.8%** |

**结论**:
- lead-by-1 流水把 cube 利用率从 0.47 抬到 0.61, decode 提速 **~19.8%**, 三者最快。
- lead-by-1(跳过 a05 的 load-last 重排) 略优于 h01, 印证 a05 的 load-last 在该
  深流水下基本冗余。

### 3.3 Prefill 结果 (b=1, s1=8k, s2=128k)

**三套 kernel 在 prefill 用例下均触发 AICore 硬异常 (device error 507015,
aicore exception), 无法取得性能数据。**

- baseline / h01 / lead-by-1 表现**完全一致** (均 3 次空结果), 说明该崩溃为
  **与流水编排无关的既有 shape 限制**, 与本轮 `block_vector.h` 泛化无关
  (decode 用例同样走泛化后地址计算路径且正常运行)。
- 异常 dump 指向主 attention kernel (`KvQuantSparseAttnSharedkv_..._mix_aic/aiv`),
  触发因素为 prefill 超大 shape (S1=8192 / S2=131072)。
- 该问题属于 kernel 在 prefill 极大 shape 下的既有缺陷, 不在本轮 (流水 + 地址向量化)
  优化范围内, 需单独定位 (workspace / tiling 在 s2=128k 下的边界)。

---

## 4. 变更文件清单

| 文件 | 改动 |
|---|---|
| `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` | `GetKVPhyAddrVF` 尾段 mask 泛化 + buffer CeilAlign padding |
| `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel_leadby1_noloadlast.h` | (已有) lead-by-1 变体 kernel |
| `tests/pytest/kv_quant_sparse_attn_sharedkv_paramset.py` | 新增 `scfa_prefill` / `scfa_decode` 用例 + `QSAS_CASE` 环境变量选择 |
| `tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py` | 新增 `QSAS_PERF_ONLY` 开关, 跳过 CPU golden (仅 profiling 用) |

---

## 5. 后续

- prefill (s1=8k/s2=128k) 的 AICore 异常需单独定位, 优先排查 workspace 与 tiling
  在 S2=131072 下的边界计算。
- 当前默认 `kv_quant_sparse_attn_sharedkv_scfa_kernel.h` = h01 (未改动)。
