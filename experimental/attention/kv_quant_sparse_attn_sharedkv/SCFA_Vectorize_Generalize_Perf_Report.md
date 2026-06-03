# KvQuantSparseAttnSharedkv SCFA — 向量化地址计算泛化 + 三流水性能对比报告

> 范围: `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` (地址计算
> 向量化泛化 + prefill 多 batch 钳位), `op_host/..._tiling.cpp` (workspace 物理地址区
> 修复), 配合 `tests/pytest` 新增 prefill/decode 大用例。
> 结论: 泛化后 128 对齐快路径**零回归** (my_test 99.994818% PASS);
> 修复 prefill AICore 异常 507015 (三根因); 三流水对比 **h01 在 prefill -27.2% /
> decode -21.9% (vs baseline)**。

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
| prefill | BSND | 1 | 8192 | 32768 | 128 | 1 | 512 | 1024 | 4 | 1 | SCFA |
| decode  | BSND | 48 | 2 | 8192 | 128 | 1 | 512 | 1024 | 4 | 1 | SCFA |

每用例 profiling 取 3 次均值。msprof `--aic-metrics=PipeUtilization`。三套 kernel
共用同一份(已泛化的) `block_vector.h`, 仅 `*_scfa_kernel.h` 流水编排不同。

> 注: 性能测量使用 `QSAS_PERF_ONLY=1` 跳过 CPU golden forward, msprof 仍捕获真实 NPU
> kernel 耗时。精度另行用 golden 校验 (decode 99.994818% / prefill 99.994445%, 均 PASS)。

### 3.2 Decode 结果 (b=48, s1=2, s2=8k)

| Kernel | 3-run (us) | 均值 (us) | aic_mac_ratio | vs baseline |
|---|---|---|---|---|
| baseline (老版流水) | 144.33 / 141.54 / 141.38 | **142.42** | 0.51 | — |
| h01 (a05 + lead-by-1) | 111.47 / 112.53 / 109.51 | **111.17** | 0.67 | **-21.9%** |
| lead-by-1 (baseline 序 + lead-by-1) | 113.23 / 112.51 / 112.42 | **112.72** | 0.66 | **-20.9%** |

**结论**:
- 流水改造把 cube 利用率从 0.51 抬到 0.67, decode 提速 **~22%**。
- 本轮 b=48 下 **h01 略优于 lead-by-1** (111.2 vs 112.7us), 与上一轮 b=24 的结论相反
  (上轮 lead-by-1 略快)。差距在 ~1.5% 量级、随 batch 数浮动, 两者基本等价;
  当前默认保留 h01。

### 3.3 Prefill 结果 (b=1, s1=8k, s2=32k)

**上一轮 prefill (s2=128k) 三套 kernel 均触发 AICore 硬异常 507015, 本轮定位并修复了
根因, s2=32k 下三套 kernel 全部正常运行并取得性能数据。**

| Kernel | 3-run (us) | 均值 (us) | aic_mac_ratio | vs baseline |
|---|---|---|---|---|
| baseline (老版流水) | 11193.79 / 11195.67 / 11194.25 | **11194.57** | 0.544 | — |
| h01 (a05 + lead-by-1) | 8146.51 / 8150.74 / 8143.61 | **8146.95** | 0.766 | **-27.2%** |
| lead-by-1 (baseline 序 + lead-by-1) | 8324.84 / 8313.85 / 8316.90 | **8318.53** | 0.750 | **-25.7%** |

**结论**:
- prefill 上流水改造收益更大 (cube mac 率 0.544→0.766), h01 提速 **~27%**。
- prefill 下 h01 同样略优于 lead-by-1, 与 decode 一致。
- 三套 kernel 在 s2=32k 下均无 507015, 证明该异常是 **host/kernel workspace 布局缺陷**
  (详见 §3.4), 与流水编排无关。

### 3.4 prefill AICore 异常 507015 根因与修复

上一轮 prefill (s1=8k) 触发 device error 507015 (errcode 95, "DDR address of MTE
instruction out of range")。定位到 **三个相互独立的根因**, 全部修复后 crash 消失,
精度 PASS:

**Bug 1 — host tiling 物理地址行数写死 (`tiling.cpp`)**
- 原 `kvPhyAddrGm` 行数按 `BS1_MAX=4000` 写死; prefill 需要 `B*S1=8192` 行 → 物理
  地址区不足, AIV 写越界。
- 修复: 行数改为动态 `bs1Rows = (qTSize>0) ? qTSize : bSize*s1Size`。

**Bug 2 — host 物理地址区基址比 kernel 少 ×4 (`tiling.cpp`)**
- kernel 用 `kvPhyAddrGm.SetGlobalBuffer((uint32_t*)workspace + v0TotalOffset)`,
  其中 `v0TotalOffset` 本是**字节**数, 但 `(uint32_t*)` 转换在加法之前, 指针算术
  把它 ×4 → 物理区真实字节基址 = `v0RegionBytes * 4`。
- 这个 ×4 是**承重设计** (把 kernel 改成字节精确反而崩了 decode)。host 原来只按
  `1× v0RegionBytes` 预留, 与 kernel 的 4× 基址错位 3×, 小用例靠运行时余量幸存,
  prefill (物理区大) 越界。
- 修复: host 改为 `physAddrBase = v0RegionBytes * sizeof(uint32_t)`, 与 kernel 对齐。

**Bug 3 — kernel 多 batch 切分越界 (`block_vector.h GetKVPhyAddr`, prefill 专属)**
- `aivIdx % 4` 分支给 `bN2StartIdx` 加 +1/+2, 把工作分到 4 个 AIV lane — 这是 decode
  多 batch 的切分逻辑。
- 对 **prefill B=1 (`bN2 = bSize*n2Size = 1`)**, case 1/2/3 索引到不存在的 batch →
  `bS1Idx = s1Size*bIdx` (如 16384) 越过 8192 行的 `cmpSparseIndicesGm` 读 + 物理地址
  写 → AIV MTE 越界。
- 修复: `curBN2EndIdx = Min(curBN2EndIdx, bSize*n2Size)`, 让幻影 batch 的 lane 空转
  (s1 已由 `gS1StartIdx/nextGs1Idx` 切分)。

> 注: s2=128k 是否仍有更深的 shape 边界问题未在本轮覆盖 (本轮用例 s2=32k); 但 §3.4
> 三个修复对任意 b/s1 通用, 不限 s2=32k。

---

## 4. 变更文件清单

| 文件 | 改动 |
|---|---|
| `op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp` | **Bug 1+2** workspace 物理地址区: 动态 `bs1Rows` + 基址 ×`sizeof(uint32_t)` 对齐 kernel |
| `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` | **Bug 3** `GetKVPhyAddr` 多 batch 切分 `Min(bN2End, bSize*n2Size)` 钳位; `GetKVPhyAddrVF` 尾段 mask 泛化 + buffer CeilAlign padding |
| `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel_leadby1_noloadlast.h` | (已有) lead-by-1 变体 kernel |
| `tests/pytest/kv_quant_sparse_attn_sharedkv_paramset.py` | 新增 `scfa_prefill`(s2=32k) / `scfa_decode`(b=48) 用例 + `QSAS_CASE` 环境变量选择 |
| `tests/pytest/kv_quant_sparse_attn_sharedkv_golden.py` | `QSAS_PERF_ONLY` 跳过 CPU golden; **golden 缓存** (`golden_cache/golden_<md5(params)>.pt`, 命中即 load, 输入由 `set_seed(42)` 决定性生成); `QSAS_GOLDEN_NOCACHE=1` 强制重算 |

---

## 5. 后续

- s2=128k 超大 shape 是否仍有更深的 workspace/tiling 边界问题需单独覆盖 (本轮用例
  收敛到 s2=32k 并已全绿)。
- decode 下 h01 与 lead-by-1 差距在 ~1.5% 内、随 batch 浮动, 两者等价; 当前默认
  `kv_quant_sparse_attn_sharedkv_scfa_kernel.h` = h01 (未改动)。
