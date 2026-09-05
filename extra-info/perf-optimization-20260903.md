# Flash MLA With KVCache 性能优化探索记录（2026-09-03）

## 目标

mla_pref.py 用例（B=1 N1=96 S1=8 S2=128000 D=576 DV=512, TND PA_NZ bf16, mask_mode=3）
从 ~780us 优化到 650us。

## 基线数据

- **原始完整版**：Task Duration ~780us，cube 72.7%，MTE2 72.9%（56.6us），MTE1 58.5%，AIV vec 35.7%
- **RoofLine**：Latency Bound（pipeline caused）
- **MTE2 带宽利用率**：45.8%（远未饱和）
- **核间均衡**：28 核 AIC wall 777-783us（方差 <1%），不存在负载不均
- **瓶颈判定**：依赖链等待（L0 数据就绪 + AIV softmax 串行 + 跨核旗标），非带宽/非 Bank 冲突

## 实验记录

### 1. 576 DMA 连续拷贝（已合入 c55cf07b8）

将 CopyQueryAndRopeTile / CopyKeyAndRopeTile 中两次 strided 拷贝（nope 512 + rope 64，各 576 行 stride）合并为一次 576 宽连续拷贝。srcDValue == dValue == 576，消除 GM 行间 gap。

### 2. MatmulK 分解实验

通过注释 matmul.h 的 MatmulK/MatmulN 中 LoadData 和 Wait/Set 做分解：

| 版本 | Duration | cube 利用率 |
|---|---|---|
| 完整版 | 780us | 72.7% |
| 同步保留+拷贝注释 | 687us | 82.3% |
| 只留 Mmad（全注释） | 585us | 98.7% |

**结论**：Wait/Set 同步开销 ~102us，LoadData（MTE1 L1→L0）开销 ~93us。

### 3. MatmulKPrefetch 软件流水

在 matmul.h 新增 MatmulKPrefetch 函数，LoadData(k+1) 在 Mmad(k) 前发出与计算重叠。
结果：796us，cube 98.6%——**无改善**（原版双槽 DB 已做大部分重叠，prefetch 增加 scalar 16us 和 MTE2 压力）。

### 4. s2BaseSize 256 方案

L0B 单槽（32K）硬限制：128×256×2=64K > 32K。L0A/L0B 硬件各 64KB，2×32K DB 已占满，不可扩。

### 5. L2 Cache 关闭（已合入 d52c22d83）

tiling 控制的 l2CacheOffFlag，gSize*s1Size<=128 时关闭。

### 6. FlashDecode 排查

printf 确认：enableFlashDecode=1 但所有 AIV core 的 fdCoreEnable=0（无实际 FD 任务）。注释 FD 前后 793us 不变，确认非瓶颈。

### 7. Q/K GM→L1 拷贝实验

注释 CopyQueryAndRopeTile + CopyKeyAndRopeTile（仅保留 MatmulK 的 Mmad 和核内同步）：
- 636us，cube 98.6%——比完整版(-157us)
- 确认 MTE2 GM→L1 搬运是最大瓶颈

## 关键代码位置

- `flash_mla_with_kvcache_block_cube.h`：IterateBmm1（L327-374）、CopyQueryAndRopeTile（L267-298）、CopyKeyAndRopeTile（L299-326）
- `flash_mla_with_kvcache_kernel.h`：ExecuteTask（L479-503）、FlashDecode（L635-651）
- `matmul.h`：MatmulK（L778-858）、MatmulKPrefetch（L864+）
- `flash_mla_with_kvcache_metadata_aicpu.cpp`：metadata producer（L248 fdOn=true, L313 HEAD_IS_FD_INDEX）

## 剩余可行方向

1. **KV L1 槽 3→4**：加深 MTE2 流水，L1 1MB 当前用 504K，4 槽=648K 放得下
2. **跨核 sync 批量**：28 核抢 2+2 旗标槽，每 2 任务合一次
