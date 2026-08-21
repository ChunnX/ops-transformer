# 用例 A3p1_FLco_bf16_T2048_K8_c64_e256_ae128 聚合搬出不启用根因分析

## 1. 用例参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n | 2048 | token 数 |
| k | 8 | top-k |
| cols | 64 | 列宽 |
| expert_num | 256 | 总专家数 |
| active_expert_range | [0,128] | actualExpertNum=128 |
| drop_pad_mode | 0 | DROPLESS |
| expert_tokens_num_type | 1 | COUNT |
| expert_tokens_num_flag | True | |
| quant_mode | -1 | UNQUANT |
| row_idx_type | 0 | SCATTER |
| expert_capacity | 1 | |
| x dtype | bfloat16 | dtypeSize=2 |

## 2. 走全载模板的判定（已通过）

### 2.1 排除分支（L138-142）不触发
排除条件首项 `n_ <= CS_COUNTING_SORT_THRESHOLD(1536)` → `2048 <= 1536` 假，排除分支不触发，进入 FullLoad 判定（else 分支 L145）。

### 2.2 FullLoad 条件（L148-152）满足
- `estUb ≈ 51.7KB`（含分桶区 3.7KB）≤ `availUbSize - 1024 = 128K - 1K = 127K` ✓
- `ep_==1 ✓ / quantMode_==UNQUANT ✓ / k_==8 ✓ / dropPadMode_==DROPLESS ✓`
- `expertTokensNumType_==COUNT ✓ / actualExpertNum=128 <= 128 ✓ / actualExpertNum>0 ✓`

→ **FullLoad 模板启用**，`ComputeArch35CountingSortFullLoadTiling()` 被调用，`csAggrEnable` 计算在此函数内。

## 3. 聚合搬出不启用的根因

### 3.1 聚合判定条件（`ComputeArch35CountingSortFullLoadTiling` L2131-2135）
```cpp
bool aggrEnable = (aggrOutRows >= NUM_TWO) &&
                  (xLocalCap - aggrBufBytes >= perCoreTokens * cols_ * inputXDtypeSize_) &&
                  (actualExpertNum <= CS_FULLLOAD_MAX_ACTUAL_EXPERT_NUM) &&
                  (quantMode_ == QUANT_MODE_UNQUANT);
```

### 3.2 数值代入（假设 aivCoreNum=24）
- `perCoreTokens = ceil(2048/24) = 86`
- `colsAligned = align(64*2, 32)/2 = 64`（元素）
- `xLocalCap = align(86 * 64 * 2, 32) = 11008` B（xLocal 原容量）
- `aggrBufBytes = min(10*1024, xLocalCap/2) = min(10240, 5504) = 5504` B
- `aggrOutRows = 5504 / (64*2) = 5504/128 = 43` → `>= 2` ✓
- **关键**：`xLocalCap - aggrBufBytes = 11008 - 5504 = 5504` B
  - 搬入需求 `perCoreTokens * cols_ * inputXDtypeSize_ = 86 * 64 * 2 = 11008` B
  - `5504 >= 11008` → **假** ✗

### 3.3 根因
**xLocal 切走 aggrBufBytes 后，剩余容量（5504B）装不下搬入需求（11008B）。**

判定条件 `xLocalCap - aggrBufBytes >= perCoreTokens * cols_ * dtypeSize` 正确地拒绝了聚合，避免搬入区溢出。这是**安全保护**，非 bug。

### 3.4 根本矛盾
- `aggrBufBytes` 上限定为 `min(10K, xLocalCap/2)`，本例 `xLocalCap/2 = 5504` 就是上限。
- 但 `xLocalCap` 本身只有 ~11KB（cols=64 bf16，每行 128B，86 行）。
- 切走一半（5.5KB）后搬入区不够装 86 行（11KB）。
- **聚合区与搬入区在 xLocal 内争用空间，本例 xLocal 太小，二者无法共存。**

## 4. 为何 xLocal 这么小

- `cols=64`、`bfloat16`(2B)：每行 128B，是窄列场景。
- `perCoreTokens=86`：n=2048 分 24 核。
- 86 行 × 128B = 11KB，确实小。
- 聚合区要装 ≥2 行才有意义（k≥2），每行 128B，最少 256B，但 `aggrBufBytes` 取的是 `xLocalCap/2=5.5KB`，占走一半太激进。

## 5. 修复方向

### 5.1 问题：aggrBufBytes 上限公式不合理
`min(10K, xLocalCap/2)` 对小 xLocal 场景切走过多。应改为**基于"搬入后剩余"反推**：

```
可用聚合区 = xLocalCap - 搬入需求 = xLocalCap - perCoreTokens*cols*dtypeSize
aggrBufBytes = min(10K, max(0, 可用聚合区))
aggrOutRows = aggrBufBytes / (colsAligned*dtypeSize)
aggrEnable = (aggrOutRows >= 2) && (可用聚合区 >= aggrBufBytes) && ...
```

但本例 `可用聚合区 = 11008 - 11008 = 0` → `aggrBufBytes=0` → `aggrOutRows=0` → 不启用。**xLocal 刚好够装搬入，无空间给聚合区**，聚合本就不该启用。

### 5.2 核心结论
本例 xLocal 容量与搬入需求几乎相等，**没有多余空间给聚合区**。聚合搬出需要 xLocal 有"搬入余量"，窄列（cols 小）+ 单核 token 多的场景，xLocal 全被搬入占满，聚合无空间可用。

### 5.3 哪些场景能启用聚合
需 xLocal 有足够余量：
- **cols 大**（每行大，但 xLocal 按 perCoreTokens 行分配，perCoreTokens 会因 UB 限制变小，xLocalCap 占比可能仍紧）。
- **perCoreTokens 小**（n 小或核数多，搬入需求小，xLocal 余量大）。
- **dtype 小**（fp8/int8 比 bf16 更省空间）。

典型可启用：cols=128+、n≤1024、fp8/int8 场景，xLocal 搬入后仍有 ≥2KB 余量。

## 6. 当前判定逻辑正确性

当前判定（`xLocalCap - aggrBufBytes >= 搬入需求`）是**正确的安全保护**：
- 满足 → 聚合启用（xLocal 切分后仍能装搬入）。
- 不满足 → 聚合不启用，回退逐行（保证搬入正确）。

本例判定为假，回退逐行路径，**行为正确，输出无误**。聚合搬出在该用例的 xLocal 容量下本就不可行。

## 7. 是否需调整

### 7.1 若希望本例启用聚合
需增大 xLocal 余量，途径：
- 增大核数（减小 perCoreTokens）→ 但核数固定。
- 减小 aggrBufBytes（让聚合区更小）→ 但 aggrOutRows 会 < 2，仍不启用。
- **结论：本例 xLocal 容量物理上无法同时容纳搬入+聚合区，不可启用。**

### 7.2 若希望扩大聚合适用范围
可让聚合区**不占 xLocal**（从 commonBuf 末尾新增独立区），但会增加总 UB 占用，需重新校验 UB 预算，且与"不增加总 UB"的设计目标冲突。

## 8. 结论

**用例走全载模板但聚合搬出不启用的根因：xLocal 容量（~11KB）与搬入需求（~11KB）几乎相等，切走聚合区后搬入区不够装，判定条件正确拒绝聚合。**

这是窄列（cols=64）+ bf16 + perCoreTokens=86 场景的固有容量限制，非 bug。聚合搬出需要 xLocal 有"搬入余量"，本例物理上无此余量。判定逻辑正确保护了搬入正确性，回退逐行路径，输出无误。