# MxFP8 低精方案整体总结

## 方案总览

| 项目 | 内容 |
|------|------|
| 目标 | MxFP8 支持低精度 softmax，V1 的 max 统计量用 fp16，高精为 fp32（边界对齐 MxFP4 mode 5 实际实现） |
| quant_mode | 3（A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16） |
| 场景 | 主打 PR（Prefill），严格限定 DN 模板 |
| 主体代码参考源 | 主线 MxFP8 高精 `/home/w00947241/ops-transformer_fp16/attention/quant_flash_attn` |
| 开发落地位置 | `/home/w00947241/ops-transformer_fp16/experimental/attention/quant_flash_attn`，新建 mxfp8 低精专属文件 |

三大低精特征（相对主线 MxFP8 高精的差异）：

1. **fp16 softmax max**：仅 V1 的 softmax max 统计量从 float 改 half；rowsum 走 Cube 后 V2 全链路保持 fp32
2. **单轮 C1V1**：消除高精 DN 路径 s2Split 分裂出的 subLoop 0/1 两轮 C1V1，回归 C1/V1/C2/V2 正常轮次
3. **rowsum 挪 Cube**：softmax 分母 sum 由 Cube 侧 mm2 顺带计算（fp32），Vector 侧不再算 expSum

fp16 精确边界（对齐 MxFP4 实际实现，已逐行核实）：

| 环节 | 精度 |
|------|------|
| V1 softmax max 统计 | **half**（MxFP4 的 localGroupMaxUB/localGlobalMaxUB/softmaxMaxUB） |
| V1 P 矩阵输出 | fp8_e4m3 量化 |
| Cube rowsum（L0C 累加 + fixpipe） | fp32（MxFP4 FixpipeMm2 COMPUTE_T=float） |
| V2 rowsum 累积/更新/缩放 | fp32（MxFP4 processUpdateVF 全 float） |
| V2 输出累积 + 除法 | fp32（MxFP4 processOutVF 全 float） |
| 最终输出 | bf16 |

整体数据流：

```
Q/K (fp8_e4m3) ──> C1: mm1 (Cube) ──> L0C ──fixpipe──> UB (fp32)
                                                        |
                     V1: exp + half max统计 + P量化 (Vector)
                     [max 走 Vector(half), sum 由 Cube 产出]
                                                        v
V (fp8_e4m3) ──> C2: mm2 [V | ones] (Cube) ──> L0C [P·V | P·ones(rowsum)] (fp32)
                                                        |
                     V2: rowsum累积 + 除法 + 输出 (Vector, 全fp32) <──fixpipe──+
```

---

## 一、Kernel 侧

新建 `op_kernel/arch35/quant_flash_attn_kernel_mxfp8_dn.h`，参考主线 `quant_flash_attn_kernel_mxfp8.h`。

### 1. 流水轮次：C1/V1/C2/V2 正常轮次

- `PRELOAD_N = 2` 软件流水沿用主线结构：C1→C1→C1→C2，loop 内本轮 C1、延迟 PRELOAD_N 轮做 C2/V2
- 该结构即"正常轮次"，与高精的区别不在流水框架，而在 subLoop：

### 2. 消除两轮 C1V1

- 主线 L593-594 根源：`c1v1Loop = CeilDiv(actSingleLoopS2Size, s2SplitSize)`，s2SplitSize 为 128/256 时 actSingleLoopS2Size 大于它则分裂两轮 subLoop C1V1
- 低精方案：tiling 侧 `s2SplitSize` 取整块 `s2BaseSize`，使 `c1v1Loop` 恒为 1
- 实现建议：模板参数编译期裁剪 subLoop 分支（如 `IS_SINGLE_C1V1`），而非运行时判断，避免残留分支开销

### 3. 跨核同步机制

- 沿用主线 `CROSS_CORE_SYNC_*` 同步 ID 框架
- 新增 rowsum 通路依赖：Vector V2 阶段（LastDivNew 读 rowsum）必须等 Cube fixpipe 搬出 rowsum 完成的事件
- 同步 ID 需在 tiling 与 kernel 两侧同步规划，避免与既有 ID 冲突

### 4. 实例化

- `op_kernel/quant_flash_attn.cpp` 新增 mxfp8 低精实例化分支，`SOFTMAX_T=half`
- 仅 DN+PR 路径实例化，控制编译产物膨胀

---

## 二、Block Cube 侧

新建 `op_kernel/arch35/quant_flash_attn_block_cube_mxfp8_dn.h`，参考主线 `quant_flash_attn_block_cube_mxfp8.h`。

### 1. mm1 通路不变

- Q·K mm1 → L0C → fixpipe → UB 供 Vector 做 exp 的数据通路完全沿用高精，无改动

### 2. rowsum 挪 Cube（核心改动）

- `IterateBmm2Dn()`：B 侧 V 矩阵追加全 1 列构造 `[V | ones]`，mm2 一次算出 `[P·V | P·ones]`，L0C 中额外得到每行 rowsum（即 softmax 分母 sum）
- ones 块来源：L1 中预置常量全 1 矩阵，随 V 一起 Load，仅初始化一次
- 约束：N 维对齐（追加 1 列后 dV+1 需满足对齐要求），L0B 空间占用增加需纳入 tiling 规划
- mm2 输出 dtype：rowsum 在 L0C 按 **fp32** 累加（对齐 MxFP4 `FixpipeMm2` COMPUTE_T=float），fixpipe 出来即 fp32，供 V2 全 fp32 链路使用

### 3. FixpipeMm2 搬出 rowsum

- `FixpipeMm2()`：fixpipe 时额外搬出 rowsum 列（mSize = dV+1 中第 dV+1 行/列），经跨核同步 buffer 交 Vector V2 阶段使用
- 替代原 Vector 计算的 `softmaxSumBuf_` 通路

### 4. 职责划分

- **sum 走 Cube**：mm2 顺带产出，Vector 不再计算 expSum
- **max 走 Vector**：max 更新与 exp 缩放逻辑保留在 Vector 侧不变

---

## 三、Block Vector 侧

新建 `op_kernel/arch35/quant_flash_attn_block_vec_mxfp8_dn.h`，参考主线 `quant_flash_attn_block_vec_mxfp8.h`。

### 1. fp16 softmax 统计量（核心改动）

- 类模板新增参数 `SOFTMAX_T`（默认 float，mode 3 实例化为 half）
- 五个统计 buffer 类型语义改为 SOFTMAX_T：
  - `softmaxMaxBuf_[PRELOAD_N+1]`、`softmaxSumBuf_[PRELOAD_N+1]`
  - `preLoopMaxBuf_`、`preLoopSumBuf_`、`firstLoopSumBuf_`
- 涉及函数：`ProcessVec1Dn()`（L334-338）、`ProcessVec1Vf()`（L546-554）、`ProcessVec2OnUb()`（L731/739/860）的 `Get<float>()` 改 `Get<SOFTMAX_T>()`
- `InitBuffer()`：5 个统计 buffer 256B 减半为 128B，元素个数 [64,1] 不变

### 2. 公共 vf API fp16 化

`attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h`：

- `ProcessVec1VfDnMxfp8()`（L1443）及内部 `ProcessVec1DnNoUpdateMxfp8()`（L1065）/ `ProcessVec1DnUpdateMxfp8()`（L1382）：
  - 模板增加统计精度参数，expSum/max 累加 `RegTensor<float>` 换 `RegTensor<half>`
  - mmRes fp32 输入 exp 前或 exp 后 Cast fp16；行内 reduce sum 与 max 比较按 fp16 执行
  - 跨轮更新 `sum = sum*exp(oldMax-newMax) + newSum`、`max = max(max, newMax)` 中间量全部 fp16
- 副本范围：全仓 3 份副本，仅修改 attention/common 副本（主线 mxfp8 实际 include 的），experimental 两份不动

### 3. rowsum 读取（配合 Cube 改动）

- `ProcessVec1Dn()`：删除 expSum 计算与回写逻辑；max 仍保留 Vector 侧计算
- `ProcessVec2OnUb()`：`LastDivNew` 除法所用 sumUb 改读 Cube 产出 buffer
- `SoftmaxDataCopyOut()`：sum 来源改为 Cube 产出

### 4. P 量化路径不变

- 输出 P 矩阵仍为 fp8_e4m3 + pScale，量化路径与高精一致

### 5. LSE 输出

- `SoftmaxLseCopyOut()` / `ComputeLogSumExpAndCopyToGm()`：LSE 计算 ln(sum)+max 时 **max 为 half 需 Cast float**，sum 为 Cube 产出 fp32 直接用
- softmaxLse 输出 dtype 保持 fp32，与高精方案一致

---

## Host 侧配套（简要）

- 主线 Host 侧：`QfaQuantMode` 枚举增 mode 3、`GetQuantMode()` 校验放行、`QfaQuantModeToSerialString` 补串、9 处 checker 兼容 mode 3
- experimental Host 侧：`quant_checker.cpp` 放行 mode 3 + layout 表追加 TND、新建 `quant_flash_attn_tiling_mxfp8_dn.{h,cpp}` tiling 类（IsCapable 仅认 mode 3，注意与现有 MxFP4 tiling 类 priority 路由二选一）
- tiling 策略：s2SplitSize 整块、统计 UB 按 half 规划、rowsum 中转 buffer 空间与同步 id 规划

## 风险点

1. 公共 vf API 为多算子共享文件，改模板签名前先 grep 全部调用方
2. half 分支实例化组合数需严格裁剪，控制编译产物膨胀
3. rowsum 跨核同步依赖不完整会导致 Vector 读到未就绪数据的读写竞争
4. fp16 累加精度：长序列下 sum 溢出 half 上限 65504 的风险，单测需覆盖大 s2 场景
