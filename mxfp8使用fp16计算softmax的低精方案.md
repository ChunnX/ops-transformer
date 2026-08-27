MxFP8需要支持低精度Softmax，即使用fp16计算softmax。主打PR场景，DN模板。
使用C1，V1，C2，V2正常轮次，区别于两轮高精度的两轮C1V1。
rowsum挪到Cube计算。

低精方案的MxFP8的quant_mode为3

代码位置与参考来源：
- **主体代码参考源**：主线 MxFP8 高精方案 `/home/w00947241/ops-transformer_fp16/attention/quant_flash_attn`，三个 mxfp8 kernel 文件（block_vec_mxfp8.h / block_cube_mxfp8.h / kernel_mxfp8.h）+ Host 侧 tiling_mxfp8.cpp
- **开发落地位置**：`/home/w00947241/ops-transformer_fp16/experimental/attention/quant_flash_attn`，在该 experimental 算子目录下新建 mxfp8 低精专属文件
- MxFP4 低精（experimental 同目录已实现，mode 5）仅作"低精三件套如何落地"的设计思路参考，不沿用其代码

高精指softmax的结果为fp32，低精的结果指的是softmax的结果为fp16。

**fp16 精确边界（对齐 MxFP4 mode 5 实际实现，已逐行核实）**：
- **half**：仅 V1 的 softmax max 统计量（MxFP4 的 `localGroupMaxUB`/`localGlobalMaxUB`/`softmaxMaxUB` 均 half）
- **fp32**：V2 全链路——Cube rowsum（L0C 累加 + fixpipe 均 float，`FixpipeMm2` COMPUTE_T=float）、rowsum 跨轮累积更新（`processUpdateVF` 全 float）、输出累积与除法（`processOutVF` 全 float）、最终 Cast bf16 输出
- 即"softmax 结果为 fp16"落在 **V1 统计量 max** 上，sum 走 Cube 后整条通路保持 fp32

---

# 修改点识别

主线 MxFP8 高精结构（已逐项核实）：
- `op_kernel/arch35/quant_flash_attn_kernel_mxfp8.h` L593-594 `c1v1Loop = CeilDiv(actSingleLoopS2Size, s2SplitSize)` 即两轮 C1V1 的根
- `op_kernel/arch35/quant_flash_attn_block_vec_mxfp8.h` L153-158 `softmaxMaxBuf_/SumBuf_/preLoopMaxBuf_/preLoopSumBuf_/firstLoopSumBuf_` 用 `Get<float>()`，L987-1001 InitBuffer 256B
- `op_kernel/arch35/quant_flash_attn_block_cube_mxfp8.h` 当前无 rowsum 通路，mm2 仅算 P·V
- Host 侧 `op_host/arch35/quant_flash_attn_tiling_mxfp8.cpp` L38-45 `IsCapable()` 只认 mode 1，L218 `GenTilingKey()` 已含 quantMode 维度，L435 注册 priority 1

低精三件套在主线 MxFP8 高精基础上的改造点：

## 一、Host 侧：mode 3 校验放行 + tiling 路由

### 主线 Host 侧（attention/quant_flash_attn）改动

1. `op_host/qfa_tiling_info.h:147` `QfaQuantMode` 枚举新增 `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16 = 3`。

2. `op_host/qfa_tiling_info_parser.cpp:268-277` `GetQuantMode()` 校验放行 quantModeVal==3，错误日志同步补 "or 3"。

3. `op_host/qfa_tiling_info.cpp:47-51` `QfaQuantModeToSerialString` 映射表补 mode 3 字符串。

4. `op_host/checkers/quant_checker.cpp` 9 处 `!= A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` 判断改为 `!= FP32 && != FP16`，参照 L616 已有 mode 6 双判断写法。涉及行：41、67、126、185、354、436、511、616、669。

5. `op_host/checkers/paged_attention_checker.cpp:114`、`op_host/checkers/mask_checker.cpp:178` 同样放行 mode 3。mode 3 限定 DN 模板，checker 需对 mode 3 + 非 DN 组合直接报错。

6. `op_api/quant_flash_attn.h:39` quant_mode 注释补 3=A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16。

### experimental Host 侧改动

7. `op_host/quant_flash_attn_tiling_info.h:93-105` quant_mode 常量**已包含 mode 3**，无需改动。

8. `op_host/checkers/quant_checker.cpp:41-51` `CheckSingleParaQuantMode` 的 `supportedQuantModes` 列表追加 `QUANT_MODE_A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16`，错误日志补 (3)。

9. `op_host/checkers/quant_checker.cpp:219-220` `QUANT_LAYOUT_CONSTRAINT_TABLE` 追加 mode 3 → TND 条目（与 mode 1 MxFP8 系列一致）。

10. `op_host/checkers/quant_checker.cpp:534-535` descale dtype 限制追加 mode 3 覆盖（mode 3 descale 也是 FLOAT8_E8M0）。

11. tiling 路由：新建 `op_host/arch35/quant_flash_attn_tiling_mxfp8_dn.{h,cpp}`，类 `QuantFlashAttnTilingMxfp8Dn` 仿照主线 `QuantFlashAttnTilingImpl` 结构。`IsCapable()` 仅当 `quantMode==3` 返回 true。注册 `REGISTER_TILING_TEMPLATE_FIA(QuantFlashAttn, QuantFlashAttnTilingMxfp8Dn, {DAV_3510}, priority)`，第 4 参为优先级（越小越先尝试）。**注意**：现有 `QuantFlashAttnTilingDn::IsCapable()` 无条件返回 true，路由需二选一：a) 新类注册更小优先级（如 209）抢先拦截 mode 3；b) 现有类 IsCapable 改为仅 mode 5 放行，新类注册 211。

12. `op_host/arch35/quant_flash_attn_tiling_mxfp8_dn.cpp` `SplitPolicy()` mode 3 专属：s2SplitSize 取整块 s2BaseSize 使 `c1v1Loop` 恒为 1（见第三节）；max 统计 UB 按 half 规划（仅 max，sum 统计已挪 Cube）；rowsum 中转 buffer 空间（fp32）与跨核同步 id 规划。`GenTilingKey()` 已含 quantMode 维度，无需改动结构，仅填充 mode 3 值。

## 二、Kernel 侧：fp16 softmax max + 单轮 C1V1 + rowsum 挪 Cube

参考主线 MxFP8 高精三个 mxfp8 文件，在 experimental `op_kernel/arch35/` 下新建对应 mxfp8 低精专属文件。

### Vector 块（参考 `attention/.../block_vec_mxfp8.h`）

新建 `op_kernel/arch35/quant_flash_attn_block_vec_mxfp8_dn.h`：

1. 类模板新增参数 `SOFTMAX_T`（仅作用于 max 统计量），默认 float，mode 3 实例化为 half。主线 L57 模板签名扩展。

2. L153-158 五个统计 buffer 拆分处理：`softmaxMaxBuf_[PRELOAD_N+1]`、`preLoopMaxBuf_` 两个 **max buffer** 改 `Get<SOFTMAX_T>()`；`softmaxSumBuf_[PRELOAD_N+1]`、`preLoopSumBuf_`、`firstLoopSumBuf_` 三个 **sum buffer 随 rowsum 挪 Cube 直接删除**（V2 改读 Cube 产出的 fp32 rowsum，不再需要 Vector 侧 sum 统计）。

3. `ProcessVec1Dn()` L318：L334-338 中 `maxUb`/`preLoopMaxUb` 相关 `Get<float>()` 改 `Get<SOFTMAX_T>()`；sum 相关引用随 buffer 删除一并清理。

4. `ProcessVec1Vf()` L546-554：仅 max 侧 `Get<float>()` 改型，sum 侧统计逻辑已删除。

5. `ProcessVec2OnUb()` L705：**保持 fp32 不动**。L731/L739 `sumUb` 改读 Cube 产出 fp32 rowsum buffer；L860 `softmaxMaxBuf_.Get<SOFTMAX_T>()[mStartVec]` 的 max 在使用处 Cast float。`FlashUpdateLastNew` 与 `LastDivNew` 的 rowsum 累积、除法、输出累加全链路 fp32（对齐 MxFP4 `processUpdateVF`/`processOutVF` 全 float 实现）。

6. `SoftmaxDataCopyOut()` L431：`maxUb` 入参参数化为 SOFTMAX_T；sum 来源改读 Cube 产出 fp32 buffer。

7. `SoftmaxLseCopyOut()` L467 / `ComputeLogSumExpAndCopyToGm()` L947：LSE 计算 ln(sum)+max 时 **max 为 half 需 Cast float**，sum 为 Cube 产出 fp32 直接用，softmaxLse 输出保持 fp32 不变。

8. `InitBuffer()` L987-1001：仅 2 个 max 统计 buffer 256B 减半为 128B，元素个数 [64,1] 不变；3 个 sum buffer 空间释放。

### 公共 Vector API（主线 `attention/common/.../vf_mul_sel_softmaxflashv2_cast_nz_dn.h`）

9. `ProcessVec1VfDnMxfp8()` L1443 及内部 `ProcessVec1DnNoUpdateMxfp8()` L1065 / `ProcessVec1DnUpdateMxfp8()` L1382：
   - 模板增加 max 统计精度参数；**仅 max 累加与跨轮 max 更新** `RegTensor<float>` 换 `RegTensor<half>`（对齐 MxFP4 `localGroupMaxUB` 为 half）
   - **expSum 统计整体删除**（rowsum 挪 Cube，Vector 不再算 sum）
   - mmRes fp32 输入 exp 计算路径不变；max 行内比较按 half，exp 缩放 `exp(x - max)` 中 max 为 half 参与时按需 Cast
   - 跨轮更新只剩 `max = max(max, newMax)`（half）；`sum = sum*exp(oldMax-newMax) + newSum` 逻辑随 rowsum 挪 Cube 删除
   - 输出 P 矩阵仍为 fp8_e4m3 + pScale，量化路径不变
   - **副本范围**：全仓 3 份副本，仅修改 attention/common 副本（L1443，主线 mxfp8 实际 include 的），experimental 两份不动

### Cube 块（参考 `attention/.../block_cube_mxfp8.h`）

新建 `op_kernel/arch35/quant_flash_attn_block_cube_mxfp8_dn.h`：

10. `IterateBmm2Dn()` L853：B 侧 V 矩阵追加全 1 列构造 [V | ones]，mm2 一次算出 [P·V | P·ones]，L0C 额外得到每行 rowsum。

11. L1 buffer 管理处：L1 预置全 1 常量矩阵，随 V 一起 Load，仅初始化一次；注意 N 维对齐与 L0B 空间占用增加。

12. `FixpipeMm2()` L832：fixpipe 额外搬出 rowsum 列到 UB/GM，经跨核同步 buffer 交 Vector V2 阶段使用。

13. mm2 输出 dtype：rowsum 在 L0C 按 **fp32** 累加（对齐 MxFP4 `FixpipeMm2` COMPUTE_T=float），fixpipe 出来即 fp32，与 V2 全 fp32 链路对齐。

### Kernel 主控（参考 `attention/.../kernel_mxfp8.h`）

新建 `op_kernel/arch35/quant_flash_attn_kernel_mxfp8_dn.h`：

14. L63 `PRELOAD_N = 2` 软件流水结构沿用，C1→C1→C1→C2 即正常轮次。

15. L593-594 `c1v1Loop = CeilDiv(actSingleLoopS2Size, s2SplitSize)` 循环：mode 3 时 s2SplitSize 取整块 s2BaseSize 使 c1v1Loop 恒为 1，消除两轮 C1V1。建议模板编译期裁剪 subLoop 分支而非运行时判断。

16. `CROSS_CORE_SYNC_*` 同步 ID 补 rowsum 通路依赖（Vector V2 需等 Cube fixpipe 搬出 rowsum 完成）。

### Vector/Cube 配合改动

17. `ProcessVec1Dn()` L318：删除 expSum 计算与回写逻辑；max 仍保留 Vector 侧计算，max 更新与 exp 缩放不变。职责固定：sum 走 Cube，max 走 Vector。

18. `ProcessVec2OnUb()` L705 `LastDivNew` 的 sumUb 改读 Cube 产出 buffer。

19. `SoftmaxDataCopyOut()` L431 sum 来源改为 Cube 产出。

## 三、轮次：单轮 C1V1

1. 主线 `kernel_mxfp8.h` L593 `c1v1Loop = CeilDiv(actSingleLoopS2Size, s2SplitSize)` 是两轮 C1V1 的根。低精方案 tiling 侧 `s2SplitSize = s2BaseSize` 整块，使 `c1v1Loop` 恒为 1。

2. `block_vec_mxfp8.h` s2Split 相关分支（L85 定义、L236-244/L348/L394/L421-422/L573/L666/L674/L695-696/L1085-1086/L1123-1124）：mode 3 下 `actSingleLoopS2Size > s2SplitSize` 恒 false，subLoop 两轮 C1V1 退化为单轮。建议模板编译期裁剪分支。

3. `block_cube_mxfp8.h` `IterateBmm1Dn()` L677：subLoop 循环恒为 1，与上对齐。

## 四、实例化与验证

1. experimental `op_kernel/quant_flash_attn.cpp` 新增 mxfp8 低精实例化分支。参照主线 L65-69 的 `QuantFlashAttnBlockCubeMxfp8/BlockVecMxfp8/KernelMxfp8` 实例化，模板第二参数统计精度类型 float 处按 `isSoftmaxFp16` 分支实例化 half 组合；仅 DN+PR 路径实例化，控制产物膨胀。

2. experimental `op_host/arch35/quant_flash_attn_tiling_mxfp8_dn.cpp` `GenTilingKey()` tiling key 区分 softmax 精度维度，与实例化对齐。

3. `torch_extension/quant_flash_attn.py:25` 已定义 mode 3 枚举，无需修改；补 mode 3 单测与精度对拍，golden 对比 fp16 softmax 数值容差单独设定。

## 已确认结论

1. **主体代码参考源**：主线 MxFP8 高精方案三个 mxfp8 kernel 文件 + Host 侧 tiling_mxfp8.cpp，不沿用 MxFP4 任何代码。MxFP4 仅作低精三件套设计思路参考。
2. **两轮 C1V1** 即 DN 路径 s2Split 分裂出的 subLoop 0/1 两遍 C1V1，低精方案按单轮 C1V1 处理。
3. **rowsum 挪 Cube 后**，max 仍留 Vector 侧计算。sum 走 Cube、max 走 Vector，职责固定。
4. **mode 3 严格限定 DN**，不支持 ND 模板路径。checker 需对 mode 3 + 非 DN 组合直接报错。
5. **fp16 边界（对齐 MxFP4 实际实现）**：half 仅作用于 V1 的 max 统计量；rowsum 走 Cube 后 L0C 累加、fixpipe、V2 累积更新、除法、输出累加全链路 fp32。Vector 侧 sum 统计 buffer（softmaxSumBuf_/preLoopSumBuf_/firstLoopSumBuf_）整体删除。
6. **LSE 需支持**，输出保持 fp32。max 为 half 时先 Cast float，sum 为 Cube 产出 fp32 直接用，计算 ln(sum)+max，softmaxLse 输出 dtype 与高精一致。
7. **vf API 副本**：仅修改 attention/common 副本（L1443），experimental 两份不动。

## 风险点

1. 公共 vf API 为多算子共享文件，改模板签名前先 grep 全部调用方，避免影响共享此 API 的其他算子。
2. half 分支实例化组合数需严格裁剪，否则编译产物显著膨胀。
3. rowsum 跨核同步事件若依赖不完整，会出现 Vector 读到未就绪 rowsum 的读写竞争。
4. max 统计量 half 舍入精度：max 用 half 表示（10 bit 尾数）引入的舍入误差会传导到 exp 缩放与 P 量化，需在精度对拍中重点验证大 scale 值域场景。
