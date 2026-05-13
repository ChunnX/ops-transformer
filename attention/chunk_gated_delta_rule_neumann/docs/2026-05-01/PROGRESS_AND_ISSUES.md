# 2026-05-01 进度与问题

## 已完成

- 确认当前默认实现为 `C64_1616_FP32_CVInverse` raw MMAD fused inverse 路线。
- 确认 `RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`，raw 路径已打开。
- 确认 latest raw 调整已在代码中：
  - `SetNdParaImpl(0x1)`
  - `LoadDataWithTranspose(...)`
- 确认最新双算子包已构建完成，产物位于 `build_out/cann-ops-transformer-custom_linux-aarch64.run`。
- 修正 raw MMAD A/B 输入 L1 布局：
  - A 侧 `CopyGmMatrixToL1A()`：RowMajor -> `zN`。
  - B 侧 `CopyGmMatrixToL1B()`：RowMajor -> `zZ`，避免 B 侧继续复用 A 侧布局。
- 修正版双算子包构建通过，最新产物时间为 2026-05-01 18:18。
- 晚间已可安装 `.run` 并运行最小 NPU smoke，环境阻塞解除。
- 完成 raw readback 二分：
  - raw 写 `outQkGm_` 可明显影响 out。
  - raw 写 `tmpWsGm_` 后可被 `MatmulImpl` 读到并写回 `outQkGm_`。
  - raw 写 `tmpWsGm_` 再 readback 到 `INVWsGm_` 仍回到 no-op 失败特征。
  - 16x16 readback 改用 `mmFp32C16` 后结果不变。

## 进行中

- `RawMmadInvAdd16()` 的真实 `INV += INV @ P` 链路仍在调试中。
- 当前验证点已收敛到 raw MMAD 写出矩阵的数值/布局是否符合后续 `INV` 消费语义。
- 当前代码仍为 debug 实验态：`RAW_MMAD_DEBUG_STEP=6`、`RAW_MMAD_DEBUG_TMP_READBACK=true`、`RAW_MMAD_DEBUG_READBACK_TO_QK=false`。

## 阻塞项

- 最小 smoke 可运行，但 Neumann 仍数值失败：
  - normal out/state PASS，out `max_abs=8.133054e-04`，state `max_abs=4.790872e-04`。
  - Neumann 当前 debug 态 out `max_abs=7.104934e-01`，state `max_abs=5.422900e-01`。
- 当前不能进入全量 `test_run.sh compare` 或性能 profiling，应继续最小 deterministic debug。

## 风险

- raw MMAD 参数错误可能表现为可返回但结果等价 no-op，不能仅凭 kernel 不挂判断正确。
- raw const MMAD 写出的布局可能与 `MatmulImpl`/后续 `INV @ rhs` 预期的 ND row-major 不一致，导致写回 `INVWsGm_` 后表现为无效 inverse。
- 当前 debug 常量不是生产态，结束前必须明确记录或恢复。

## 进度快照

- 阶段定位：已从 MatmulImpl 串联优化推进到 raw MMAD fused inverse 链路调试。
- 代码状态：`C64_1616_FP32_CVInverse` raw 路径为默认实现，最新 A/B L1 布局修正已进入构建产物。
- 验证状态：build-pass，但 18:18 包未安装，未完成 precision smoke。
- 阻塞条件：当前环境无法访问 NPU 设备，安装与设备 smoke 需要用户允许写入 Ascend OPP/vendor 并访问设备。
- 下一动作：安装最新 `.run` 后跑最小 smoke；失败时增加 `RawMmadInvAdd16()` 可观测 debug 输出。

## 本次状态核对补充

- 默认实现：仍为 `C64_1616_FP32_CVInverse`，selector 和 host `chunkSize=64` 已复核。
- raw 状态：`RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`，当前默认执行 raw MMAD 路径。
- 构建状态：晚间多轮 debug 包均构建通过，最新包已安装并完成最小 smoke。
- 验证状态：新增设备精度结果确认 raw 路径仍失败，但已排除安装/selector、stage workspace 可见性、micro tiling 三类问题。
- 下一动作：构造更直接的 raw MMAD 输出布局/数值验证，优先检查 `Fixpipe` `unitFlag/srcStride/dstStride` 与 L0C row-major 解释是否匹配。

## Final Matmul C64 Tiling 修复后状态

### 已完成

- 新增并初始化 `stage1MT_C64_`，接入 host 侧 `matmulTilingFp32C64`。
- 将 `C64_1616_FP32_CVInverse` final 两次 `64 x D` matmul 改为使用 `mmFp32C64`：
  - `INVWsGm_ @ gBKWsGm_ -> outKCumdecayGm_`
  - `INVWsGm_ @ vBetaWsGm_ -> outVInnerGm_`
- 探针确认 final A 侧可见：`RAW_MMAD_DEBUG_FINAL_USE_KK=true` 时 out/state 明显变化，说明 `INVWsGm_`/`kkWsGm_` 切换已被 final matmul 读取。
- raw full path 已切回生产态调试配置：
  - `RAW_MMAD_DEBUG_STEP=0`
  - `RAW_MMAD_DEBUG_OVERWRITE_PRODUCT=false`
  - `RAW_MMAD_DEBUG_FINAL_USE_KK=false`
- 最小 deterministic smoke 已通过：`B=1,S=16,nk=1,nv=1,dk=16,dv=16,seed=123,CGDR_GOLDEN_CHUNK_SIZE=64,atol=rtol=0.08`。

### 进行中

- 只扩大小范围 smoke 覆盖，继续验证 raw `Mmad/Fixpipe -> INVWsGm_ -> final C64 MatmulImpl` 链路。
- 暂不运行 full compare/perf，避免偏离当前 final matmul 与 raw inverse 消费链路调试目标。

### 风险

- 尚未验证 tail、`dk/dv=32/128`、GQA、多 batch。
- 当前 small-smoke 阈值为 `0.08`，可证明链路连通和粗精度可接受，但不能替代全量三方精度验证。

## Raw full path tail smoke 最新状态

### 已完成

- 维持默认 selector 为 `C64_1616_FP32_CVInverse`，host `chunkSize=64`。
- raw full path 当前配置为 `RAW_MMAD_DEBUG_STEP=0`、`RAW_MMAD_USE_MATMULIMPL_FALLBACK=false`、`RAW_MMAD_DEBUG_FINAL_USE_KK=false`。
- 已完成小范围 smoke：
  - `S=16,dk=dv=16` raw full path PASS。
  - `S=16,dk=dv=32` raw full path PASS。
  - `S=17,dk=dv=16` raw full path FAIL，out `max_abs=1.203164e-01`，state PASS。
  - `S=17,dk=dv=16` 切 `MatmulImpl` fallback 后 PASS，且 Neumann 与 normal 输出一致。
- cache flush 与禁用 HF32 均未改善 S17 raw 失败。
- 曾尝试把 `CopyL1MatrixToL0B()` 改为 N 方向 C0 循环，但触发 `507015 ADDR_MISALIGN`，已撤回。

### 进行中

- 继续只排查 raw `Mmad/Fixpipe -> INVWsGm_ -> final INV@rhs` 链路。
- 当前最小失败 case 固定为 `B=1,S=17,nk=1,nv=1,dk=16,dv=16,seed=125,CGDR_GOLDEN_CHUNK_SIZE=64,atol=rtol=0.08`。
- 下一步优先二分 raw B 侧布局与 raw 输出布局，不进入 Stage2/Stage3，也不跑性能。

### 阻塞项

- `S=17` tail case raw full path 精度仍失败，而 fallback PASS，说明 raw MMAD/Fixpipe 链路存在 tail 相关布局或累加语义问题。

### 风险

- S16 PASS 只能证明完整 16x16 子块连通；S17 会启用第二个 16x16 对角块和 32 compose，raw 链路可能在有效长度跨子块后暴露布局错误。
- B 侧 L1/L0B 直接改循环容易触发地址对齐错误，后续实验应采用可观测 GM debug 或严格参考 CANN/FIA 现有 raw MMAD 搬运方式。

## S17 对角块二分结果

### 已完成

- 在 `S=17,dk=dv=16,seed=125` 上完成对角块 fallback 二分：
  - 仅 `base=16` fallback：仍 FAIL，out `max_abs=1.203164e-01`，失败特征不变。
  - 仅 `base=0` fallback：PASS，out/state `max_abs=6.067441e-02/9.724498e-04`。

### 进行中

- 定位已收敛到 `base=0` raw 16x16 inverse 的 raw `Mmad/Fixpipe atomic add` 结果。
- 下一步直接对比 raw product/atomic add 与 `MatmulImpl` fallback 的数值或布局差异。

### 阻塞项

- raw base0 在 S16 单块场景可过阈值，但在 S17 的 32 compose 场景会放大误差；尚未确认是 raw product 本身数值偏差、Fixpipe row-major 解释问题，还是 atomic add 语义问题。

## Raw L1->L0 布局修复最新状态

### 已完成

- 验证 `RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=true` 探针：非 atomic 写 `tmpWsGm_` 再 AIC 标量累加仍未修复 S17，排除 `Fixpipe atomic add` 为主因。
- 将 `CopyL1MatrixToL0A()` / `CopyL1MatrixToL0B()` 改为 `LoadData3DParamsV2<float>` 路径，按仓库通用 raw FP32 matmul helper 对齐 L0A/L0B 布局。
- 关闭临时标量累加探针，当前默认配置为 raw full path：
  - `RAW_MMAD_DEBUG_STEP=0`
  - `RAW_MMAD_USE_MATMULIMPL_FALLBACK=false`
  - `RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=false`
  - `RAW_MMAD_DEBUG_FINAL_USE_KK=false`
- 构建与安装通过。
- 小范围 smoke 全部 PASS：`S=16/17/33/49/64,dk=dv=16` 以及 `S=16,dk=dv=32`。

### 进行中

- 当前阶段仍只验证 `C64_1616_FP32_CVInverse` 的 raw `Mmad/Fixpipe -> INVWsGm_ -> final INV@rhs` 链路。
- 尚未运行 full compare，也未开始性能 profiling。

### 阻塞项

- 本轮 S17 阻塞已解除；当前没有新的设备或构建阻塞。
- 仍需后续用 full compare 覆盖更多 batch/GQA/维度组合，才能把该 raw 路径视为完整精度闭环。

### 风险

- 小范围 smoke 使用 `atol=rtol=0.08`，证明 tail/compose 主路径已修复，但不能替代全量三方精度。
- 当前文件仍保留若干 debug 常量开关，虽然默认关闭；后续收尾前应清理或明确保留原因。
