# 2026-05-01 工作日志

## Raw MMAD fused inverse 状态核对

- 核对当前 selector：`op_kernel/chunk_gated_delta_rule_neumann_stage1.h` 当前包含 `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`。
- 核对 raw 路径开关：`RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`，说明当前默认走 raw MMAD 路径，而不是 MatmulImpl fallback。
- 核对最新已存在代码变更：
  - `InitAicLocalBuffers()` 中已有 `SetNdParaImpl(0x1)`。
  - `CopyL1MatrixToL0B()` 中已使用 `LoadDataWithTranspose(...)`。
- 核对构建产物：`/mnt/workspace/gitCode/cann/ops-transformer/build_out/cann-ops-transformer-custom_linux-aarch64.run` 已存在，时间为 2026-05-01 17:37。
- 尝试安装最新构建包用于精度验证，但安装命令需要写入 Ascend OPP vendor 目录，审批被拒绝，因此本轮未安装、未运行最新 smoke。

## 当前结论

- 最新代码处于“已构建、未安装、未验证”状态。
- 上一个已知实测结果仍是 raw 路径精度失败，表现接近 no-op：最小 case 中 CPU vs Neumann `out max_abs≈7.10e-01`，`state max_abs≈5.42e-01`。
- 下一步必须先安装最新 `.run` 包并运行最小精度 smoke，才能判断 `SetNdParaImpl(0x1)` 与 `LoadDataWithTranspose(...)` 是否修复 raw MMAD 链路。

## 下一步建议

1. 安装最新包：`bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet`。
2. 运行最小 smoke：`CGDR_GOLDEN_CHUNK_SIZE=64 timeout 120s python3 compare_chunk_gdr_three_way.py --batch-size 1 --seqlen 16 --nk 1 --nv 1 --dk 16 --dv 16 --seed 123 --atol 0.08 --rtol 0.08`。
3. 若仍保持 no-op 级别误差，不再继续盲调 L0B 参数；应增加 `RawMmadInvAdd16()` debug 模式，将 raw `INV @ P` 写到 GM workspace，与 MatmulImpl fallback 输出对比。

## Raw MMAD L1 布局修正

- 安装 17:37 构建包后尝试运行最小 smoke：
  - 第一次因 `torch_extensions` 默认写 `/home/developer/.cache` 被只读文件系统拦截。
  - 改用 `TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_py312_cpu` 后，PyTorch extension 编译通过。
  - 进入 NPU 初始化时 `aclInit` 失败：`507899 / drvGetDevNum failed`，说明当前执行环境无法访问 NPU 设备。
  - 申请沙箱外运行同一 smoke 命令被拒绝，因此没有得到最新 precision 结果。
- 对照 CANN template_linear_algebra 与 FIA raw MMAD 写法，修正 `RawMmadInvAdd16()` 输入搬运：
  - 将原 `CopyGmMatrixToL1()` 拆为 `CopyGmMatrixToL1A()` 与 `CopyGmMatrixToL1B()`。
  - A 侧保持 RowMajor -> `zN`，供 `CopyL1MatrixToL0A()` 加载到 L0A。
  - B 侧改为 RowMajor -> `zZ`，供 FP32 `LoadDataWithTranspose()` 加载到 L0B；关键参数 `dstNzC0Stride = BLOCK_CUBE`，`dstNzMatrixStride = AlignUp(cols, RAW_FLOAT_C0) * BLOCK_CUBE`。
- 重新构建双算子包通过：
  - 命令：`bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 产物：`/mnt/workspace/gitCode/cann/ops-transformer/build_out/cann-ops-transformer-custom_linux-aarch64.run`
  - 时间：2026-05-01 18:18
- 尝试安装 18:18 修正版 `.run` 包时审批被拒绝，因此此修正仍处于 build-pass、未安装、未精度验证状态。

## 当前进度总结

- 当前默认实现仍为 `C64_1616_FP32_CVInverse` raw MMAD fused inverse 路线，selector 与代码开关均已核对。
- `RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`，说明默认不再走 MatmulImpl fallback。
- 最新 raw 链路已完成两处关键修正：A/B 侧 GM->L1 布局拆分，以及 B 侧 `zZ + LoadDataWithTranspose` 加载路径。
- 最新双算子包构建通过，产物为 `build_out/cann-ops-transformer-custom_linux-aarch64.run`，时间为 2026-05-01 18:18。
- 当前真实阻塞仍是设备验证：18:18 包尚未安装，沙箱内 NPU 初始化失败，精度未确认。
- 下一步应先安装 18:18 包并跑最小 `B=1,S=16,NK=NV=1,DK=DV=16` smoke；若仍失败，给 `RawMmadInvAdd16()` 加 debug 输出，对比 raw `INV @ P` 与 MatmulImpl fallback。

## 任务状态核对

- 当前分支为 `chunk-gdr-neumann-20260429-upload`，本地相对 `fork/chunk-gdr-neumann-20260429-upload` 领先 4 个提交，工作区存在未提交改动和 profiling 输出清理/新增目录。
- 当前默认 selector 确认为 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`；host tiling 当前 `chunkSize=64`，对应 C64 16x16 子块 raw MMAD fused inverse 调试方向。
- 当前 raw 路径开关为 `RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`，因此默认不走 MatmulImpl fallback。
- 关键 raw 调整已在代码中：`SetNdParaImpl(0x1)`、A/B 侧 GM->L1 拆分、B 侧 `zZ + LoadDataWithTranspose(...)`。
- 最新构建产物仍是 `build_out/cann-ops-transformer-custom_linux-aarch64.run`，文件时间为 2026-05-01 18:18，构建状态为 build-pass。
- 验证边界未变化：最新 18:18 包尚未完成设备侧安装和最小 precision smoke；当前可信结论仍是“已构建、未安装、未精度验证”。
- 下一步仍应先安装 18:18 包并运行最小 smoke；若 raw 结果仍接近 no-op，应增加 `RawMmadInvAdd16()` debug 输出，与 MatmulImpl fallback 的 `INV @ P` 结果逐项对比。

## Raw MMAD readback 最小实测

- 安装并运行 2026-05-01 晚间重新构建包，当前环境可完成 `.run --quiet` 安装和最小 NPU smoke，不再是“未安装/无法访问设备”状态。
- 固定最小 case：
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
  - `B=1,S=16,nk=1,nv=1,dk=16,dv=16,seed=123`
  - 阈值 `atol=0.08, rtol=0.08`
- baseline normal `ChunkGDR` 持续 PASS：
  - out `max_abs=8.133054e-04`
  - state `max_abs=4.790872e-04`
- 实验 1：`RAW_MMAD_DEBUG_STEP=6`，raw const L0A/L0B MMAD 直接写 `outQkGm_`，再 `MatmulImpl(outQkGm_, powerGm -> INVWsGm_)`。
  - Neumann out 明显变化，out `max_abs=8.187056e+00`，state 仍为旧 no-op 特征 `max_abs=5.422900e-01`。
  - 结论：raw Fixpipe 写 GM 能影响可观测输出链路，不是 selector/安装/分支问题。
- 实验 2：raw const L0A/L0B MMAD 写 `tmpWsGm_`，再 `MatmulImpl(tmpWsGm_, powerGm -> outQkGm_)`。
  - Neumann out 变化为 `max_abs=4.229003e-01`，state 仍 `max_abs=5.422900e-01`。
  - 结论：`MatmulImpl` 能读到 raw 写过的 stage workspace；“stage workspace raw 写完全不可见”假设被排除。
- 实验 3：同实验 2，但 readback 输出改回 `INVWsGm_`。
  - Neumann 回到 no-op 失败特征：out `max_abs=7.104934e-01`，state `max_abs=5.422900e-01`。
  - 结论：更新 `INVWsGm_` 后没有得到有效 inverse 语义，问题更可能是 raw 写出矩阵的数值/布局不符合后续 `INV` 消费，而不是 GM 可见性。
- 实验 4：将 16x16 readback 与 `AICBlockProcess` 改用 `mmFp32C16/mmFp32C32` micro tiling。
  - 最小 smoke 结果不变：out `max_abs=7.104934e-01`，state `max_abs=5.422900e-01`。
  - 结论：不是 16x16 子块误用通用 64x64 `mmFp32` tiling 导致。

## 当前代码调试态

- `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h` 当前仍处于 debug 实验态：
  - `RAW_MMAD_DEBUG_STEP = 6`
  - `RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`
  - `RAW_MMAD_DEBUG_WRITE_QK = false`
  - `RAW_MMAD_DEBUG_MATMUL_READBACK = true`
  - `RAW_MMAD_DEBUG_TMP_READBACK = true`
  - `RAW_MMAD_DEBUG_READBACK_TO_QK = false`
- 已加入 `AICBlockProcess()` 按 blockSize 选择 `mmFp32C16/mmFp32C32` 的修正；该修正不改变当前失败特征，但应保留为 C64_1616 micro tiling 路径一致性修正。
- 下一步应直接验证 raw const MMAD 写出的 16x16 矩阵布局/数值，可继续用可观测 `outQkGm_` 或增加更小的 deterministic matrix debug，而不是继续从 cache/tiling 可见性方向盲调。

## Final Matmul C64 Tiling 修复与 raw full path smoke

- 继续沿 final `INVWsGm_` 消费链排查，没有修改 Stage2/Stage3，也没有运行 full compare/perf。
- 固定最小 case 继续使用：
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
  - `B=1,S=16,nk=1,nv=1,dk=16,dv=16,seed=123`
  - 阈值 `atol=0.08, rtol=0.08`
- 复测交接状态：`RAW_MMAD_DEBUG_STEP=4`、`RAW_MMAD_DEBUG_FINAL_USE_KK=true`，只将 final 左矩阵从 `INVWsGm_` 替换为 `kkWsGm_`。
  - 使用通用 `mmFp32` 时结果仍是旧失败特征：out/state `max_abs=7.104934e-01/5.422900e-01`。
  - 结论：final `64 x D` matmul 使用 128 基础 tiling 时，A 侧切换没有可靠反映到输出。
- 给 Stage1 增加 `stage1MT_C64_`，使用 host 已生成的 `matmulTilingFp32C64`，并仅将 final 两次 `64 x D` matmul (`INV@gBK`、`INV@vBeta`) 改为 `AICProcessWith(..., mmFp32C64)`。
- 探针验证：保留 `RAW_MMAD_DEBUG_FINAL_USE_KK=true` 后，Neumann out/state 明显变化为 `1.177194e+00/1.921196e+00`，证明 final A 侧改为 `kkWsGm_` 已真正生效。
- 将 `RAW_MMAD_DEBUG_FINAL_USE_KK=false` 切回 `INVWsGm_`，保留 C64 final tiling，`RAW_MMAD_DEBUG_STEP=4` no-op inverse 下最小 smoke PASS：
  - CPU vs Neumann out/state `max_abs=5.903786e-02/1.096180e-02`
  - ChunkGDR vs Neumann out/state `max_abs=5.859375e-02/1.098633e-02`
- 将 raw MMAD 真实路径打开：`RAW_MMAD_DEBUG_STEP=0`、`RAW_MMAD_DEBUG_OVERWRITE_PRODUCT=false`，即执行 `INV += INV @ Lpow` 的 raw `Mmad/Fixpipe atomic add` 链路。
- 重新构建、安装并运行同一最小 smoke，结果 PASS：
  - CPU vs ChunkGDR out/state `max_abs=8.133054e-04/4.790872e-04`
  - CPU vs ChunkGDRNeumann out/state `max_abs=5.903786e-02/1.096180e-02`
  - ChunkGDR vs ChunkGDRNeumann out/state `max_abs=5.859375e-02/1.098633e-02`
- 当前关键结论：
  - `INVWsGm_` 写入/读回链路不是根因；final `INV@rhs` 使用不匹配的通用 `mmFp32` tiling 才是前面“换 A 不生效/像 no-op”的直接原因。
  - raw `Mmad/Fixpipe -> INVWsGm_ -> final C64 MatmulImpl` 在最小 `S=16,D=16` case 已连通并通过精度阈值。
  - 尚未运行全量 `test_run.sh compare`，也未跑性能 profiling；后续应先做小范围 tail/维度 smoke，再决定是否全量 compare。

## Raw full path tail smoke 与 fallback 对照

- 继续只沿 `C64_1616_FP32_CVInverse` 的 raw `Mmad/Fixpipe -> INVWsGm_ -> final INV@rhs` 链路排查；未修改 Stage2/Stage3，未运行 full compare/perf。
- 当前默认调试配置：
  - `RAW_MMAD_DEBUG_STEP = 0`
  - `RAW_MMAD_USE_MATMULIMPL_FALLBACK = false`
  - `RAW_MMAD_DEBUG_OVERWRITE_PRODUCT = false`
  - `RAW_MMAD_DEBUG_FINAL_USE_KK = false`
  - raw `Mmad` 已显式 `SetHF32Mode(HF32Mode::DISABLE)`，该设置不改变失败特征。
- 小范围 smoke 结果：
  - `S=16,dk=dv=16` raw full path PASS：CPU vs Neumann out/state `max_abs=5.903786e-02/1.096180e-02`。
  - `S=16,dk=dv=32` raw full path PASS：CPU vs Neumann out/state `max_abs=4.757836e-02/4.254328e-02`。
  - `S=17,dk=dv=16` raw full path FAIL：CPU vs Neumann out `max_abs=1.203164e-01`，state 仍 PASS。
  - 同一 `S=17,dk=dv=16` 切 `RAW_MMAD_USE_MATMULIMPL_FALLBACK=true` 后 PASS，且 Neumann 与 normal 输出一致。
- 已排除项：
  - raw atomic 写回后补 `FlushInvBlockCacheLines(invOffset, INVERSE_SHAPE)`，S17 失败无改善。
  - 禁用 HF32 后 S17 失败无改善。
  - 直接将 `CopyL1MatrixToL0B()` 改为 N 方向 C0 循环会触发 `507015 ADDR_MISALIGN`，该改动已撤回。
- 当前判断：
  - final `64 x D` matmul、16x16 compose、`INVWsGm_` 被 final 消费链路基本排除；fallback 同 case PASS 证明 MatmulImpl 串联语义正确。
  - 疑点集中在 raw `GM->L1->L0B->Mmad->Fixpipe atomic add` 的 B 侧布局、raw 输出布局或 atomic add 语义。

## S17 对角块 fallback 二分

- 继续固定最小失败 case：`B=1,S=17,nk=1,nv=1,dk=16,dv=16,seed=125,CGDR_GOLDEN_CHUNK_SIZE=64,atol=rtol=0.08`。
- 实验 1：仅让 `base=16` 的第二个 16x16 对角 inverse 走 `MatmulImpl` fallback，`base=0` 保持 raw full path。
  - 结果仍 FAIL，失败特征不变：CPU vs Neumann out `max_abs=1.203164e-01`，state PASS。
  - 结论：第二个 16x16 对角块不是主因。
- 实验 2：仅让 `base=0` 的第一个 16x16 对角 inverse 走 `MatmulImpl` fallback，`base=16` 保持 raw full path。
  - 结果 PASS：CPU vs Neumann out/state `max_abs=6.067441e-02/9.724498e-04`；ChunkGDR vs Neumann out/state `max_abs=6.054688e-02/0`。
  - 结论：S17 raw 失败来自第一个 16x16 raw inverse 的数值/布局误差；该误差在 S16 单块时仍低于阈值，但在 S17 触发 `Compose32From16()` 后被下三角 compose 放大。
- 当前二分结论：
  - `Compose32From16()` 和 final C64 matmul 在 fallback 组合下可工作。
  - 下一步应直接对比 `base=0` raw `INV += INV@P` 与 MatmulImpl fallback 的写回结果，重点检查 raw product/atomic add 的数值和 layout，而不是继续怀疑尾块或 final 消费链路。

## Raw FP32 L1->L0 布局修复与小范围闭环

- 接续交接点，先验证临时探针 `RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=true`：raw `Mmad` product 先非 atomic `Fixpipe` 到 `tmpWsGm_`，再由 AIC 标量循环累加到 `INVWsGm_`。
  - `S=17,dk=dv=16,seed=125` 仍 FAIL，out `max_abs=1.203164e-01`，state PASS。
  - 结论：问题不是 `Fixpipe atomic add` 本身；即使绕过 atomic add，raw product 或其 L0/写回布局仍会导致 base0 inverse 误差。
- 对齐仓库通用 raw FP32 matmul helper，将 `CopyL1MatrixToL0A()` 与 `CopyL1MatrixToL0B()` 从手写 `LoadData2d/LoadDataWithTranspose` 参数改为 `LoadData3DParamsV2<float>`：
  - A 侧：`enTranspose=0`，`channelSize=blockSize`，按 `MK` 语义载入 L0A。
  - B 侧：`enTranspose=1`，`channelSize=blockSize`，按 `KN` 语义载入 L0B。
  - 新增 `RAW_LOAD3DV2_CONFIG={true,true}`，保留 `SetLoadDataPaddingValue<uint64_t>(0)` 与 `SetNdParaImpl(0x1)`。
- 将临时探针关闭：`RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT=false`；当前默认仍是 raw full path：
  - `RAW_MMAD_DEBUG_STEP=0`
  - `RAW_MMAD_USE_MATMULIMPL_FALLBACK=false`
  - `RAW_MMAD_DEBUG_OVERWRITE_PRODUCT=false`
  - `RAW_MMAD_DEBUG_FINAL_USE_KK=false`
- 构建与安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16` 通过。
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet` 安装通过。
- 小范围 smoke 结果全部 PASS，均使用 `CGDR_GOLDEN_CHUNK_SIZE=64`、`atol=rtol=0.08`：
  - `S=16,dk=dv=16,seed=123`：CPU vs Neumann out/state `5.903786e-02/4.790872e-04`。
  - `S=17,dk=dv=16,seed=125`：CPU vs Neumann out/state `6.067441e-02/9.724498e-04`，修复前 out 为 `1.203164e-01` FAIL。
  - `S=16,dk=dv=32,seed=126`：CPU vs Neumann out/state `4.002171e-02/4.877895e-04`。
  - `S=33,dk=dv=16,seed=127`：CPU vs Neumann out/state `5.728060e-02/4.730672e-04`。
  - `S=49,dk=dv=16,seed=128`：CPU vs Neumann out/state `9.144460e-02/9.301305e-04`。
  - `S=64,dk=dv=16,seed=129`：CPU vs Neumann out/state `7.720757e-02/8.848906e-04`。
- 当前结论：
  - S17 根因确认是 raw FP32 `L1->L0A/L0B` 布局参数不匹配，尤其在 `base=0` inverse 经过 `Compose32From16()` 后被放大。
  - `Fixpipe atomic add`、final `INV@rhs`、`Compose32From16()`/`Compose64From32()` 的 MatmulImpl 串联语义不是本轮根因。
  - 当前只完成小范围 raw inverse 链路 smoke，尚未运行 full compare 或性能 profiling。
