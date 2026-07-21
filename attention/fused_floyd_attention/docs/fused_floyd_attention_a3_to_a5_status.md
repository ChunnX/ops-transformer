# FusedFloydAttention A3→A5 迁移当前状态

> 更新时间：2026-07-21
> 算子路径：`ops-transformer/attention/fused_floyd_attention`

## 一、代码状态总览

| 阶段 | 状态 | 说明 |
|---|---|---|
| A5 编译 | ✅ 通过 | `bash build.sh --pkg --soc=ascend950 --ops=fused_floyd_attention` |
| 算子包安装 | ✅ 成功 | kernel 产物在 `ascend950/fused_floyd_attention/` |
| Tiling（host 侧） | ✅ 通过 | A3/A5 UT 全部 PASSED，tiling 数据合理 |
| Kernel 执行（仿真） | ❌ Hang | kernel 入口已注释，防止仿真卡死 |
| 端到端功能验证 | ❌ 未通过 | 仿真 hang，kernel 需进一步适配 |

## 二、与初始代码的差异（8 个文件）

### 2.1 算子注册（P0 阻塞项，已完成）

**文件**：`op_host/fused_floyd_attention_def.cpp:89`

```cpp
this->AICore().AddConfig("ascend950", aicore_config);
```

新增 A5 算子配置注册。原代码仅注册 `ascend910b`（A2）和 `ascend910_93`（A3）。

### 2.2 Kernel Cube 宏分支扩展（P0 阻塞项，已完成）

**文件**：`op_kernel/fused_floyd_attention.cpp:22`

```cpp
// 原: #ifdef __DAV_C220_CUBE__
// 改: #if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
```

使 A5（`__DAV_C310_CUBE__`）能进入 Cube 侧分支，正确提取 tiling data 和注册 matmul 对象。

### 2.3 DropOutBitModeInit 架构条件编译（编译期发现，已完成）

**文件**：
- `op_kernel/fused_floyd_attention_s1s2_bn2gs1.h:357-360`
- `op_kernel/fused_floyd_attention_s1s2_bn2gs1special.h:270-273`

```cpp
LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
#if defined(__DAV_C220_CUBE__)
    DropOutBitModeInit(apiTmpBuffer);
#endif
```

**原因**：`DropOutBitModeInit` 是 A2/A3 SIMD Membase 架构的 cmp mask 初始化函数，A5 Regbase 架构不提供（dropout 重写为寄存器级 API）。本算子 `hasDrop` 固定为 `false`，该调用无实际作用，用架构宏条件编译跳过。详见 `docs/fused_floyd_attention_a3_to_a5_issues.md`。

### 2.4 Kernel 入口注释（调试临时措施）

**文件**：`op_kernel/fused_floyd_attention.cpp:120-169`

kernel 入口函数体内的 `TPipe`/`SetMaskNorm` 和所有 `TILING_KEY_IS` 分支均被注释，使 kernel 成为空函数。目的：在 A5 仿真时绕过 kernel 执行卡死问题，单独验证 tiling 阶段。

> **注意**：此为临时调试状态，功能验证完成后需恢复。

### 2.5 文档支持矩阵更新（P0 阻塞项，已完成）

**文件**：
- `README.md:7`
- `docs/aclnnFusedFloydAttention.md:8`

A5 支持状态 × → √。

### 2.6 Example shape 修正 + 缩小

**文件**：`examples/test_aclnn_fused_floyd_attention.cpp:98-116`

两处修改：
1. **atten_mask shape 修正**：`{B, H, N, M, K}` → `{B, 1, N, 1, K}`，满足 `CheckAttentionMaskShape` 约束（dim1=1, dim3=1）
2. **shape 缩小**：`H=32, N=128` → `H=1, N=16`，降低仿真数据量（256 倍），避免仿真耗时过长

### 2.7 UT 新增 A3/A5 tiling dump 用例

**文件**：`tests/ut/op_host/test_fused_floyd_attention_tiling.cpp`

新增两个用例：
- `fused_floyd_attention_a5_dump_tiling`：A5 平台参数（28 cube / 56 vector, UB=248KB, L0C=256KB）
- `fused_floyd_attention_a3_dump_tiling`：A3 平台参数（24 cube / 48 vector, UB=192KB, L0C=128KB）

用相同的最小 shape（B=1,H=1,N=16,M=128,K=128,D=32）执行 tiling，将结果按 int64_t 数组打印到 stdout 和文件，便于 A3/A5 对比。

## 三、Tiling 数据测试状态

### 3.1 测试结果

| 平台 | UT 用例数 | PASSED | dump 文件 |
|---|---|---|---|
| A3 (`--soc=ascend910_93`) | 6 | 6 | `docs/fused_floyd_attention_a3_tiling_dump.txt` |
| A5 (`--soc=ascend950`) | 5 | 5 | `docs/fused_floyd_attention_a5_tiling_dump.txt` |

### 3.2 A3 vs A5 tiling 对比

测试 shape: B=1, H=1, N=16, M=128, K=128, D=32 (FP16 + atten_mask)

**结论**：tiling 结果几乎完全一致，仅 5 处差异，全部源于 L0C 扩容（128KB→256KB）：

| 差异项 | A3 | A5 | 原因 |
|---|---|---|---|
| workspace[0] | 136 MB | 156 MB | L0C 扩容，单核 workspace 增大 |
| 4 个 Bmm 的 `shareL0CSize` | 131072 (128KB) | 262144 (256KB) | L0C 翻倍 |

**一致项**：tilingKey、blockNum、tilingDataSize、InputParams、MultiCoreParams、CoreParams、TensorSizeParams、SoftmaxTiling 全部相同。

> tilingKey = `10000000001022030943`（A3/A5 一致），说明 tiling key 不含架构信息，kernel 侧分支选择不受架构影响。

详见 `docs/fused_floyd_attention_a3_a5_tiling_diff.md`。

### 3.3 Tiling 正确性结论

- tiling 逻辑在 A5 上工作正常，`PlatformAscendC` 接口正确获取 A5 平台参数
- 差异完全符合 L0C 扩容预期，无异常
- 仿真 hang 问题与 tiling 无关，出在 kernel 执行阶段

## 四、主要适配点总结

| # | 适配点 | 文件 | 状态 | 类型 |
|---|---|---|---|---|
| 1 | 算子注册 ascend950 | `def.cpp:89` | ✅ 完成 | P0 阻塞 |
| 2 | Cube 宏分支扩展 | `fused_floyd_attention.cpp:22` | ✅ 完成 | P0 阻塞 |
| 3 | DropOutBitModeInit 条件编译 | `s1s2_bn2gs1.h:357`、`s1s2_bn2gs1special.h:270` | ✅ 完成 | 编译期发现 |
| 4 | 文档支持矩阵 | `README.md`、`aclnnFusedFloydAttention.md` | ✅ 完成 | P0 阻塞 |
| 5 | Example shape 修正+缩小 | `test_aclnn_fused_floyd_attention.cpp` | ✅ 完成 | 调试 |
| 6 | UT tiling dump 用例 | `test_fused_floyd_attention_tiling.cpp` | ✅ 完成 | 验证 |
| 7 | Kernel 执行 hang 修复 | `fused_floyd_attention.cpp:120-169` | ❌ 待解决 | 运行时 |

## 五、待解决问题

### 5.1 Kernel 执行 hang（当前主要问题）

- **现象**：A5 仿真时 kernel 在硬件上 hang（task begin 但无 task end）
- **当前状态**：kernel 入口已注释，仅验证 tiling
- **可能原因**：
  1. kernel 中 100+ 处 `SetFlag/WaitFlag` 强配对同步，A5 BufferID 同步机制可能不兼容
  2. `SetMaskNorm()` 等 A2/A3 特定 API 在 A5 上行为不同
  3. matmul 库内部某些路径依赖 A5 已删除的 `L1→GM` 通路
- **下一步**：参考 A5 已适配算子（如 `flash_attention_score` arch35 regbase 实现）排查

### 5.2 Kernel 入口需恢复

功能验证完成后，需取消 `fused_floyd_attention.cpp:120-169` 的注释，恢复 kernel 执行逻辑。

## 六、相关文档索引

| 文档 | 内容 |
|---|---|
| `docs/fused_floyd_attention_a3_to_a5_todo.md` | 待修改点清单（P0 阻塞/测试/已就绪/架构速查） |
| `docs/fused_floyd_attention_a3_to_a5_migration.md` | 完整迁移分析（架构映射/差异/关联点/修改清单） |
| `docs/fused_floyd_attention_a3_to_a5_issues.md` | 遗留问题（DropOutBitModeInit A5 缺失） |
| `docs/fused_floyd_attention_a3_a5_tiling_diff.md` | A3 vs A5 tiling 对比 |
| `docs/fused_floyd_attention_a5_tiling_dump.txt` | A5 tiling 原始数据 |
| `docs/fused_floyd_attention_a3_tiling_dump.txt` | A3 tiling 原始数据 |
| `docs/run_floyd_a5_sim.sh` | A5 仿真运行脚本 |
