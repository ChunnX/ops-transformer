# quantMatmulReduceScatter 审查报告

## Round 0 审查报告（Step 4 初审）

- **审查日期**：2026-08-15
- **审查阶段**：Step 4 初审
- **判定**：FAIL
- **总分**：81 / 100

---

## 1. 审查概要

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| 1. 编译验证 | 10 | 9 | 独立编译通过；`-w` 抑制所有警告，无法验证警告状态 |
| 2. 架构合规 | 15 | 15 | Apace MC2 架构合规，入口属性/定义顺序/数据流完整 |
| 3. 编码规范 | 15 | 15 | 矢量 API 正确，数据对齐满足，命名规范 |
| 4. 性能优化 | 20 | 9 | **UB_SIZE_BYTES 硬编码为 192KB（DAV_3510 实际 248KB）**；无双缓冲；无独立性能采集 |
| 5. 测试覆盖 | 15 | 13 | Golden/验证脚本完善；缺少 Level 0（8-16 元素）基础用例 |
| 6. 精度验证 | 10 | 7 | K=3584 全部 PASS；**K=8192 时 4-rank BF16 精度 FAIL**（rel_err > 1e-2） |
| 7. 文档 | 15 | 13 | README/DESIGN 基本完善；缺 API 映射表和已知限制清单 |
| **合计** | **100** | **81** | |

**判定依据**：总分 81 ≥ 80，但存在必须修复问题（4.1 硬件参数硬编码），且 BF16 精度在 K=8192 场景下不达标 → **FAIL**

---

## 2. 独立构建验证

### 2.1 CMake 配置验证

```bash
python3 workflows/scripts/verify_cmake_config.py operators/quantMatmulReduceScatter/CMakeLists.txt
```

- ✅ `find_package(ASC REQUIRED)` 存在
- ✅ `project(... LANGUAGES CXX ASC)` 正确
- ✅ `--npu-arch=dav-3510` 匹配 Ascend 950PR
- ✅ 链接 `tiling_api` 等必需库
- ⚠️ 建议链接 `register` 库（非阻塞）

### 2.2 独立编译

- **编译器**：`/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/bin/bisheng`（environment.md 指定路径）
- **编译命令**：`cmake -S . -B build_reviewer && cmake --build build_reviewer --target quant_matmul_reduce_scatter_st --parallel 4`
- **结果**：✅ 编译成功，生成 `quant_matmul_reduce_scatter_st`（507608 bytes）
- **警告状态**：CMakeLists.txt 使用 `-w` 选项抑制所有编译器警告，无法验证是否存在潜在警告。**建议移除 `-w`，暴露并修复编译器警告后重新编译。**

### 2.3 NPU 架构核对

| 项 | 值 | 来源 |
|----|----|----|
| 芯片型号 | Ascend 950PR | environment.md |
| SocVersion | ASCEND950 | environment.md |
| NpuArch | DAV_3510 | `/npu-arch` skill 查得 |
| `--npu-arch` | dav-3510 | CMakeLists.txt ✅ 匹配 |
| `ASC_DEVKIT_MAJOR` | 9 | CMakeLists.txt ✅ 匹配 |

---

## 3. 代码质量评估

### 3.1 架构合规性（维度 2：15/15）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 2.1 架构模式 | ✅ | MC2 通算融合算子，使用 Apace 框架（CollectiveComm + QuantMatmulMxKernel + Blaze::Gemm），tensor_api 内存管理，符合 DAV_3510 MC2 架构要求 |
| 2.2 入口属性 | ✅ | `__global__ __aicore__` 正确（kernel_launcher.h:22） |
| 2.3 定义顺序 | ✅ | Kernel 入口函数定义在 Impl 类之后，无前向声明 |
| 2.4 内存管理 | ✅ | UB 使用 `MakeMemPtr` 分配，GM 使用 `aclrtMalloc/aclrtFree` 配对 |
| 2.5 数据流完整 | ✅ | AIC matmul → CrossCoreSetFlag → AIV WaitFlag → AllToAll PUT → reduceSum → output GM，数据流闭环 |

### 3.2 编码规范（维度 3：15/15）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 3.1 矢量 API | ✅ | 使用 `Add`、`DataCopy` 矢量 API，无 GetValue/SetValue 逐元素操作 |
| 3.2 API 约束 | ✅ | DataCopy 用于 32B 对齐数据（rowBytes = n×sizeof(bfloat16) = 4096×2 = 8192B，满足 32B 对齐） |
| 3.3 数据对齐 | ✅ | reduce_sum.h subM 强制 8 行对齐，rowBytes 天然 32B 对齐 |
| 3.4 命名规范 | ✅ | 文件/类/函数命名一致，`quant_matmul_reduce_scatter_*` 命名风格 |

### 3.3 性能优化（维度 4：9/20）

#### 4.1 动态硬件参数（0/4）— **必须修复**

**问题**：`kernel/reduce_sum.h:65` 硬编码 UB 大小：

```cpp
constexpr uint32_t UB_SIZE_BYTES = 192 * 1024;  // 192KB
```

- DAV_3510（Ascend 950PR）的实际 UB 大小为 **248KB**（通过 `/npu-arch` skill 查得），而非 192KB。
- 192KB 是 DAV_2201（Ascend 910B）的 UB 大小，属于跨架构硬编码错误。
- 同文件 line 77 已使用框架提供的 `TOTAL_UB_SIZE` 常量分配 LocalTensor，但 line 65 的 `UB_SIZE_BYTES` 仍硬编码，导致 subM 计算偏低，浪费 ~56KB UB 空间，增加 reduceSum 迭代次数。

**修复建议**：将 `UB_SIZE_BYTES` 替换为 `TOTAL_UB_SIZE` 或通过 `GetCoreMemSize` 动态获取。

#### 4.2 多核并行（3/4）

- AIC 侧：所有核通过 `BlockSchedulerQuantBatchMatmulV3` 参与 matmul 计算 ✅
- AIV 侧：`totalJobs=1`，仅 block 0 执行通信，其余 AIV 核空转等待 SyncAll。这是 MC2 PUT 模式的正确设计（避免多核 barrier 竞争），但 AIV 利用率低。

#### 4.3 流水线/双缓冲（1/4）

- reduce_sum.h 使用单缓冲 accum/temp 模式，无 `TQue<..., 2>` 双缓冲
- AIC 计算与 AIV 通信完全串行（AIV 等待 AIC 完成后才开始 AllToAll），无计算-通信重叠
- DESIGN.md section 11 将"AIC/AIV 流水重叠"列为未来优化方向

#### 4.4 同步策略（2/4）

**同步 API 逐项依赖分析**（reduce_sum.h）：

| 行号 | Flag 类型 | 前操作 (Pipe) | 后操作 (Pipe) | 依赖类型 | 判定 |
|------|-----------|---------------|---------------|----------|------|
| 96-97 | MTE2_V | DataCopy(self→accum) (MTE2) | Add (V) | RAW 跨 pipe | ✅ 必要 |
| 107-108 | MTE2_V | DataCopy(recv→temp) (MTE2) | Add (V) | RAW 跨 pipe | ✅ 必要 |
| 111-112 | V_MTE2 | Add (V) | 下一轮 DataCopy(temp) (MTE2) | WAR 跨 pipe | ✅ 必要 |
| 119-120 | V_MTE3 | Add (V) | DataCopy(accum→out) (MTE3) | RAW 跨 pipe | ✅ 必要 |
| 122-123 | MTE3_V | DataCopy(out) (MTE3) | 下一轮 DataCopy(accum) (MTE2) | WAR MTE3→MTE2 | ⚠️ 见下 |

**潜在问题**（line 122-123）：`MTE3_V` flag 保护的依赖是 MTE3 读 accumTensor → 下一轮 MTE2 写 accumTensor（WAR）。但 `MTE3_V` 语义为"MTE3 完成后 V 可执行"，并未直接保护 MTE3→MTE2 依赖。如果 MTE2 和 MTE3 是独立 pipe，下一轮 MTE2 可能在 MTE3 完成前覆写 accumTensor，导致数据竞争。

**建议**：确认 DAV_3510 上 MTE2/MTE3 是否共享硬件单元（隐式保序）。若不共享，需使用正确的 MTE3→MTE2 同步机制。

**CrossCoreFlag 同步**（impl.h）：

| 行号 | Flag | 说明 | 判定 |
|------|------|------|------|
| 218 | `CrossCoreSetFlag<0x2, PIPE_FIX>(0)` | AIC FixPipe 完成后通知 AIV | ✅ 正确（PIPE_FIX 是 AIC 输出阶段） |
| 224 | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)` | AIV MTE2 等待 AIC 完成 | ✅ 正确 |
| 313,320 | `SyncAll<true>()` | AIV 核间同步（isAIVOnly=true） | ✅ 正确（AIC 不参与） |

**冗余率**：5 个 flag 中 4 个必要、1 个存疑 → 冗余率 ~0%（无明确冗余 flag）

#### 4.5 计算效率与上板性能（2/4）

- ✅ 无循环内逐行 API 调用，使用批量 `Add(tensor, tensor, tensor, count)`
- ✅ 无不必要的重复 GM 读取（每个 chunk 仅读一次）
- ❌ 无独立性能采集（PLAN.md 标记性能验收 ⬜ 未完成）
- ❌ perf 模式使用 wall-clock 计时，非 msprof 采集
- ❌ `src/main.cpp:283` `HEAVY_BLOCK_NUM = 56` 死代码（定义但未使用）

---

## 4. 设计合规检查

### 4.1 DESIGN.md vs 代码一致性

| 检查项 | DESIGN.md | 代码 | 一致性 |
|--------|-----------|------|--------|
| 通信模式 | "PUT 或 GET，按硬件/场景选择" (§6.2) | `CommMode::PUT` (impl.h:131) | ✅ 一致 |
| CrossCore Flag Pipe | `CrossCoreSetFlag<0x2, PIPE_MTE3>` (§8.3) | `CrossCoreSetFlag<0x2, PIPE_FIX>` (impl.h:218) | ❌ **不一致** |
| L0C 原子累加 | "不使用 SetAtomicAdd" (§8.4) | 无 SetAtomicAdd 调用 | ✅ 一致 |
| MatmulMode | LOCAL 模式，splitK=1 (§9.2) | `MatmulMode::LOCAL` (impl.h:273) | ✅ 一致 |
| reduceSum 逐块累加 | "每收齐一个 rank 的分片即累加" (§6.3) | 逐 rank Add 累加 (reduce_sum.h:100-113) | ✅ 一致 |
| workspace 用途 | 通信窗口 (§7.3) | AIC 写入 + PUT 推送源 (impl.h:277) | ✅ 一致 |

**DESIGN.md 不一致项**：§8.3 写 `PIPE_MTE3`，代码使用 `PIPE_FIX`。`PIPE_FIX` 是正确的（AIC FixPipe 是输出阶段），DESIGN.md 应更新为 `PIPE_FIX`。

### 4.2 MC2 最佳实践合规

- ✅ 使用 Apace 框架（CollectiveComm + QuantMatmulMxKernel）
- ✅ Matmul 走 Blaze::Gemm（BlockMmad + BlockScheduler），非 asc-devkit matmul API
- ✅ PUT 模式为官网验证模式
- ✅ AIC/AIV 分工明确（AIC 计算，AIV 通信+求和）
- ⚠️ 单 tile 通信（无多 tile 流水），DESIGN.md §11 标注为未来优化

---

## 5. 测试覆盖评估

| 测试级别 | 要求 | 状态 | 说明 |
|---------|------|------|------|
| Level 0 | 8-16 元素基础功能 | ❌ 未覆盖 | 最小用例 M=1024，无极小 shape 验证 |
| Level 1 | 1K 元素典型场景 | ✅ | cases.csv 含 M=1024~4096, K=3584 |
| Level 2 | 极值/零值边界 | ⚠️ 部分 | M=4040（非对齐）在 cases.csv 中但精度 FAIL |
| Level 3 | 大数据量性能 | ❌ 未覆盖 | 无性能测试用例 |

### 5.1 测试数据生成（4/4）

`gen_data.py` 正确实现：
- MXFP8 量化 A/B 矩阵 + fp8_e8m0 Scale
- float32 累加 reduceSum（避免 BF16 累加误差）
- B 矩阵所有 rank 共享，A 矩阵每卡独立
- 输出 per rank: [chunkM, n] bfloat16

### 5.2 结果验证脚本（4/4）

`verify_result.py` 正确实现：
- `torch.allclose(rtol=1e-2, atol=1e-2, equal_nan=True)` 双容差验证
- 逐 rank 输出比对
- 详细的误差统计（max/mean abs_err, max rel_err, 超差计数）

### 5.3 精度标准（3/3）

- BF16: atol=1e-2, rtol=1e-2 ✅ 符合 `/ops-precision-standard` 中 BF16 标准

---

## 6. 精度验证

### 6.1 独立精度测试结果

| 用例 | M | K | N | rank | 结果 | max rel_err | 说明 |
|------|---|---|---|------|------|-------------|------|
| T1 | 2048 | 3584 | 4096 | 2 | ✅ PASS | 6.71e-3 | 基本功能验证 |
| T2 | 2048 | 3584 | 4096 | 4 | ✅ PASS | 7.81e-3 | 4 卡通信验证 |
| T3 | 4096 | 3584 | 4096 | 4 | ✅ PASS | <1e-2 | 大 M 4 卡验证 |
| T4 | 4096 | 8192 | 4096 | 4 | ❌ **FAIL** | **1.09e-2** | **K=8192 精度超标** |
| T5 | 4040 | 8192 | 4096 | 4 | ❌ **FAIL** | **1.09e-2** | **K=8192 + 非对齐 M** |

### 6.2 精度问题诊断

**问题类型**：精度问题（BF16 累加精度不足）

**根因分析**：

| 侧 | 累加方式 | 精度 |
|----|---------|------|
| NPU (reduce_sum.h) | BF16 Add 累加（`Add(accumTensor, accumTensor, tempTensor)` 其中 CType=bfloat16_t） | 7 位尾数，4 次累加误差 ~0.8% |
| Golden (gen_data.py) | float32 累加后转 BF16 | 23 位尾数，误差可忽略 |

- K=3584 时：matmul 输出值较小，BF16 累加误差 < 1%，PASS
- K=8192 时：matmul 输出值增大（正比于 K），BF16 累加误差 > 1%，FAIL
- Developer 已意识到此问题（PLAN.md 记录"golden float32 累加"），但 **NPU 侧 reduceSum 未做对应优化**

**问题特征**：所有 dtype 均匀地精度不足 → 精度问题（数值稳定性）

### 6.3 修复建议

**reduceSum float32 累加优化**：

```cpp
// reduce_sum.h 修改方案：
// 1. 将 accumTensor 声明为 float 类型（或额外分配 float accumBuf）
// 2. 读入 BF16 数据后 Cast 到 float32
// 3. Add 在 float32 下累加
// 4. 最终输出前 Cast 回 BF16
//
// 伪代码：
LocalTensor<float> accumF32 = ...;  // float32 累加缓冲区
LocalTensor<CType> tempBf16 = ...;  // BF16 临时缓冲区
LocalTensor<float> tempF32 = ...;   // float32 临时缓冲区

// 读 self → tempBf16 → Cast tempF32 → Copy to accumF32
DataCopy(tempBf16, selfGm, curCount);
Cast(tempF32, tempBf16, RoundMode::CAST_NONE, curCount);
// accumF32 = tempF32 (首次)
// ...

// 逐 rank：读 received → tempBf16 → Cast tempF32 → Add(accumF32, accumF32, tempF32)
// ...

// 最终：Cast accumF32 → accumBf16 → DataCopy to output
Cast(accumBf16, accumF32, RoundMode::CAST_RINT, curCount);
DataCopy(outGm, accumBf16, curCount);
```

**参考**：`/ascendc-precision-debug` skill 中 BF16 累加精度优化指南

### 6.4 精度评分

| 检查项 | 满分 | 得分 | 说明 |
|--------|------|------|------|
| 6.1 FP32 全用例 PASS | 4 | 4 | N/A（算子仅输出 BF16，无 FP32 路径） |
| 6.2 FP16 全用例 PASS | 3 | 3 | N/A（算子仅输出 BF16，无 FP16 路径） |
| 6.3 BF16 全用例 PASS | 3 | 0 | K=8192 场景 FAIL（cases.csv 中声明的用例未通过） |
| **合计** | **10** | **7** | |

---

## 7. 文档审查

| 检查项 | 满分 | 得分 | 说明 |
|--------|------|------|------|
| 7.1 README.md 存在 | 3 | 3 | ✅ |
| 7.2 数学公式 | 3 | 3 | ✅ `output = reduceSum(AlltoAll(Dequant(QuantA @ QuantB)))` |
| 7.3 编译运行指南 | 3 | 3 | ✅ 含前置条件、编译命令、运行命令 |
| 7.4 API 映射/约束 | 3 | 2 | ⚠️ 有关键设计决策表，但无完整 API 映射表（Apace/Blaze API → 算子功能映射） |
| 7.5 已知限制 | 3 | 2 | ⚠️ DESIGN.md §10 有约束，但 README 未明确列出已知限制（如 K=8192 精度问题、GET 模式不稳定性） |
| **合计** | **15** | **13** | |

---

## 8. 必须修复问题清单

### MF-1：UB_SIZE_BYTES 硬编码（4.1，阻塞项）

- **文件**：`kernel/reduce_sum.h:65`
- **问题**：`constexpr uint32_t UB_SIZE_BYTES = 192 * 1024;` 硬编码为 DAV_2201 的 UB 大小（192KB），DAV_3510 实际为 248KB
- **影响**：浪费 56KB UB 空间，reduceSum subM 偏小，迭代次数增加，性能下降
- **修复**：替换为 `TOTAL_UB_SIZE`（框架提供的编译时常量，已在 line 77 使用）或通过 `GetCoreMemSize` 动态获取
- **参考路径**：`/npu-arch` skill → DAV_3510 硬件参数

### MF-2：BF16 reduceSum 精度不足（6.3，阻塞项）

- **文件**：`kernel/reduce_sum.h:110`
- **问题**：`Add(accumTensor, accumTensor, tempTensor, curCount)` 在 BF16 下累加，K=8192 时 4-rank 精度超标（rel_err > 1e-2）
- **影响**：cases.csv 中 K=8192 用例全部 FAIL
- **修复**：reduceSum 中 Cast 到 float32 累加，最终 Cast 回 BF16（见 §6.3 修复建议）
- **参考路径**：`/ascendc-precision-debug` skill → BF16 累加精度优化

---

## 9. 建议修复问题清单（非阻塞）

### RN-1：注释与代码模式不一致（GET vs PUT）

- **范围**：`quant_matmul_reduce_scatter_impl.h`（9 处）、`reduce_sum.h`（2 处）、`main.cpp`（7 处）、`kernel_launcher.h`（2 处），共 20+ 处注释仍写 "GET 模式"，但代码实际使用 `CommMode::PUT`
- **修复**：全局替换 GET → PUT 相关注释，确保注释与代码一致

### RN-2：reduce_sum.h 参数名与实际语义相反

- **文件**：`kernel/reduce_sum.h:51-52`
- **问题**：`Run()` 参数名 `selfWinAddr` 实际接收 `workspaceGm`，参数名 `workspaceGM` 实际接收 `selfWinAddr`（调用处 impl.h:344-346 交换了顺序）
- **修复**：修正参数名使其与实际语义一致，或修正调用顺序

### RN-3：DESIGN.md §8.3 PIPE 标识不一致

- **DESIGN.md**：`CrossCoreSetFlag<0x2, PIPE_MTE3>(0)`
- **代码**：`CrossCoreSetFlag<0x2, PIPE_FIX>(0)`（impl.h:218）
- **修复**：更新 DESIGN.md 为 `PIPE_FIX`（代码是正确的）

### RN-4：main.cpp 死代码

- **文件**：`src/main.cpp:283`
- **问题**：`constexpr int64_t HEAVY_BLOCK_NUM = 56;` 定义但未使用
- **修复**：删除该死代码

### RN-5：-w 编译选项抑制警告

- **文件**：`CMakeLists.txt:90`
- **问题**：`KERNEL_OPTS` 含 `-w`，抑制所有编译器警告，无法发现潜在问题
- **修复**：移除 `-w`，修复暴露的编译器警告

### RN-6：reduce_sum.h MTE3_V 同步可能不正确

- **文件**：`kernel/reduce_sum.h:122-123`
- **问题**：`MTE3_V` flag 保护的依赖是 MTE3→MTE2（WAR on accumTensor），但 MTE3_V 语义为 MTE3→V，可能无法正确保护 MTE3→MTE2 依赖
- **修复**：确认 DAV_3510 MTE2/MTE3 pipe 关系；若独立则使用正确的同步机制

### RN-7：缺少 Level 0 基础功能测试

- **问题**：cases.csv 最小用例 M=1024，无 8-16 元素极小 shape 验证
- **修复**：添加 M=16, K=64, N=32, rank=2 的极小用例

### RN-8：无独立性能采集

- **问题**：PLAN.md 标记性能验收 ⬜ 未完成，main.cpp perf 模式仅 wall-clock 计时
- **修复**：使用 `/ops-profiling` skill 独立采集 msprof 数据，归档到 `docs/perf/`

---

## 10. 同步策略审查汇总

### 10.1 CrossCoreFlag（AIC↔AIV 跨核同步）

| 位置 | API | 判定 |
|------|-----|------|
| impl.h:218 | `CrossCoreSetFlag<0x2, PIPE_FIX>(0)` | ✅ AIC FixPipe 完成后通知 AIV |
| impl.h:224 | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)` | ✅ AIV MTE2 等待 AIC 完成 |

### 10.2 SyncAll（AIV 核间同步）

| 位置 | API | 判定 |
|------|-----|------|
| impl.h:313 | `SyncAll<true>()` (CrossDevice 后) | ✅ 确保所有 rank AIC 完成 |
| impl.h:320 | `SyncAll<true>()` (每轮 Commit 后) | ✅ 确保所有 rank 通信完成 |

### 10.3 reduce_sum.h HardEvent Flag

| 行号 | Flag | 判定 | 说明 |
|------|------|------|------|
| 96-97 | MTE2_V | ✅ 必要 | DataCopy→Add，RAW 跨 pipe |
| 107-108 | MTE2_V | ✅ 必要 | DataCopy→Add，RAW 跨 pipe |
| 111-112 | V_MTE2 | ✅ 必要 | Add→下一轮 DataCopy，WAR 跨 pipe |
| 119-120 | V_MTE3 | ✅ 必要 | Add→DataCopy(out)，RAW 跨 pipe |
| 122-123 | MTE3_V | ⚠️ 存疑 | MTE3→MTE2 WAR 依赖，flag 类型可能不匹配 |

**冗余率**：0/5 = 0%（无冗余 flag，但 1 个 flag 类型存疑）

---

## 11. 审查结论

### 判定：FAIL

**原因**：
1. **必须修复 MF-1**：`reduce_sum.h:65` UB_SIZE_BYTES 硬编码为 192KB（DAV_3510 实际 248KB），违反"硬件参数动态获取"规则（检查项 4.1 未通过 → 自动 FAIL）
2. **必须修复 MF-2**：K=8192 时 BF16 reduceSum 精度不达标（cases.csv 声明用例 FAIL），reduceSum 需改为 float32 累加

### 修复优先级

| 优先级 | 问题 | 修复难度 |
|--------|------|----------|
| P0 | MF-1: UB_SIZE_BYTES 硬编码 | 低（替换常量） |
| P0 | MF-2: BF16 reduceSum 精度 | 中（需增加 float32 Cast + 累加逻辑，注意 UB 空间分配） |
| P1 | RN-1: GET→PUT 注释更新 | 低（批量替换） |
| P1 | RN-2: 参数名修正 | 低 |
| P1 | RN-3: DESIGN.md PIPE 修正 | 低 |
| P2 | RN-4~RN-8: 其他清理和补充 | 低-中 |

### 下一步

Developer 修复 MF-1 和 MF-2 后，重新提交进入 Step 5 复审。建议同时处理 RN-1~RN-3（注释一致性），以避免复审时产生不必要的质疑。

---

## Round 1 审查报告（Step 5 复审）

- **审查日期**：2026-08-15
- **审查阶段**：Step 5 复审（Round 1）
- **判定**：PASS
- **总分**：87 / 100

---

## 1. 审查概要

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| 1. 编译验证 | 10 | 9 | 独立编译通过；`-w` 仍抑制警告（RN-5 未修复，非阻塞） |
| 2. 架构合规 | 15 | 15 | Apace MC2 架构合规，入口属性/定义顺序/数据流完整 |
| 3. 编码规范 | 15 | 15 | 矢量 API 正确，float32 Cast 累加正确处理 repeatTime 限制，数据对齐满足 |
| 4. 性能优化 | 20 | 14 | **MF-1 已修复**（TOTAL_UB_SIZE）；同步策略全部必要（RN-6 已解决）；无双缓冲；无独立性能采集 |
| 5. 测试覆盖 | 15 | 11 | Golden/验证脚本完善；仍缺 Level 0 基础用例（RN-7 未修复） |
| 6. 精度验证 | 10 | 10 | **MF-2 已修复**；所有用例 PASS（K=3584/K=8192/M=4040，2-rank/4-rank） |
| 7. 文档 | 15 | 13 | README/DESIGN 基本完善；缺 API 映射表和已知限制清单 |
| **合计** | **100** | **87** | |

**判定依据**：总分 87 ≥ 80，无必须修复问题 → **PASS**

---

## 2. 上轮修复验证

### 2.1 MF-1：UB_SIZE_BYTES 硬编码 → TOTAL_UB_SIZE ✅ 已修复

- **文件**：`kernel/reduce_sum.h:67-69`
- **修复内容**：
  ```cpp
  // 修复前: constexpr uint32_t UB_SIZE_BYTES = 192 * 1024;
  // 修复后:
  constexpr uint32_t COMM_RESERVED = 1024;
  uint32_t availUb = TOTAL_UB_SIZE - COMM_RESERVED;
  ```
- **验证结果**：✅ 确认使用框架常量 `TOTAL_UB_SIZE`，Grep 检查无硬编码 UB 大小
- **影响**：DAV_3510 UB 从错误的 192KB 修正为正确的 248KB（TOTAL_UB_SIZE），subM 增大，reduceSum 迭代次数减少

### 2.2 MF-2：BF16 reduceSum 精度不足 → float32 累加 ✅ 已修复

- **文件**：`kernel/reduce_sum.h:130-172`
- **修复内容**：
  1. `Duplicate<float>` 初始化 accumF32 为 0（分批处理，避免 repeatTime 溢出）
  2. 逐 rank：`DataCopy` BF16 → `Cast` BF16→float32 → `Add` float32 累加
  3. 最终：`Cast` float32→BF16（`CAST_RINT` 舍入）→ `DataCopy` 写入 output GM
- **关键改进**：
  - float Vector API repeatTime 为 uint8_t（max 255），每次处理 64 个 float 元素，单次上限 255×64=16320
  - 代码通过 `MAX_FLOAT_ELEMS_PER_CALL=16320` 限制并分批调用，避免 repeatTime 溢出
- **验证结果**：✅ 独立精度测试全部 PASS（见 §6）
- **参考**：vec_comp.h DequantReduceSum

### 2.3 附带修复

| Round 0 问题 | 状态 | 说明 |
|-------------|------|------|
| RN-2: 参数名修正 | ✅ 已修复 | reduce_sum.h 参数名改为 `selfDataGM`/`receivedDataGM`/`outputGM`，语义与调用一致 |
| RN-6: MTE3_V 同步存疑 | ✅ 已解决 | 新代码在每个 rank 迭代开始添加 `V_MTE2` sync（line 149-150），形成 MTE3→V→MTE2 链，正确保护 tempCType 的 WAR 依赖 |

### 2.4 未修复的非阻塞问题

| Round 0 问题 | 状态 | 说明 |
|-------------|------|------|
| RN-1: GET→PUT 注释 | ❌ 未修复 | 仍有 20+ 处注释写 "GET 模式"，代码实际使用 PUT |
| RN-3: DESIGN.md PIPE 标识 | ❌ 未修复 | DESIGN.md §8.3 仍写 `PIPE_MTE3`，代码使用 `PIPE_FIX` |
| RN-4: main.cpp 死代码 | ❌ 未修复 | `HEAVY_BLOCK_NUM = 56` 定义但未使用 |
| RN-5: -w 编译选项 | ❌ 未修复 | CMakeLists.txt:90 仍有 `-w` |
| RN-7: 缺 Level 0 测试 | ❌ 未修复 | 无 8-16 元素极小 shape 用例 |
| RN-8: 无独立性能采集 | ❌ 未修复 | PLAN.md 性能验收仍 ⬜ |

---

## 3. 独立构建验证

### 3.1 CMake 配置验证

```bash
python3 workflows/scripts/verify_cmake_config.py operators/quantMatmulReduceScatter/CMakeLists.txt
```

- ✅ `find_package(ASC REQUIRED)` 存在
- ✅ `project(... LANGUAGES CXX ASC)` 正确
- ✅ `--npu-arch=dav-3510` 匹配 Ascend 950PR
- ✅ 链接 `tiling_api` 等必需库
- ⚠️ 建议链接 `register` 库（非阻塞）

### 3.2 独立编译

- **编译器**：`/home/l00947526/Ascend806/cann-9.2.0/x86_64-linux/bin/bisheng`（environment.md 指定路径）
- **编译目录**：`build_reviewer`（Reviewer 独立目录，非 Developer 的 build）
- **编译命令**：`cmake -S . -B build_reviewer && cmake --build build_reviewer --target quant_matmul_reduce_scatter_st --parallel 4`
- **结果**：✅ 编译成功，生成 `quant_matmul_reduce_scatter_st`（510224 bytes）
- **警告状态**：CMakeLists.txt 使用 `-w` 选项抑制所有编译器警告（RN-5，非阻塞）

### 3.3 NPU 架构核对

| 项 | 值 | 来源 |
|----|----|----|
| 芯片型号 | Ascend 950PR | environment.md |
| SocVersion | ASCEND950 | environment.md |
| NpuArch | DAV_3510 | `/npu-arch` skill 查得 |
| `--npu-arch` | dav-3510 | CMakeLists.txt ✅ 匹配 |
| `ASC_DEVKIT_MAJOR` | 9 | CMakeLists.txt ✅ 匹配 |
| UB 大小 | 248 KB | `/npu-arch` skill 查得（代码使用 TOTAL_UB_SIZE ✅） |

### 3.4 硬件参数 Grep 检查

```bash
grep -n "blockDim\s*=\s*[0-9]" → 无匹配 ✅
grep -n "blockIdx\s*=\s*[0-9]" → 无匹配 ✅
grep -n "192\s*\*\s*1024" → 仅注释中出现（修复历史说明）✅
```

---

## 4. 代码质量评估

### 4.1 架构合规性（维度 2：15/15）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 2.1 架构模式 | ✅ | MC2 通算融合算子，Apace 框架（CollectiveComm + QuantMatmulMxKernel + Blaze::Gemm），tensor_api 内存管理 |
| 2.2 入口属性 | ✅ | `__global__ __aicore__` 正确（kernel_launcher.h:22） |
| 2.3 定义顺序 | ✅ | Kernel 入口函数定义在 Impl 类之后 |
| 2.4 内存管理 | ✅ | UB 使用 `MakeMemPtr`/`LocalTensor`，GM 使用 `aclrtMalloc/aclrtFree` 配对 |
| 2.5 数据流完整 | ✅ | AIC matmul → CrossCoreSetFlag → AIV WaitFlag → AllToAll PUT → reduceSum → output GM |

### 4.2 编码规范（维度 3：15/15）

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 3.1 矢量 API | ✅ | 使用 `Add`、`Cast`、`Duplicate`、`DataCopy` 矢量 API，无 GetValue/SetValue |
| 3.2 API 约束 | ✅ | float Vector API repeatTime 限制（max 255×64=16320 元素）通过分批调用正确处理；DataCopy 32B 对齐；Cast 使用 `CAST_NONE`（BF16→F32）和 `CAST_RINT`（F32→BF16） |
| 3.3 数据对齐 | ✅ | subM 强制 alignRows 对齐；rowBytes = n×sizeof(bfloat16) = 4096×2 = 8192B（32B 对齐）；UB 偏移均 32B 对齐 |
| 3.4 命名规范 | ✅ | 文件/类/函数命名一致，`quant_matmul_reduce_scatter_*` 风格 |

### 4.3 性能优化（维度 4：14/20）

#### 4.1 动态硬件参数（4/4）✅ 已修复

- `reduce_sum.h:69` 使用 `TOTAL_UB_SIZE` 框架常量，无硬编码
- Grep 检查：无 `blockDim=N`、`blockIdx=N`、硬编码 UB/TILE 大小
- 核数通过 `tilingData.mmTilingData.usedCoreNum` 动态获取

#### 4.2 多核并行（3/4）

- AIC 侧：所有核通过 `BlockSchedulerQuantBatchMatmulV3` 参与 matmul 计算 ✅
- AIV 侧：`totalJobs=1`，仅 block 0 执行通信，其余 AIV 核空转等待 SyncAll。MC2 PUT 模式的正确设计，但 AIV 利用率低

#### 4.3 流水线/双缓冲（1/4）

- reduce_sum.h 使用单缓冲 accumF32/tempCType/tempF32 模式，无 `TQue<..., 2>` 双缓冲
- AIC 计算与 AIV 通信完全串行（AIV 等待 AIC 完成后才开始 AllToAll）
- DESIGN.md §11 将"AIC/AIV 流水重叠"列为未来优化方向

#### 4.4 同步策略（4/4）✅ RN-6 已解决

**同步 API 逐项依赖分析（reduce_sum.h — float32 版本）**：

| 行号 | Flag 类型 | 前操作 (Pipe) | 后操作 (Pipe) | 依赖类型 | 判定 |
|------|-----------|---------------|---------------|----------|------|
| 149-150 | V_MTE2 | Duplicate/Add/Cast (V) | DataCopy (MTE2) | WAR on tempCType | ✅ 必要 |
| 153-154 | MTE2_V | DataCopy (MTE2) | Cast (V) | RAW on tempCType | ✅ 必要 |
| 161 | PipeBarrier<V> | Cast (V) | Add (V) | RAW on tempF32（同 pipe，不同 tensor） | ✅ 必要 |
| 164 | PipeBarrier<V> | Add (V) | 下一批 Cast (V) | RAW on accumF32（同 pipe，不同 tensor） | ✅ 必要 |
| 173-174 | V_MTE3 | Cast float→BF16 (V) | DataCopy out (MTE3) | RAW on tempCType | ✅ 必要 |
| 181-182 | MTE3_V | DataCopy out (MTE3) | 下一轮 Duplicate (V) | MTE3→V，间接保护 MTE3→MTE2 WAR | ✅ 必要 |

**MTE3→MTE2 WAR 依赖链分析（RN-6 解决方案验证）**：

Round 0 担忧 MTE3_V 无法保护 MTE3→MTE2 的 WAR 依赖。新代码通过以下链式同步解决：

```
迭代结束: DataCopy out (MTE3 读 tempCType)
    → SetFlag<MTE3_V> + WaitFlag<MTE3_V>   (MTE3 完成 → V 可执行)
    → Duplicate (V 写 accumF32, 不访问 tempCType)
    → PipeBarrier<V>
    → SetFlag<V_MTE2> + WaitFlag<V_MTE2>   (V 完成 → MTE2 可执行)
    → DataCopy in (MTE2 写 tempCType)
```

链式保证：MTE3 → V → MTE2，因此 MTE3 完成先于 MTE2 写 tempCType，WAR 依赖被正确保护。✅

**CrossCoreFlag 同步**（impl.h）：

| 行号 | Flag | 说明 | 判定 |
|------|------|------|------|
| 218 | `CrossCoreSetFlag<0x2, PIPE_FIX>(0)` | AIC FixPipe 完成后通知 AIV | ✅ 正确 |
| 224 | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)` | AIV MTE2 等待 AIC 完成 | ✅ 正确 |
| 313 | `SyncAll<true>()` (CrossDevice 后) | 所有 AIV 核同步 | ✅ 正确 |
| 320 | `SyncAll<true>()` (每轮 Commit 后) | 所有 AIV 核同步 | ✅ 正确 |

**冗余率**：0/6 = 0%（所有 flag 均必要，无冗余）

#### 4.5 计算效率与上板性能（2/4）

- ✅ 无循环内逐行 API 调用，使用批量 `Add`/`Cast`/`Duplicate`
- ✅ 无不必要的重复 GM 读取（每个 chunk 仅读一次）
- ❌ 无独立性能采集（PLAN.md 标记性能验收 ⬜ 未完成）
- ❌ perf 模式使用 wall-clock 计时，非 msprof 采集
- ❌ `src/main.cpp:283` `HEAVY_BLOCK_NUM = 56` 死代码（RN-4 未修复）

---

## 5. 设计合规检查

### 5.1 DESIGN.md vs 代码一致性

| 检查项 | DESIGN.md | 代码 | 一致性 |
|--------|-----------|------|--------|
| 通信模式 | "PUT 或 GET，按硬件/场景选择" (§6.2) | `CommMode::PUT` (impl.h:131) | ✅ 一致 |
| CrossCore Flag Pipe | `CrossCoreSetFlag<0x2, PIPE_MTE3>` (§8.3) | `CrossCoreSetFlag<0x2, PIPE_FIX>` (impl.h:218) | ❌ **不一致**（RN-3 未修复，代码正确） |
| L0C 原子累加 | "不使用 SetAtomicAdd" (§8.4) | 无 SetAtomicAdd 调用 | ✅ 一致 |
| MatmulMode | LOCAL 模式，splitK=1 (§9.2) | `MatmulMode::LOCAL` (impl.h:273) | ✅ 一致 |
| reduceSum 逐块累加 | "每收齐一个 rank 的分片即累加" (§6.3) | 逐 rank Add 累加 (reduce_sum.h:138-166) | ✅ 一致 |
| reduceSum 精度策略 | 未明确提及 float32 累加 | float32 Cast 累加 (reduce_sum.h) | ⚠️ DESIGN.md 应补充 float32 累加策略说明 |

### 5.2 MC2 最佳实践合规

- ✅ 使用 Apace 框架（CollectiveComm + QuantMatmulMxKernel）
- ✅ Matmul 走 Blaze::Gemm（BlockMmad + BlockScheduler），非 asc-devkit matmul API
- ✅ PUT 模式为官网验证模式
- ✅ AIC/AIV 分工明确（AIC 计算，AIV 通信+求和）
- ⚠️ 单 tile 通信（无多 tile 流水），DESIGN.md §11 标注为未来优化

---

## 6. 精度验证（独立运行）

### 6.1 独立精度测试结果

**测试环境**：Reviewer 独立编译（build_reviewer 目录），通过 `qrun` 排队在 NPU 上独立运行

| 用例 | M | K | N | rank | 结果 | max rel_err | 说明 |
|------|---|---|---|------|------|-------------|------|
| T1 | 2048 | 3584 | 4096 | 2 | ✅ PASS | ~6.5e-3 | 2 卡基本功能验证 |
| T2 | 2048 | 3584 | 4096 | 4 | ✅ PASS | 6.94e-3 | 4 卡通信验证 |
| T3 | 2048 | 8192 | 4096 | 4 | ✅ PASS | **6.21e-3** | **K=8192 精度修复验证**（Round 0: 1.09e-2 FAIL） |
| T4 | 4040 | 8192 | 4096 | 4 | ✅ PASS | ~6.2e-3 | **K=8192 + 非对齐 M**（Round 0: 1.09e-2 FAIL） |

### 6.2 精度标准

| 数据类型 | rtol | atol | 标准 |
|---------|------|------|------|
| BF16 | 1e-2 | 1e-2 | `/ops-precision-standard` BF16 标准 ✅ |

### 6.3 精度评分

| 检查项 | 满分 | 得分 | 说明 |
|--------|------|------|------|
| 6.1 FP32 全用例 PASS | 4 | 4 | N/A（算子仅输出 BF16，无 FP32 路径） |
| 6.2 FP16 全用例 PASS | 3 | 3 | N/A（算子仅输出 BF16，无 FP16 路径） |
| 6.3 BF16 全用例 PASS | 3 | 3 | ✅ 全部用例 PASS（含 Round 0 FAIL 的 K=8192 场景） |
| **合计** | **10** | **10** | |

### 6.4 Developer 自报数据核对

| 用例 | Developer 报告 | Reviewer 独立结果 | 一致性 |
|------|---------------|-------------------|--------|
| K=3584 4-rank | max rel_err=6.99e-3 | max rel_err=6.94e-3 | ✅ 一致（微小差异属正常） |
| K=8192 4-rank | max rel_err=6.21e-3 | max rel_err=6.21e-3 | ✅ 一致 |

---

## 7. 测试覆盖评估

| 测试级别 | 要求 | 状态 | 说明 |
|---------|------|------|------|
| Level 0 | 8-16 元素基础功能 | ❌ 未覆盖 | 最小用例 M=1024（RN-7 未修复） |
| Level 1 | 1K 元素典型场景 | ✅ | M=1024~4096, K=3584 |
| Level 2 | 极值/零值边界 | ✅ | M=4040（非对齐）+ K=8192（大 K）已覆盖并 PASS |
| Level 3 | 大数据量性能 | ❌ 未覆盖 | 无性能测试用例（RN-8 未修复） |

### 7.1 测试数据生成（4/4）

`gen_data.py` 正确实现：
- MXFP8 量化 A/B 矩阵 + fp8_e8m0 Scale
- **float32 累加 reduceSum**（line 103-108: `torch.zeros(..., dtype=torch.float32)` + `output += chunk.float()` + `output.to(torch.bfloat16)`）
- B 矩阵所有 rank 共享，A 矩阵每卡独立

### 7.2 结果验证脚本（4/4）

`verify_result.py` 正确实现：
- `torch.allclose(rtol=1e-2, atol=1e-2, equal_nan=True)` 双容差验证
- 逐 rank 输出比对 + 详细误差统计

### 7.3 精度标准（3/3）

- BF16: atol=1e-2, rtol=1e-2 ✅ 符合 `/ops-precision-standard`

---

## 8. 文档审查

| 检查项 | 满分 | 得分 | 说明 |
|--------|------|------|------|
| 7.1 README.md 存在 | 3 | 3 | ✅ |
| 7.2 数学公式 | 3 | 3 | ✅ `output = reduceSum(AlltoAll(Dequant(QuantA @ QuantB)))` |
| 7.3 编译运行指南 | 3 | 3 | ✅ 含前置条件、编译命令、运行命令 |
| 7.4 API 映射/约束 | 3 | 2 | ⚠️ 有关键设计决策表，但无完整 API 映射表 |
| 7.5 已知限制 | 3 | 2 | ⚠️ DESIGN.md §10 有约束，README 未明确列出已知限制 |
| **合计** | **15** | **13** | |

---

## 9. 最终轮附加检查

### 9.1 交付件检查清单（D1-D8）

| # | 交付件 | 状态 | 说明 |
|---|--------|------|------|
| D1 | 算子源码 | ✅ | 独立编译通过（`-w` 抑制警告，RN-5） |
| D2 | 构建文件 | ✅ | CMakeLists.txt 依赖项完整 |
| D3 | Golden 数据生成 | ✅ | gen_data.py 支持 BF16 输出（算子唯一输出 dtype） |
| D4 | 运行脚本 | ✅ | run.sh 可正常执行 |
| D5 | 算子文档 | ⚠️ | README 缺 API 映射表和已知限制（非阻塞） |
| D6 | 设计文档 | ✅ | DESIGN.md 含需求分析、架构设计、精度策略 |
| D7 | 开发计划 | ⚠️ | PLAN.md 性能验收 ⬜ 未完成（Step 6b 处理） |
| D8 | 审查报告 | ✅ | 本报告已写入 |

### 9.2 代码清洁检查（C1-C4）

| # | 检查项 | 结果 | 说明 |
|---|--------|------|------|
| C1 | printf/cout 残留 | ✅ | 仅 utils.h 中 INFO_LOG/ERROR_LOG 宏（合法日志工具） |
| C2 | TODO/FIXME 残留 | ✅ | 无残留 |
| C3 | 注释掉的代码块 | ✅ | 无大段注释代码 |
| C4 | 调试用硬编码 | ⚠️ | `HEAVY_BLOCK_NUM = 56` 死代码（RN-4，非阻塞）；BOOST_ELEM_COUNT/CACHE_FLUSH 为 perf 模式合法常量 |

### 9.3 精度全覆盖验证

算子仅支持 BF16 输出（MXFP8 输入 → BF16 输出），dtype 覆盖范围为 BF16 单类型。

| dtype | Case | M | K | N | rank | Max Abs Err | Max Rel Err | 状态 |
|-------|------|---|---|---|------|-------------|-------------|------|
| BF16 | T1 | 2048 | 3584 | 4096 | 2 | ~1.6e+4 | ~6.5e-3 | ✅ PASS |
| BF16 | T2 | 2048 | 3584 | 4096 | 4 | ~1.6e+4 | 6.94e-3 | ✅ PASS |
| BF16 | T3 | 2048 | 8192 | 4096 | 4 | 3.28e+4 | 6.21e-3 | ✅ PASS |
| BF16 | T4 | 4040 | 8192 | 4096 | 4 | 3.28e+4 | ~6.2e-3 | ✅ PASS |

---

## 10. 同步策略审查汇总

### 10.1 CrossCoreFlag（AIC↔AIV 跨核同步）

| 位置 | API | 判定 |
|------|-----|------|
| impl.h:218 | `CrossCoreSetFlag<0x2, PIPE_FIX>(0)` | ✅ AIC FixPipe 完成后通知 AIV |
| impl.h:224 | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(0)` | ✅ AIV MTE2 等待 AIC 完成 |

### 10.2 SyncAll（AIV 核间同步）

| 位置 | API | 判定 |
|------|-----|------|
| impl.h:313 | `SyncAll<true>()` (CrossDevice 后) | ✅ 确保所有 rank AIC 完成 |
| impl.h:320 | `SyncAll<true>()` (每轮 Commit 后) | ✅ 确保所有 rank 通信完成 |

### 10.3 reduce_sum.h HardEvent Flag（float32 版本）

| 行号 | Flag | 判定 | 说明 |
|------|------|------|------|
| 149-150 | V_MTE2 | ✅ 必要 | V(Duplicate/Add/Cast)→MTE2(DataCopy)，WAR on tempCType；**解决 RN-6**：形成 MTE3→V→MTE2 链 |
| 153-154 | MTE2_V | ✅ 必要 | MTE2(DataCopy)→V(Cast)，RAW on tempCType |
| 161 | PipeBarrier<V> | ✅ 必要 | Cast→Add，RAW on tempF32（同 pipe 不同 tensor） |
| 164 | PipeBarrier<V> | ✅ 必要 | Add→下一批 Cast，RAW on accumF32（同 pipe 不同 tensor） |
| 173-174 | V_MTE3 | ✅ 必要 | Cast(float→BF16)→DataCopy(out)，RAW on tempCType |
| 181-182 | MTE3_V | ✅ 必要 | DataCopy(out)→下一轮 Duplicate，间接保护 MTE3→MTE2 WAR |

**冗余率**：0/6 = 0%（所有 flag 均必要，无冗余）

---

## 11. 剩余非阻塞建议

以下问题不影响 PASS 判定，建议在后续迭代中处理：

| # | 问题 | 优先级 | 修复建议 |
|---|------|--------|----------|
| RN-1 | GET→PUT 注释不一致（20+ 处） | P1 | 全局替换注释中的 "GET 模式" → "PUT 模式" |
| RN-3 | DESIGN.md §8.3 PIPE_MTE3 → PIPE_FIX | P1 | 更新 DESIGN.md |
| RN-4 | main.cpp:283 HEAVY_BLOCK_NUM 死代码 | P2 | 删除未使用变量 |
| RN-5 | CMakeLists.txt -w 抑制警告 | P2 | 移除 -w，修复暴露的警告 |
| RN-7 | 缺 Level 0 基础测试 | P2 | 添加 M=16, K=64, N=32, rank=2 极小用例 |
| RN-8 | 无独立性能采集 | P2 | Step 6b 使用 /ops-profiling 采集 msprof 数据 |
| RN-9 | DESIGN.md 缺 float32 累加策略说明 | P2 | 在 §8 或 §9 补充 reduceSum float32 累加设计 |
| RN-10 | README 缺 API 映射表和已知限制 | P2 | 补充 Apace/Blaze API → 算子功能映射表和已知限制清单 |

---

## 12. 审查结论

### 判定：PASS

**依据**：
1. **MF-1 已修复**：`reduce_sum.h:69` 使用 `TOTAL_UB_SIZE` 框架常量，Grep 检查无硬编码（检查项 4.1 通过）
2. **MF-2 已修复**：float32 Cast 累加正确实现，所有精度用例独立验证 PASS（检查项 6.3 通过）
3. **无必须修复问题**：所有必须修复检查项（1.1/2.1/2.2/3.1/3.2/4.1/6.1）均通过
4. **总分 87 ≥ 80**：符合 PASS 标准
5. **独立编译验证通过**：Reviewer 独立编译成功（build_reviewer 目录）
6. **独立精度验证通过**：4 组测试用例全部 PASS，含 Round 0 FAIL 的 K=8192 场景

### 修复效果对比

| 指标 | Round 0 | Round 1 | 改善 |
|------|---------|---------|------|
| 总分 | 81/100 | 87/100 | +6 |
| 必须修复项 | 2 (MF-1, MF-2) | 0 | -2 |
| K=8192 4-rank 精度 | FAIL (1.09e-2) | PASS (6.21e-3) | ✅ 修复 |
| M=4040 K=8192 精度 | FAIL (1.09e-2) | PASS (~6.2e-3) | ✅ 修复 |
| UB 参数 | 硬编码 192KB | TOTAL_UB_SIZE (248KB) | ✅ 修复 |
| 同步策略 RN-6 | 存疑 | 已解决 | ✅ 修复 |
| 判定 | FAIL | **PASS** | ✅ |
