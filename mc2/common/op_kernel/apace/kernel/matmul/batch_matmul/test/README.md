# AllToAllMmKernel Kernel UT 测试

对 `../all_to_all_mm_kernel.h`（非量化 all-to-all MatMul，splitKIdx 版）的 Kernel UT 测试，验证三种调度模式下的矩阵乘、L0C 跨 rank 累加（splitKIdx + 单次 fixpipe）与 bias 语义的正确性。

## 测试原理

在单进程 ICPU 模拟器环境中：

1. **预填通信 buffer**：`remoteAGM` 装入全部 rank 的 A 数据（`rankSize * originalM × K`），`localAGM` 装入 self rank 的 A 段；CommPolicy 用 `NoopCommWaitPolicy`（`WaitTile` 空实现），模拟通信已完成，聚焦 kernel 计算逻辑本身。
2. **三种模式分别验证**：
   - 模式1（`localMatmul=0, matmulMode=REMOTE`）：单次 launch，遍历所有 rank，splitKIdx=0..rankSize-1 在 L0C 累加，末次单次 fixpipe（无 atomic）；
   - 模式2（`localMatmul=1, matmulMode=DEFERRED_SYNC`）：单次 launch，self 先算驻留 L0C，wait 后其余 rank 累加，末次单次 fixpipe；
   - 模式3（`localMatmul=2`）：LOCAL launch（覆盖写 + bias，splitKNum=1）后 REMOTE launch（跳过 self，Run 级 atomic 累加）。
3. **参考实现**：
   ```
   C[m][n] = bias[n] + Σ_{r=0}^{rankSize-1} Σ_{k=0}^{K-1} A[r*originalM+m][k] * B[r*K+k][n]
   ```
   在 host 侧用 fp32 累加计算，与 kernel 输出（fp16）逐元素比对，容差 `0.05 + 0.01*maxRef`。

## 用例覆盖

| 用例 | rankSize | oriM | K | N | blockNum | 模式 |
|---|---|---|---|---|---|---|
| Mode1_Remote_MultiRank | 2 | 16 | 16 | 16 | 1 | 模式1 单 launch |
| Mode2_DeferredSync_MultiRank | 2 | 16 | 16 | 16 | 1 | 模式2 单 launch |
| Mode3_LocalRemote_MultiRank | 2 | 16 | 16 | 16 | 1 | 模式3 两 launch |
| Mode1_Remote_MultiCore | 2 | 32 | 16 | 32 | 4 | 模式1 多核并行 |
| Mode1_Remote_SingleRank | 1 | 16 | 16 | 16 | 1 | 退化单 rank |
| Mode2_DeferredSync_SingleRank | 1 | 16 | 16 | 16 | 1 | 退化单 rank |

## 接入与运行

本测试依赖 ops-tensor 的 Kernel UT 构建框架（ICPU 模拟器 + gtest）。将本目录接入 ops-tensor：

```bash
# 1. 复制到 ops-tensor Kernel UT 目录（或软链接）
cp -r <ops-transformer>/mc2/common/op_kernel/apace/kernel/matmul/batch_matmul/test \
      <ops-tensor>/tests/ut/op_kernel/all_to_all_mm_kernel

# 2. 确保 OPS_TRANSFORMER_ROOT 指向 ops-transformer 仓库根目录
#    （默认取 ops-tensor 的兄弟目录 ../ops-transformer，也可显式指定）

# 3. 先初始化 tensor_api 子模块（ops-tensor 构建 Kernel UT 的前置）
cd <ops-tensor>
git submodule update --init --recursive

# 4. 构建并运行
source <CANN安装目录>/set_env.sh
./build.sh --opkernel -u --ops=all_to_all_mm_kernel
```

也可不加 `--ops` 构建全部 Kernel UT。

## 注意事项

- 需在安装了 CANN（含 tikicpulib）且已初始化 `include/tensor_api` 子模块的环境中运行。
- ICPU 模拟器需支持 `SetAtomicAdd` / `SetAtomicNone` 的语义模拟（`block_mmad_matmul_basic_split_k.h` 已在使用同类原语）。若模拟器不支持原子累加，需在真实 NPU 上验证模式1/2/3 的累加路径。
- 测试数据确定性生成，无外部 `.bin` 依赖。
