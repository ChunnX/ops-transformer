# 0721 待优化项

## C1 Cube 双缓冲与 MMA/Fixpipe 重叠流水

### 状态

- 当前仅记录方案，暂不修改 `0721.py`。
- 前置条件：全量泛化测试的精度问题全部解决并形成稳定基线。
- 实施时必须保留当前精度正确版本和对应生成的 AscendC C++，一次只改一个流水层级。

### 后端新增能力

当前 TileLang-Ascend 已支持由 kernel 显式驱动分块矩阵流水：

```python
T.copy(src_l1, dst_l0a, real_k=real_k)
T.copy(src_l1, dst_l0b, transpose=True, real_k=real_k, real_n=real_n)
T.mma(
    a_l0,
    b_l0,
    c_l0,
    init=init,
    k_actual=real_k,
    n_actual=real_n,
    unit_flag=unit_flag,
)
T.copy(c_l0, output_region, unit_flag=0b11)
```

参数约束：

- A 侧 L1 到 L0A copy 使用 `real_k`。
- B 侧 L1 到 L0B copy 使用 `real_k` 和 `real_n`。
- 同一块的 `real_k/real_n` 必须分别等于 MMA 的 `k_actual/n_actual`。
- 中间 MMA 使用 `unit_flag=0b10`，结果继续保留在 L0C 中累加。
- 最后一次 MMA 使用 `unit_flag=0b11`，紧接着的 L0C 到 GM copy 也使用 `unit_flag=0b11`。
- 最后一次 MMA 与配对 copy 之间不插入 `M_FIX/FIX_M` 同步。

`real_k/real_n` 负责描述实际矩阵分形形状，本身不创建流水。真正的重叠依赖双物理槽和正确的 event ownership。

### 0721 的适用位置

优先目标是 C1-P。当前每个 `128` 列输出块需要将 Query 的四个 `K_SPLIT=128` 分块以及可选 Rope 分块连续累加到同一个 L0C，和新增 kernel-driven MMA 的使用场景一致。

期望稳态流水：

```text
MTE2：预取第 n+2 块 GM -> L1
MTE1：搬运第 n+1 块 L1 -> L0A/L0B
M：   计算第 n 块 MMA
FIX： 写回第 n-1 个输出块
```

当前 C1-P 已实现下一块 GM 到 L1 的单槽预取，但 L1 到 L0 与 MMA 仍主要串行。新增能力主要用于补齐以下两层重叠：

1. 双 L0A/L0B：下一 K 分块的 L1 到 L0 搬运与当前 MMA 重叠。
2. 双 L0C：当前输出块的 Fixpipe 写回与下一输出块的 MMA 重叠。

C1-SY 可在 C1-P 验证稳定后继续评估。C2 当前耗时较短，优先级低于 C1。

### 第一阶段实施范围

第一版只针对固定完整块实现，不同时引入动态尾块优化：

```text
real_k = k_actual = 128
real_n = n_actual = 128
```

0721 的尾部目前依靠零 gather、完整 `128` 列 workspace 和下游 mask 保证正确。不能直接把 `n_actual` 改为 `kRealSize` 的剩余列数，否则完整 workspace 中未写出的列可能残留旧数据。动态 `real_n/n_actual` 应在固定块流水稳定后单独验证。

### 同步与内存约束

- `TL_ASCEND_AUTO_SYNC` 继续保持 `False`。
- `unit_flag` 只替代 MMA 与 FIX 之间的 ownership，不替代 L0A/L0B 的 `M_MTE1` 和 `MTE1_M` 协议。
- 不能同时保留旧的 `M_FIX/FIX_M` 协议并给同一 L0C 操作增加 `unit_flag`，否则可能形成两套冲突的 ownership。
- L0A、L0B、L0C 必须使用物理精确的 ping/pong 槽，不能通过更大父 buffer 的不精确切片模拟不同矩阵形状。
- C1-P 与 C1-SY 的 L0 总容量可能超过硬件上限，必须确认 memory planning 能在两个顺序 phase 之间复用物理地址，或者将 allocation 生命周期拆分到可复用的独立阶段。
- 每个槽必须完成初始化、ready 发布、free 归还和结束 drain，不能在 event 未回收时进入下一 phase。

### 推荐实施顺序

1. 编写与 C1-P 相同形状的最小两块 K 累加验证，确认新参数的生成 C++ 和设备精度。
2. 只增加双 L0A/L0B，验证 MTE1 与 MMA 重叠，不改 L0C 写回协议。
3. 增加双 L0C，使用 `unit_flag=0b10/0b11` 配对 MMA 和 Fixpipe。
4. 集成到 C1-P 的 full-valid 版本，完成重复精度和性能测试。
5. 验证泛化 shape 后，再评估 C1-SY 和 tail-protected 版本。

### 生成 C++ 验收条件

必须确认：

- L0A/L0B 使用两个不同且对齐的物理 offset。
- 若启用双 L0C，两个 L0C offset 的容量、边界和 MMA/FIX shape 均正确。
- `copy_l1_to_l0a/b` 的实际 `real_k/real_n` 与对应 MMA 一致。
- 中间 MMA 生成 `unitFlag=0b10`，最后 MMA 和紧随其后的 copy 生成 `unitFlag=0b11`。
- 最后 MMA 与配对 copy 之间没有旧的 `M_FIX/FIX_M` 等待。
- L0A/L0B 仍保留完整的 `M_MTE1`、`MTE1_M` ping/pong ownership 和结束 drain。
- timeline 中真实出现 MTE2、MTE1、MMA、FIX 的重叠，而不只是源码中存在两个逻辑 buffer。

### 收益预期

这是当前最值得重新评估的 Cube 结构优化之一，有机会减少 C1 中的 MTE1 和 FIX 等待。预期收益应以设备 timeline 和重复中位数为准，不能仅依据新增 API 或单次端到端时间判断，也不预设该改动能够单独追平官方 AscendC 性能。
