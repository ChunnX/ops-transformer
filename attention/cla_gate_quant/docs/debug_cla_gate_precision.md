# CLA Gate MX Quant 精度问题定位方案

本文记录如何系统定位 `cla_gate_mx_quant_with_dual_axis` 的精度问题，避免“猜测式调试”。

## 1. 问题链路

```text
CopyIn:
  global_out / local_out
  global_gate / local_gate
        |
        v
ComputeClaGateFullTile:
  sigmoid(global_gate)
  sigmoid(local_gate)
  merged = global_out * sigmoid(global_gate)
         + local_out * sigmoid(local_gate)
        |
        v
ComputeScaleOcp / ComputeScaleCuBLAS:
  MX scale 计算
        |
        v
ComputeY1/Y2ToFP8:
  使用 reciprocal 量化
        |
        v
CopyOut -> GM
```

精度问题可能出现在任意一段，因此必须先隔离。

## 2. 推荐隔离顺序

### 2.1 先拿到同一份输入

TTK dump 输入与 kernel 打印必须在**同一次运行**中产生，否则无法对比。

```bash
TTK_TESTCASE=ClaGateMxQuantWithDualAxis_bf16_ocp_random \
    bash tests/st/arch35/run_ttk_kernel.sh \
        --dump full --dump-format npy
```

产物：

```text
*_dyn_input_0.npy   # global_out
*_dyn_input_1.npy   # local_out
*_dyn_input_2.npy   # global_gate_logits
*_dyn_input_3.npy   # local_gate_logits
*_dyn_output_*.npy  # 实际输出
*_golden_*.npy      # golden 输出
```

### 2.2 用 fixed-scale 隔离 merge 和 MX scale

思路：旁路 MX scale，将 reciprocal 设为 1.0，输出变成：

```text
row_data = FP8(merged)
col_data = FP8(merged)
row_scale = 1.0
col_scale = 1.0
```

若 fixed-scale 通过，说明 `ComputeClaGateFullTile` 基本正确；
若 fixed-scale 失败，则问题在 merge。

### 2.3 用 deterministic mapping 隔离 lane/row 映射

用固定输入替代随机输入，例如：

```text
global_out = 1
local_out  = 1
global_gate(row) = (row - 32) * 0.25
local_gate(row)  = (row - 32) * 0.25
merged(row) = 2 * sigmoid(gate(row))
```

这个 case 能暴露：

- sigmoid 是否算对；
- sigmoid buffer 行顺序是否对；
- global/local 双路是否被存储覆盖。

### 2.4 直接打印 kernel 中间 buffer

在 `ComputeClaGateFullTile` 前后打印：

```text
globalOut row0
localOut row0
globalGate rows0-7
localGate rows0-7
globalSigmoid row0-7
localSigmoid row0-7
mergedU16 row0
```

将打印值转成 bf16/float 与 golden 对比。

### 2.5 对比 golden merged

读取同一次 dump 的输入，按 golden 公式计算 merged：

```python
global_scale = sigmoid(global_gate).unsqueeze(-1)
local_scale  = sigmoid(local_gate).unsqueeze(-1)
merged = global_out * global_scale + local_out * local_scale
merged = merged.reshape(S*B, N*D)
```

再与 kernel 打印的 `mergedU16` 对比。

## 3. 关键结论（截至当前调试）

目前已确认：

- `globalOut` / `localOut` / `globalGate` / `localGate` 打印与 npy 输入一致；
- `globalSigmoid` / `localSigmoid` 打印正确；
- 但 `mergedU16` 与 golden merged 不一致。

因此问题定位到：

```text
ComputeClaGateFullTile 的 Phase2:
  attention 读取 / sigmoid 广播 / Mul / Add
```

而不是 CopyIn 或 sigmoid 计算。

## 4. 下一步建议：重点检查 Phase2

当前 Phase2 伪代码：

```cpp
for (row = 0; row < 64; ++row) {
    // 正确 sigmoid 已保存在 buffer
    Load globalSigmoidBroadcastF;
    Load localSigmoidBroadcastF;

    for (vf = 0; vf < vfsPerRow; ++vf) {
        AddrReg offset = CreateAddrReg<xDtype>(row, ONCE_ROW_LEN,
                                               vf, VF_LEN_FP32);

        LoadAlign(globalReg, globalAddr, offset);
        Cast(globalF, globalReg);
        Mul(outputF, globalF, globalSigmoidBroadcastF);

        LoadAlign(localReg, localAddr, offset);
        Cast(localF, localReg);
        Mul(localF, localF, localSigmoidBroadcastF);

        Add(outputF, outputF, localF);
        Cast(outputReg, outputF);
        StoreAlign(outputAddr, outputReg, offset);
    }
}
```

重点怀疑：

1. `CreateAddrReg` 的维度/stride 是否与当前 `LoadAlign` 语义匹配；
2. `AddrReg` 是否被错误地复用到 global/local 两路；
3. `DIST_UNPACK_B16` + `Cast<float>` 在当前 Reg 版本下读取的 lane 是否与 `AddrReg` 偏移一致；
4. `Mul/Add` 是否因为寄存器复用导致 global/local 互相覆盖。

## 5. 如何验证 AddrReg/Phase2

方案 A：先绕过 `AddrReg`，用“显式指针偏移”读取 attention：

```cpp
// 每 vf 用元素偏移自己计算
__ubuf__ xDtype* globalRow = globalAddr + row * ONCE_ROW_LEN;
LoadAlign<xDtype, DIST_UNPACK_B16>(globalReg, globalRow + vf * VF_LEN_FP32);
```

如果显式指针版 merged 正确，则说明 `AddrReg` 使用有问题。

方案 B：打印 Phase2 中加载后的 attention：

```cpp
// 只打印 row0, vf0 前 8 个，与 golden globalOut row0 对比
```

如果 Phase2 读出的 attention 与输入不一致，则确认是地址/寻址问题。

## 6. TTK 常用调试命令

```bash
# dump 全量数据
... --dump full --dump-format npy

# 失败自动 dump
... --dump-on-fail --dump-format npy

# 固定随机种子
... --seed 42
```

分析 dump 脚本：

```bash
python3 tests/assets/analyze_ttk_dump.py \
    /home/g00940185/ops-test-kit/ClaGateMxQuantWithDualAxis_bf16_ocp_random
```

## 7. 注意事项

1. 每次修改 kernel 后，需要同步两份文件：
   - 源码目录
   - CANN vendor 编译目录
2. kernel 打印和 npy dump 必须在同一次运行中产生。
3. AscendC `printf` 对浮点格式支持有限，建议打印整数/hex：
   - bf16/fp16：打印 `uint16_t`
   - fp32：打印 `uint32_t`
4. 不要使用 `%f`、`%04x` 等不确定格式；先用 `%x`。

## 8. 具体在哪里加打印

### 8.1 需要修改的文件

必须同时修改两份，否则编译可能用旧代码：

1. 源码：
```text
op_kernel/arch35/cla_gate_mx_quant_with_dual_axis.h
```

2. CANN vendor 编译路径：
```text
/home/g00940185/work/CANN9.2.0/cann-9.2.0/opp/vendors/custom_nn/op_impl/ai_core/tbe/custom_nn_impl/ascendc/cla_gate_mx_quant_with_dual_axis/arch35/cla_gate_mx_quant_with_dual_axis.h
```

### 8.2 在文件顶部增加调试宏

在类定义之前加：

```cpp
// ===== TEMP DEBUG =====
#define CLA_GATE_DEBUG_PRINT_MERGED 1
```

### 8.3 打印输入：在 `ComputeClaGateFullTile` 调用之前

定位在 `Process()` 中这段代码之后：

```cpp
auto mergedAddr = (__ubuf__ xDtype*)merged.GetPhyAddr();
```

在 `ComputeClaGateFullTile(...)` 之前插入：

```cpp
#if CLA_GATE_DEBUG_PRINT_MERGED
        if (blockIdx_ == 0 && i == 0) {
            auto inputU16 = input.template ReinterpretCast<uint16_t>();

            AscendC::printf("globalOut row0:");
            for (int32_t dbgIdx = 0; dbgIdx < 32; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(inputU16.GetValue(dbgIdx)));
            }
            AscendC::printf("\n");

            AscendC::printf("localOut row0:");
            for (int32_t dbgIdx = 0; dbgIdx < 32; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(inputU16.GetValue(inHalfSize_ + dbgIdx)));
            }
            AscendC::printf("\n");

            AscendC::printf("globalGate rows0-7:");
            for (int32_t dbgIdx = 0; dbgIdx < 8; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(inputU16.GetValue(2 * inHalfSize_ + dbgIdx)));
            }
            AscendC::printf("\n");

            AscendC::printf("localGate rows0-7:");
            for (int32_t dbgIdx = 0; dbgIdx < 8; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(inputU16.GetValue(2 * inHalfSize_ + ubRowCount_ + dbgIdx)));
            }
            AscendC::printf("\n");
        }
#endif
```

### 8.4 打印 sigmoid：在 `ComputeClaGateFullTile` 调用之后

在 `ComputeClaGateFullTile(...)` 之后插入：

```cpp
#if CLA_GATE_DEBUG_PRINT_MERGED
        if (blockIdx_ == 0 && i == 0) {
            auto sigU32 = sigmoidBroadcastBuf_.template Get<float>().template ReinterpretCast<uint32_t>();

            AscendC::printf("globalSigmoid row0-7:");
            for (int32_t dbgIdx = 0; dbgIdx < 8; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(sigU32.GetValue(dbgIdx)));
            }
            AscendC::printf("\n");

            AscendC::printf("localSigmoid row0-7:");
            for (int32_t dbgIdx = 0; dbgIdx < 8; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(sigU32.GetValue(ubRowCount_ * VF_LEN_FP32 + dbgIdx)));
            }
            AscendC::printf("\n");
        }
#endif
```

### 8.5 打印 merged：也在 `ComputeClaGateFullTile` 之后

继续插入：

```cpp
#if CLA_GATE_DEBUG_PRINT_MERGED
        if (blockIdx_ == 0 && i == 0) {
            auto mergedU16 = merged.template ReinterpretCast<uint16_t>();
            AscendC::printf("mergedU16 rowStart=%d colOff=%d:", absRowStart, colOffset);
            for (int32_t dbgIdx = 0; dbgIdx < 64; ++dbgIdx) {
                AscendC::printf(" %x", static_cast<uint32_t>(mergedU16.GetValue(dbgIdx)));
            }
            AscendC::printf("\n");
        }
#endif
```

### 8.6 打印注意点

- 必须限制：
  ```cpp
  if (blockIdx_ == 0 && i == 0)
  ```
  否则 64 个核会打印大量内容。
- 不要用 `%f`，AscendC `printf` 对浮点支持差，优先打印 raw hex。
- 不要用 `%04x` 这类宽度格式，先用 `%x`。
- bf16/fp16 通过 `ReinterpretCast<uint16_t>` 打印；
- fp32 通过 `ReinterpretCast<uint32_t>` 打印。

## 9. 最终定位结论

通过打印确认的根因有两个：

### 9.1 缺少 `LocalMemBar`（主要问题）

`ComputeClaGateFullTile` 中：

1. Phase1 将 sigmoid 从 `RegTensor` StoreAlign 到 UB；
2. Phase2 再从同一个 UB LoadAlign 读回 sigmoid；
3. 同一 `__VEC_SCOPE__` 内，`VEC_STORE -> VEC_LOAD` 必须加本地 barrier。

修复：

```cpp
AscendC::Reg::LocalMemBar<
    AscendC::Reg::MemType::VEC_STORE,
    AscendC::Reg::MemType::VEC_LOAD>();
```

### 9.2 sigmoid 必须先落回 bf16/fp16

golden 使用：

```python
global_scale = torch.sigmoid(global_logits)  # bf16/fp16
local_scale  = torch.sigmoid(local_logits)   # bf16/fp16
merged = global_out * global_scale + local_out * local_scale
```

因此 kernel 必须模拟窄精度逐步舍入：

- sigmoid 在 FP32 计算后，立刻 cast 成 xDtype 再存 UB；
- Phase2 从 UB 加载 xDtype sigmoid，再转 FP32 乘；
- 乘法结果 cast 回 xDtype；
- 加法结果再 cast 回 xDtype。

最终修复后的 kernel 已通过精度验证。
