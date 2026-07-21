# FusedFloydAttention A3→A5 遗留问题

> 算子：`ops-transformer/attention/fused_floyd_attention`
> 架构：DAV_2201（A3）→ DAV_3510（A5）

## 1. `DropOutBitModeInit` A5 缺失

### 现象

A5 编译报错：
```
error: use of undeclared identifier 'DropOutBitModeInit'
    DropOutBitModeInit(apiTmpBuffer);
```

出现位置：
- `op_kernel/fused_floyd_attention_s1s2_bn2gs1.h:358`
- `op_kernel/fused_floyd_attention_s1s2_bn2gs1special.h:271`

### 根因

`DropOutBitModeInit` 定义在 CANN 库的 `dropout_membase_impl.h:32-47`（SIMD Membase 架构实现），用于初始化 cmp mask 状态：

```cpp
template <typename T>
__aicore__ inline void DropOutBitModeInit(const LocalTensor<T>& sharedTmpBuffer)
{
    ResetMask();
    LocalTensor<int16_t> stackBuffer = sharedTmpBuffer.template ReinterpretCast<int16_t>();
    Muls<int16_t, false>(stackBuffer, stackBuffer, 0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    SetCmpMask<int16_t>(stackBuffer);
    PipeBarrier<PIPE_V>();
}
```

CANN 库按 `__NPU_ARCH__` 条件编译 dropout 实现（`dropout_impl.h:25-34`）：

| `__NPU_ARCH__` | 包含文件 | 是否提供 `DropOutBitModeInit` |
|---|---|---|
| 1001 / 2002 | `dropout_m200_impl.h` | ✓（经 `dropout_membase_impl.h`） |
| 2201（A2/A3） | `dropout_c220_impl.h` | ✓（经 `dropout_membase_impl.h`） |
| 3002 | `dropout_m300_impl.h` | ✓（经 `dropout_membase_impl.h`） |
| 3510 / 3003 / 3113（A5） | `dropout_3510_impl.h` | **✗**（Regbase 重写，不依赖 `SetCmpMask`） |

A5 的 `DropOutBitMode`（`dropout_3510_impl.h:268-293`）改用 Regbase 寄存器级 API（`Reg::RegTensor`/`Reg::MaskReg`/`Reg::Select`），初始化通过 `Reg::Duplicate(scalarReg, 0)` 内嵌完成，不再需要独立的 `DropOutBitModeInit`。

### 影响评估

- 本算子 `hasDrop` 模板参数固定为 `false`（`fused_floyd_attention.cpp:126,131,136,141`），dropout 代码路径永不执行
- 原代码无条件调用 `DropOutBitModeInit` 是 A2/A3 的预防性初始化，对计算结果无影响
- A5 上跳过此调用不改变算子逻辑，不影响功能正确性

### 处理方式

用架构宏条件编译，A2/A3 保留原调用，A5 跳过：

```cpp
LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
#if defined(__DAV_C220_CUBE__)
    DropOutBitModeInit(apiTmpBuffer);
#endif
```

### 遗留风险

- 若后续算子需要支持 dropout（`hasDrop=true`），A5 上不能直接调用 `DropOutBitModeInit`，需改用 A5 Regbase 风格的 dropout 实现
- 当前本算子 README 约束 `keepProb` 默认 1.0，不支持 dropout，遗留风险可接受
