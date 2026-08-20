# CGDR arch35 重构后数据流与改动点

> 本文档描述 arch35 CGDR 算子修正后的实现状态。
> RDV 288 用例全部 PASS。

---

## 修正状态

PR10209 的 R2/R4/R5/R7/CV 重构方案存在 6 个致命 Bug（Broadcast 方向错误、strided matmul 不可行、FIX→MTE2 竞态等），全部回退至 upstream 原始代码。仅保留 tiling workspace 修复。

| 编号 | 状态 | 说明 |
|:-----|:-----|:-----|
| R2 | **回退** | strided-A matmul 不可行：`SetOrgShape(orgKa)` 同时控制 K 迭代计数和 A 地址步进 |
| R4 | **回退** | 列缩放精度回归：BF16 舍入路径改变导致 ~47/288 超阈值 |
| R5 | **回退** | FIX→MTE2 竞态：AIC AttnInverseMMCompute(PIPE_FIX) 修改 attnWs 后 AIV 读取(PIPE_MTE2) |
| R7 | **回退** | strided-B K 不可行：transB=false 的 strided B 同样不支持 |
| CV | **回退** | 依赖 R7，一并回退 |
| tiling | **已修复** | stateFp32Wk_ + attnInter_ 分配 |

## tiling.cpp 修改

### 1. stateFp32Wk_ 分配修复

arch35 kernel 在 BF16-state 时无条件分配 `stateFp32Wk_`（FP32 中间 state workspace），但 tiling 中 `socVersion != ASCEND950` 条件跳过了 ASCEND950。

```cpp
// 修复前:
} else if (socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
    tilingData_.interWorkspaceSz += sizeHigh * b * nv * dv * dk;
}
// 修复后:
} else {
    tilingData_.interWorkspaceSz += sizeHigh * b * nv * dv * dk;
}
```

### 2. attnInter_ 分配恢复

重构删除了 `attnInter_` workspace 分配（"死代码清理"），但 arch22 兼容代码仍引用，导致 workspace 溢出。

```cpp
// 恢复:
tilingData_.interWorkspaceSz += sizeLow * nv * s * dv; // attnInter (BF16, arch22 compat)
```

### 3. stage WS 公式恢复

重构修改了 stage WS 布局（3 C×C + 2 C×Dk），回退至原始（2 C×C + 3 C×Dk + 1 C×Dv）。

```cpp
constexpr uint32_t STAGE_ONE_TWO = 2;   // 2 C×C: kkWs + attnWs
constexpr uint32_t STAGE_ONE_THREE = 3; // 3 C×Dk: gBKWs + queryCon + keyCon
// + 1 C×Dv: vBetaWs
```

## Stage1/Stage2/Stage3 计算流程

全部回退至 upstream 原始实现，无改动。详见 `chunk_gated_delta_rule_golden.py` 中的参考实现。

## 验证

```
Total: 288, PASS: 288, FAIL: 0, ERROR: 0
```

覆盖路径：gOptional/!gOptional, BF16/FP32 state, 连续/非连续, GQA, 多种 shape (S=128~262144, nv=2~64, dk/dv=1~128)
