# Step 4: RunInfoX 超大结构体瘦身 [P7] + 按值传递 [P6]

## 严重度
🟠 高优先级

## 涉及文件
- `op_kernel/utils/flash_attn_common_def.h`
- `flash_attn_block_vec_noquant_gqa.h`

## 现状问题
`RunInfoX` 含 25+ 字段，估算 >160B，远超 64B cacheline。在 `CreateTask`→`CalcParams`→`ExecuteTask`→`ComputeMm1/Vec1/Mm2/Vec2` 的调用链中，RunInfoX 以引用传递，编译器因别名顾虑无法优化。

## 修改内容

### 1. 按使用阶段拆分为子结构体

```cpp
struct RunInfoCore {     // ~48B，Cube/Vec 计算热路径用
    uint32_t loop, mloop;
    uint32_t bIdx, n2Idx, gS1Idx, s1Idx, s2Idx;
    uint32_t actMSize, actVecMSize, vecMbaseIdx;
    uint64_t actS1Size;
};

struct RunInfoS2 {       // ~24B，S2 循环控制
    uint32_t actSingleLoopS2Size, actSingleLoopS2SizeAlign;
    uint32_t actMSizeAlign32;
    uint64_t actS2Size;
    bool isFirstS2Loop, isLastS2Loop, isS2SplitCore;
};

struct RunInfoMisc {     // 冷字段，仅特殊路径用
    bool isValid, isChangeBatch;
    uint32_t gIdx, faTmpOutWsPos;
    int64_t preTokensLeftUp, nextTokensLeftUp;
    uint64_t qPaddingBeginOffset, kvPaddingBeginOffset;
};

struct RunInfoX : RunInfoCore, RunInfoS2, RunInfoMisc {};
```

### 2. ProcessVec1Dn 改为按值传递小子结构体

```cpp
void ProcessVec1Dn(..., RunInfoCore ri, RunInfoS2 rs)
// 按值传递 → 无别名顾虑 → 编译器激进优化
```

- 函数内所有 `runInfo.xxx` 替换为 `ri.xxx`（Core 字段）或 `rs.xxx`（S2 字段）
- 冷路径调用 SoftmaxDataCopyOut 时从 ri/rs 重建 RunInfoX

### 3. 调用点更新

```cpp
ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo, runInfo);
// RunInfoX 继承自 RunInfoCore 和 RunInfoS2，隐式切片传递
```

## 修改统计
- 2 files changed, 43 insertions(+), 37 deletions(-)

## 实施状态
✅ 已完成（commit 82fb9bad6）
