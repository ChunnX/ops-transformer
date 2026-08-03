# dropPadMode 参数详解

> 关联文档：`operator_functionality.md`、`aclnnMoeInitRoutingV3.md`
> 适用算子：`aclnnMoeInitRoutingV3`

## 1. 一句话理解

`dropPadMode` 决定 **每个专家能处理多少个 token**：是「来多少处理多少」（Dropless），还是「固定容量，超出丢弃、不足补零」（DropPad）。

这是 MoE 中两种经典的负载/显存控制策略。

## 2. 取值定义

| 取值 | 常量名 | 模式名 | 含义 |
|:-:|------|------|------|
| **0** | `DROP_LESS` | Dropless | 不丢不补，每个专家处理它实际分到的 token 数 |
| **1** | `DROP_PAD` | DropPad | 固定容量 `expertCapacity`，超出 drop、不足 pad 0 |

## 3. 两种模式对比

| 模式 | dropPadMode | 每个专家处理量 | 输出 expandedXOut shape | 浪费显存 | 负载均衡 |
|------|:-:|------|------|:-:|:-:|
| **Dropless** | 0 | 等于它实际分到的 token 数（不丢不补） | `[NUM_ROWS*K, H]` 或 `[min(activeNum, NUM_ROWS*K), H]` | 紧凑，无浪费 | 不均衡：热门专家处理多，冷门专家处理少 |
| **DropPad** | 1 | 固定 = `expertCapacity`（超出 drop，不足 pad 0） | `[expertNum, expertCapacity, H]` | 有 padding 浪费 | 强制均衡：所有专家处理量相同 |

## 4. Dropless（`dropPadMode=0`）的工作方式

- 排序后按 expert 连续排布 token，每个 expert 拿到它实际应该处理的 token 数。
- 输出是「紧凑」的 2D：`[NUM_ROWS*K, H]`。
- `activeNum` 可限制总行数（`min(activeNum, NUM_ROWS*K)`），实现 active routing。
- 没有浪费，但下游专家计算时每个 expert 处理量不同，存在负载不均衡。

**示例**：`NUM_ROWS=4, K=2, expertNum=4`，假设排序后：
```
expert 0: 3 个 token
expert 1: 1 个 token
expert 2: 2 个 token
expert 3: 2 个 token
```
输出 `[8, H]`，每个 expert 段长度不同。

## 5. DropPad（`dropPadMode=1`）的工作方式

- 每个 expert 固定分配 `expertCapacity` 个槽位。
- 排序后，**前 `expertCapacity` 个 token** 被保留，**超出的丢弃**（drop）。
- **不足 `expertCapacity` 的位置补零**（pad）。
- 输出是 3D：`[expertNum, expertCapacity, H]`。
- 每个 expert 处理量相同，下游可做均衡并行，但浪费显存和算力（pad 的部分是无效计算）。

**示例**：`NUM_ROWS=4, K=2, expertNum=4, expertCapacity=2`，排序后：
```
expert 0: 3 个 token → 保留前 2 个，丢第 3 个
expert 1: 1 个 token → 保留这 1 个，第 2 槽位补 0
expert 2: 2 个 token → 刚好满
expert 3: 2 个 token → 刚好满
```
输出 `[4, 2, H]`，每个 expert 都是 2 行（实际/补零）。

## 6. 关键代码佐证

`op_kernel/moe_v3_gather_out_droppad.h:110-114` 体现了 drop 的逻辑：

```cpp
int32_t outIndex = indicesLocal.GetValue(curLoopRow);
curLoopRow++;
initialRow++;
if (outIndex == -1) {
    continue;  // -1 表示该位置在 DropPad 中被 drop 掉，跳过搬运
}
outOffset_ = outIndex * cols_ + colsLoop * perLoopCols_;
DataCopyPad(expandedXGm_[outOffset_], inLocal, intriParams);
```

排序阶段会为每个 `(expert, capacity_slot)` 生成一个输出索引，超出 capacity 的位置标记为 `-1`（drop），不足的位置索引有效但对应数据由初始的零填充（pad）。

## 7. DropPad 模式的额外约束

从 `op_host/moe_init_routing_v3_tiling.cpp:363-379`（`CheckAttr`）与算子文档可见：

| 约束 | 原因 |
|------|------|
| `expertCapacity` 必须 `(0, NUM_ROWS]` | 容量必须为正且不超过总行数 |
| `rowIdxType` 仅支持 0（gather） | 输出是 3D，需要 gather 索引才能紧凑映射 |
| `activeExpertRangeOptional` 必须 `[0, expertNum]` | 不允许部分专家范围 |
| `expertTokensNumType` 仅支持 1（count） | DropPad 下统计语义不同 |
| A3：`quantMode` 仅支持 -1 | A3 模板限制 |
| A5：`quantMode=-1`，dtype 仅 FP16/BF16/FP32/INT8/HIF8 | A5 模板限制 |

## 8. 选型建议

| 场景 | 推荐模式 |
|------|---------|
| 训练大模型，追求显存效率、负载不均衡可接受 | **Dropless（0）** |
| 推理服务，要求每个 expert 计算量固定、可并行均衡 | **DropPad（1）** |
| Token 严重分布不均（部分 expert 远超 capacity） | DropPad + 合适的 `expertCapacity`，配合 `activeNum` 避免丢太多 |
| 需要量化输出 | Dropless（A3/A5 都支持量化，DropPad 几乎不支持） |

## 9. 一句话总结

- **`dropPadMode=0`**：动态长度，「来的全是客」，输出紧凑 2D。
- **`dropPadMode=1`**：固定容量，「超员拒之、不足补零」，输出对齐 3D，换取下游负载均衡与固定 shape。
