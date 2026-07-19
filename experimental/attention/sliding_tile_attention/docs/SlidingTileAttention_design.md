# SlidingTileAttention 算子设计

## 1. 目标与范围

本实现面向任务书声明的 18 个固定 FastVideo case，在 Atlas A2/A3 上提供 solution-owned Ascend C
实现。动态 head、动态 head dimension、额外 text 长度或任意窗口模式不属于本次验收范围，入口显式拒绝。

语义保持为：

```text
softmax((Q @ K^T) / sqrt(64), masked_by_sliding_tile_window) @ V
```

## 2. 接口与调用链

```text
aclnnSlidingTileAttentionGetWorkspaceSize
  -> 固定 case 校验与 window 规范化
  -> L0 SlidingTileAttention
  -> 分配仅承载 runtime signature 的内部 tensor
  -> ADD_TO_LAUNCHER_LIST_AICORE(SlidingTileAttentionFai)
  -> op_host tiling/workspace/blockDim
  -> 单个 MIX_AIC_1_2 Ascend C kernel
```

公开接口只有一组标准两段式 ACLNN，不再按窗口组合生成 128 个 API。

## 3. 固定签名与 dispatch

L2 将合法 metadata 规范化为 `seqShape + windowPattern`。18 个输入中 case 18 的单窗口广播与完整
8-head `[3,3,3]` 列表等价，因此设备侧共有 17 个唯一签名。

| 条件 | 设备路径 | 行为 |
| --- | --- | --- |
| FP16 固定签名 | `SlidingTileAttentionFai<half,...>` | 单 kernel CATLASS QK/softmax/PV |
| BF16 固定签名 | `SlidingTileAttentionFai<bfloat16_t,...>` | 单 kernel 内部转换、PV 修正、BF16 写回 |
| 其他 metadata | 无 | L2/op_host 显式失败，不 fallback |

SoC 闭合范围为 `ascend910b` 与 `ascend910_93`：OpDef、binary config、CMake compile target 和 package
路径保持一致。

## 4. Host tiling

画布枚举映射如下：

| `seqShape` | 画布 | text | 序列长度 |
| ---: | --- | ---: | ---: |
| 1 | `30x48x80` | 128 | 115328 |
| 2 | `36x48x48` | 0 | 82944 |
| 3 | `18x48x80` | 0 | 69120 |

tile 固定为 `[6,8,8]`，每 tile 384 token。一个 query chunk 处理 128 行。`blockDim` 根据总 query
chunk 数和平台 AIC core 数取较小值。workspace 按 core 划分 5 个 FAI ring stage；BF16 额外拥有
输入 staging 和 fp32 correction/output 状态。所有区域由当前 kernel 独占，不依赖外部 Cast op。

## 5. Window 与 mask

query token 首先映射到 tile id，再还原 `(t,h,w)` 坐标。每个轴使用 `radius=window/2`，将中心限制在
`[radius, extent-1-radius]`，从而在首尾 tile 使用与 FastVideo 相同的 clamp 窗口。KV 以连续 width
segment 枚举，减少离散 GM 访问。

- image query：局部 image window；有 text 时再追加 text segment。
- text query：完整 image + text 序列。
- 非 causal，无额外上三角 mask。

## 6. Kernel 数据流

### 6.1 FP16

1. AIC 通过 CATLASS 完成 QK。
2. AIV 执行 fp32 online softmax，维护全局 row max/row sum。
3. AIC 执行 P@V，AIV 完成跨 stack rescale 和最终归一化。
4. 输出转换为 FP16 并写回 GM。

### 6.2 BF16

旧版本在设备外发射三个输入 Cast、FAI 和一个输出 Cast。当前实现移除该多 kernel 链：

1. 同一个 STA MIX kernel 在自有 UB/GM workspace 中完成 BF16 tile staging。
2. dense PV 使用 kernel 内部 half 计算，同时 AIV 以 fp32 计算 row mean 与 `sum(V)` 修正项。
3. 使用恒等式 `(P-rowMean(P))@V + rowMean(P)*sum(V)` 恢复 BF16 目标精度。
4. 最终 fp32 累积一次性转回 BF16 并写回。

这里的内部表示转换是同一次 STA kernel launch 的私有数据流，不是独立 Cast 算子，也没有 vendor
或 framework whole-op 替代路径。

## 7. 同步与数据所有权

每个 core 拥有独立 ring workspace。AIC 发布只读 `oTmp`，AIV 负责 recurrence、correction、归一化和
copy-out；event id 使用命名常量并限制在 FFTS 消息编码范围。输出行按 batch/head/query chunk 分配，
不存在跨 core 重叠写。

## 8. 验证

- Host guard：17 个规范化签名全部接受，固定任务表之外的代表性 metadata 签名全部拒绝。
- Repository package：A2/A3 均生成 FP16/BF16 kernel artifact。
- Reviewer runner：只使用 PR 内 C++ 代码生成并运行固定 18 case。
- 迁移前冻结源码在 Ascend910_9382（physical=6, card=3, chip=0, logical=npu:0）通过 OpForge 18/18；
  该记录用于说明迁移来源，最终 PR 验收以当前源码仓 fresh package/runner 日志为准。

## 9. 无 fallback 边界

生产路径只调用当前目录构建出的 ACLNN custom op。源码中没有 Python reference、CPU、PyTorch、
FastVideo、Triton/CUDA、vendor whole-op、peer 或其他 backend fallback；不支持的 metadata 直接失败。
