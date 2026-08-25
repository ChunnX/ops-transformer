# MoeDistributeDispatchV2/CombineV2 层次通信动态 BS 设计

## 1. 文档目的

本文说明 Atlas A3 上 `MoeDistributeDispatchV2`/`MoeDistributeCombineV2` 在
`commAlg="hierarchy"` 场景支持各 EP rank 输入不同 BS 的设计，包括需求边界、现有实现可复用能力、
代码修改、参数约束、兼容性、风险和验收方案。

本文中的“动态 BS”特指：同一次 EP 通信中，不同 rank 的输入 `x`/`expertIds` 第 0 维可以不同，
而不是在一个 Tensor 内通过 padding 和 `xActiveMaskOptional` 表示不同有效长度。

## 2. 需求背景

修改前，hierarchy 路径要求每个 rank 的输入 BS 与 `globalBS / epWorldSize` 相等。上层框架为了满足该约束，
需要在模型推理前增加一次 DP 间通信，先收集各 DP rank 的 token 数，再将输入 padding 到统一 BS。虽然
`xActiveMaskOptional` 可以避免 padding token 进入 MoE 计算，但额外的 DP 通信仍会增加时延，在多机场景中
性能劣化可超过 10%，同时增加上层框架的维护成本。

本需求希望取消这次前置 DP 对齐：各 rank 直接以本地实际 BS 调用 Dispatch/Combine，算子通过已有的层次通信
结束协议识别实际数据量。

## 3. 设计目标与边界

### 3.1 目标

- Atlas A3、`commAlg="hierarchy"` 支持不同 rank 使用不同的实际 BS。
- 保持 DispatchV2 与 CombineV2 的配套通信及元数据语义一致。
- 保持原 hierarchy 场景的数据类型、数据格式、精度和确定性语义不变。
- 所有 rank BS 相同时，原有调用方式和通信主流程不变，性能不得劣化。
- 使用统一的容量参数完成输出、元数据和 HCCL Window 空间规划，不增加运行时 rank 间 BS 协商。

### 3.2 支持范围

| 项目 | 范围 |
|---|---|
| 硬件 | Atlas A3，SoC `Ascend910_93` |
| 通信算法 | `commAlg="hierarchy"` |
| `epWorldSize` | `[16, 256]`，且为 16 的整数倍 |
| `moeExpertNum` | `(0, 512]` |
| `sharedExpertNum` | `0` |
| 输入格式 | ND |
| 输入数据类型 | 与原 hierarchy 版本一致，包含 FP16/BF16 |
| 单 rank BS 容量 | `[1, 256]` |
| 确定性 | 默认支持，行为与原 hierarchy 场景一致 |

### 3.3 非目标

本需求不新增以下能力：

- 二维 `xActiveMaskOptional`；
- 动态 BS 与 `xActiveMaskOptional` 同时使用；
- 共享专家、零专家、拷贝专家、常量专家等特殊专家；
- `performanceInfoOptional` 性能打点；
- TP 通信、弹性缩容；
- A2 或非 hierarchy 路径的动态 BS 能力扩展。

`performanceInfoOptional` 是否传入 Tensor 直接决定算子侧 `isPerformance`。因此 hierarchy 用例必须不传该
可选 Tensor；测试框架中的计时开关不能替代这一配置。

## 4. 参数与容量模型

为了避免在算子执行前增加 rank 间通信，动态 BS 使用“统一容量、各 rank 实际长度独立”的模型。

定义：

- `localBs[r]`：rank `r` 本次调用的实际 BS，即本地 `x`/`expertIds` 第 0 维；
- `maxBs`：业务配置的单 rank BS 容量上限；
- `globalBS = maxBs * epWorldSize`：EP 域统一容量，不表示各 rank 实际 BS 的求和；
- `A`：本 rank 可能接收的最大 token 数，按 `globalBS` 和专家布局计算。

动态 BS 调用必须满足：

```text
0 < localBs[r] <= maxBs <= 256
globalBS = maxBs * epWorldSize
globalBS % epWorldSize = 0
所有 rank 的 globalBS 相同且非 0
```

例如，`epWorldSize=16`，各 rank 实际 BS 分别为 8、12、20 等，业务为单 rank 预留的最大容量为 32，
则所有 rank 均传入 `globalBS=32*16=512`。每个 rank 的输入 Tensor 仍使用自己的实际 BS，不需要 padding。

当所有 rank BS 相同时，允许继续传 `globalBS=0`。Tiling 会将其转换为
`localBs * epWorldSize`，保持原有调用方式。

## 5. 可行性分析

### 5.1 hierarchy Dispatch 已具备变长接收协议

hierarchy Dispatch 并不是按固定 BS 一次性收齐远端数据，而是按 server 发送实际需要的 token，并在该 server
发送完成后写入 `EndFlag`：

```text
发送端：实际 token 0 ... token N-1 -> EndFlag

接收端：
  1. 检查下一个 token 的 TokenFlag；
  2. 已到达：处理当前 token；
  3. 未到达：检查该 server 的 EndFlag；
  4. EndFlag 未到达：继续等待；
  5. EndFlag 已到达：重新检查当前 TokenFlag；
  6. 当前 token 存在则处理，否则该 server 接收结束。
```

接收端为每个 server 独立统计 `realBS`，后续 inner count/offset 元数据也按该实际值生成。因此不同源 server
发送不同数量的 token 不会破坏通信状态机。

当前 A3 实现对“下一个 TokenFlag”“EndFlag”“当前 TokenFlag”分别执行搬运和同步，动态 BS 修改没有改变
这段协议，也没有引入 PR 5540 所描述的 TokenFlag/EndFlag 异步读取顺序问题。

### 5.2 Combine 已通过 Dispatch 元数据恢复实际长度

Dispatch 会把实际 token 的 inner/outer count 和 offset 写入 `epRecvCountsOut`。Combine 使用这些元数据：

- 在 server 内按 token 的实际 topK 分布进行聚合；
- 每生成一个有效 token 就写入到达标志；
- 以 token 到达标志或结束标志终止轮询；
- 最终按本 rank 的实际 `axisBS` 还原输出。

因此 Combine 不需要预先知道其他 rank 的本地 BS，只需要所有 rank 对元数据分区容量 `globalBS` 的理解一致。

### 5.3 修改前不能直接放开校验的原因

虽然通信协议已经能够处理不同实际长度，但修改前仍存在三处容量语义问题：

1. Tiling 明确拒绝 `localBs != globalBS / epWorldSize`，能力无法进入 Kernel。
2. Dispatch hierarchy 使用 `worldSize * localBs * localMoeExpertNum` 设置 `expandXOut` 的逻辑长度。
   动态 BS 下该值可能小于 Tiling 按统一容量计算的输出 `A`，导致 Kernel 的逻辑访问边界与实际输出容量不一致。
3. Combine hierarchy 直接把 EP 域的 `globalBS` 当作单 rank 最大 BS。动态 BS 下会造成 UB 分配、元数据搬运和
   循环上界语义错误；正确值应为 `globalBS / worldSize`。

所以只删除可变 BS 校验是不完整的，必须同时修正 Dispatch 输出边界和 Combine 的单 rank 容量计算。

## 6. 详细设计

### 6.1 Tiling 校验调整

公共 Window 校验中将原来的无条件限制：

```text
hierarchy && localBs != maxBs -> 失败
```

调整为：

```text
hierarchy && SoC != Ascend910_93 && localBs != maxBs -> 失败
hierarchy && maxBs > 256 -> 失败
```

这样只在 A3 开放 variable BS，A2 等其他产品仍保持原限制。同时继续复用已有 `globalBS` 校验：

- 非 0 `globalBS` 必须能被 `epWorldSize` 整除；
- `globalBS >= localBs * epWorldSize`，即本地实际 BS 不能超过统一容量；
- `globalBS > localBs * epWorldSize` 时不能传 `xActiveMaskOptional`。

单个算子的 Tiling 无法在不增加通信的前提下验证“所有 rank 的 `globalBS` 完全相同”，该条件由接口契约和
上层框架保证。

### 6.2 Dispatch Kernel 输出边界修正

Tiling 已根据统一容量计算输出上界 `A`，并写入
`moeDistributeDispatchV2Info.a`。Kernel 新增 `expandXRows_` 保存该字段，初始化 `expandXOutGMTensor_` 时使用：

```text
逻辑元素数 = A * H
```

替代原来的：

```text
逻辑元素数 = epWorldSize * localBs * localMoeExpertNum * H
```

这样逻辑边界与 Host 校验过的输出 Tensor shape 一致，不再依赖本 rank 的实际 BS 推导全局最坏接收量。
实际写出的 token 数仍由通信接收结果决定，不会因为容量变大而产生冗余 token 计算。

### 6.3 Combine Kernel 单 rank 容量修正

Combine 中 `maxLocalBs_` 用于：

- 分配 inner count/offset 的 UB 临时空间；
- 从 Dispatch 元数据区搬入本 rank 对应的有效容量；
- 控制 server 内聚合和结束标志轮询上界。

修改后计算为：

```text
maxBsPerRank = globalBS / worldSize
maxLocalBs = min(maxBsPerRank, 256)
```

这与 `globalBS` 的容量定义一致。对于原来所有 rank BS 相同且 `globalBS=0` 的调用，Tiling 转换后
`maxBsPerRank` 正好等于实际 BS；循环范围更准确，不增加通信或计算量。

### 6.4 元数据布局保持不变

Dispatch 和 Combine 之间的 hierarchy 元数据仍按 `globalBS` 划分容量：

```text
| expert recv count |
| inner count:  globalBS * serverNum |
| inner offset: globalBS * K * serverNum |
| outer count:  local actual BS |
| outer offset: local actual BS * K |
```

其中 inner 区需要覆盖来自整个 EP 域的容量，所以使用 `globalBS`；outer 区只描述本 rank 的输入 token，
所以使用本地 `axisBS`。本次没有改变字段顺序、数据类型和 Dispatch/Combine 配套关系。

### 6.5 Window 空间规划

hierarchy 的 HCCL Window 最坏空间继续按统一容量 `maxBs=globalBS/epWorldSize` 计算，而不是按当前 rank 的
实际 BS 计算：

```text
actualSize = moeExpertNum * maxBs
             * (H * maxOutputDtypeSize + tokenMetadataSize + flagAndScaleSize)
             + hierarchyFixedStateSpace
```

这样各 rank 即使实际 BS 不同，也会得到相同的 Window 布局和地址偏移。新增 `maxBs<=256` 校验避免容量超过
hierarchy Kernel 的固定数组和 UB 设计上限。

### 6.6 资料更新

README 和 ACLNN 接口资料统一明确：

- A3 hierarchy 支持不同 rank 的 BS；
- 动态 BS 时 `globalBS` 是“单 rank 配置容量 × EP rank 数”，不是实际 BS 求和；
- 所有 rank 必须配置相同且非 0 的 `globalBS`；
- 动态 BS 场景不能传 `xActiveMaskOptional`；
- hierarchy 仍不支持二维 mask、特殊专家和 `performanceInfoOptional`。

## 7. 文件级修改清单

| 文件 | 修改 | 原因 |
|---|---|---|
| `moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h` | 仅 A3 放开 `bs != maxBs`；新增 hierarchy `maxBs<=256` 校验 | 开放目标硬件能力，并保护 Kernel 容量上限 |
| `moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` | 将错误信息中的 maxBS 语义改为配置容量 | 避免用户误认为必须先通信得到实际最大 BS |
| `moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_layered.h` | 使用 Tiling 的 `A` 初始化 `expandXOut` 逻辑长度 | 修复本地实际 BS 无法代表全局最大接收量的问题 |
| `moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp` | 同步修正 `globalBS` 提示语义 | 保证 Dispatch/Combine 参数解释一致 |
| `moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_layered.h` | `maxLocalBs_` 改为 `globalBS/worldSize` 并限制为 256 | 修正 EP 总容量与单 rank 容量混用 |
| 两个算子的 arch22 Tiling UT | 新增 A3 hierarchy `BS=8`、`maxBs=32`、`epWorldSize=16` 成功用例 | 覆盖 `localBs != maxBs` 的 Host 放行路径 |
| 两个算子的 README | 更新 dynamic BS、`globalBS` 和限制说明 | 更新算子资料 |
| DispatchV4/CombineV4 ACLNN 文档 | 更新 A3 hierarchy 约束和参数说明 | 更新外部 API 资料 |

Combine README 中同时修正了非 hierarchy 场景 `expertScalesOptional` 的既有资料描述；该项不改变本需求的
Kernel 行为。

## 8. 兼容性与性能分析

### 8.1 向后兼容

- 所有 rank BS 相同且 `globalBS=0`：Tiling 仍转换为 `BS * epWorldSize`，接口行为不变。
- 所有 rank BS 相同且显式配置 `globalBS=BS*epWorldSize`：行为不变。
- A2 hierarchy：仍要求 `localBs == maxBs`，不扩大支持范围。
- 非 hierarchy 路径：TilingKey 和 Kernel 选择不变。
- Dispatch/Combine 元数据格式、TilingKey、输入输出 dtype 和确定性语义不变。

### 8.2 性能

动态 BS 场景消除了上层框架为 BS 对齐增加的 DP 间通信和 padding 管理。Kernel 只发送本 rank 的实际 token，
统一容量仅用于地址规划和边界校验，不引入按 `maxBs` 补齐的 token 通信或专家计算。

原等 BS 场景没有新增通信步骤。Combine 的循环上界从误用 EP 总容量修正为单 rank 容量，理论上不会造成
性能退化；在小 BS、大 EP 场景还会减少无效元数据扫描。

## 9. 风险与防护

| 风险 | 后果 | 防护措施 |
|---|---|---|
| 不同 rank 配置不同 `globalBS` | 元数据分区和 Window 地址不一致，可能产生错误或等待超时 | 接口资料要求所有 rank 配置一致；上层使用统一模型配置下发 |
| 本地 BS 超过配置容量 | 输出或 Window 容量不足 | Tiling 校验 `globalBS >= localBs * epWorldSize` |
| `maxBs>256` | 超出 hierarchy 固定数组/UB 上限 | Tiling 显式拒绝 |
| 动态 BS 同时传 mask | 实际长度与 mask 长度形成两套语义 | 复用已有冲突校验，动态 BS 时禁止 mask |
| 测试误传 `performanceInfoOptional` | hierarchy Tiling 失败 | 测试配置关闭 `need_performance_info`，而非只关闭框架计时开关 |
| 结束标志先于最后 token 被错误判断 | 少收最后一个 token | A3 接收侧分三次同步读取 next TokenFlag、EndFlag 和 current TokenFlag；本次不改该协议 |

## 10. 验证与验收方案

### 10.1 已增加的 Host Tiling 覆盖

- Dispatch：A3 hierarchy，`localBs=8`、`maxBs=32`、`epWorldSize=16`，期望 Tiling 成功；
- Combine：使用与 Dispatch 一致的容量和元数据 shape，期望 Tiling 成功。

### 10.2 建议的功能验收矩阵

在跨超节点 A3 环境、`commAlg="hierarchy"` 下至少覆盖：

| 类别 | 用例 |
|---|---|
| 基本动态 BS | 16 个 rank 使用不同 BS，如 1、2、4、8、16、32 的混合分布 |
| 边界 | `maxBs=1`、`maxBs=256`，实际 BS 分别取最小值和容量值 |
| 等 BS 回归 | 所有 rank BS 相同，分别使用 `globalBS=0` 和显式容量 |
| EP 泛化 | `epWorldSize=16/32/64/128/256` |
| 专家泛化 | `moeExpertNum` 覆盖最小值、常用值和 512 |
| 数据类型 | FP16、BF16；通信量化模式按原 hierarchy 支持范围覆盖 |
| 路由分布 | 均匀、单专家热点、单 server 热点、跨 server 极不均衡 |
| 负向参数 | `maxBs=257`、`globalBS` 不整除 EP、本地 BS 超容量、动态 BS 传 mask |
| 不支持项 | hierarchy 传 performanceInfo、二维 mask、特殊专家时继续校验失败 |

每个功能用例需要将 Dispatch 输出送入专家计算和 Combine，并与 padding 后等 BS 的基准实现比较最终输出、
`expertTokenNums` 和有效 token 数。精度标准沿用原 hierarchy 场景。

### 10.3 性能验收

- 等 BS：与修改前 hierarchy 版本比较 Dispatch/Combine 单算子耗时，不得劣化；
- 动态 BS：比较“上层 DP 通信 + padding + 原算子”与“直接传实际 BS + 新算子”的端到端耗时；
- 重点覆盖多机、BS 差异较大和路由极不均衡场景。

## 11. 结论

hierarchy 的 A3 通信状态机原本已经能够通过 TokenFlag/EndFlag 识别每个 server 的实际 token 数，动态 BS 的
主要阻塞点是 Host 校验和容量语义，而不是通信协议本身。本方案以统一 `globalBS` 容量保证跨 rank 内存布局
一致，以本地实际 BS 驱动计算和通信，并修正 Dispatch 输出边界及 Combine 单 rank 容量计算，从而在不增加
前置 rank 间通信、不改变原协议和元数据格式的前提下支持不同 rank 输入不同 BS。
