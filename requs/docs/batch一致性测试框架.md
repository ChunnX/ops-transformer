# 大纲

![](images/batch%E4%B8%80%E8%87%B4%E6%80%A7%E6%A1%86%E6%9E%B6%E5%A4%A7%E7%BA%B2.png)

```txt
# batch一致性测试框架开发大纲

## 1. 图例说明 (Legend)
- 🔴 **实现速度快，优先实现** (粉色背景)
- ⬜ **现有功能** (白色背景)
- 🔵 **实现速度慢，后期实现** (蓝色背景)

## 2. 核心流程 (Main Flow)
- **S2_B分析模块** 🔵
  - *描述：保障至少有token能被切到多核*
  - ⬇️
- **给定的参数** ⬜
  - ⬇️
- **当前生成golden的代码** ⬜
  - ⬇️
- **黑盒场景覆盖模块** 🔴
  - ⬇️
- **metadata算子调用** ⬜
  - ⬇️
- **自动场景覆盖模块** 🔵
  - ⬇️
- **白盒覆盖情况自动分析** 🔵
  - ⬇️
- **一致性对比模块** 🔴
  - ⬇️
- **精度对比** ⬜

## 3. 并行/辅助逻辑 (Parallel Logic)
- **两次执行间random方式** 🔴
  - *说明：此步骤与“黑盒场景覆盖模块”并行，并汇入“一致性对比模块”*

```

# 支持的黑盒场景

## batch间一致性

| 黑盒表现                    | 测试模式 | 测试方法                                                |
| --------------------------- | -------- | ------------------------------------------------------- |
| 调换batch顺序，对应结果一致 | 模式1    | 标记batch，打乱顺序，保证Q和KV对应，对比各batch结果md5  |
| batch拆分                   | 模式2    | 标记batch，拆分batch，保证Q和KV对应，对比各batch结果md5 |

### 模式1代码

new_order random

```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

# 按 [2, 0, 3, 1] 的顺序重排
new_order = [2, 0, 3, 1]
t_reordered = t[new_order]

print(t_reordered)
# tensor([[ 9, 10, 11, 12],
#         [ 1,  2,  3,  4],
#         [13, 14, 15, 16],
#         [ 5,  6,  7,  8]])

```

### 模式2代码

split_size_or_sections random

```python
t = torch.randn(10, 4)

# 拆分成 3 份，大小分别为 [2, 3, 5]
splits = torch.split(t, split_size_or_sections=[2, 3, 5], dim=0)
t1, t2, t3 = splits[0], splits[1], splits[2]

print(t1.shape)  # torch.Size([2, 4])
print(t2.shape)  # torch.Size([3, 4])
print(t3.shape)  # torch.Size([5, 4])

```

## batch内一致性

| 黑盒表现                     | 测试模式 | 测试方法                                                     |
| ---------------------------- | -------- | ------------------------------------------------------------ |
| 一个batch拆分成多个batch计算 | 模式3    | 方法1：拆分到多个batch间计算，每个batch跟原batch的shape一致，KV保持一致，标记token，对比各token结果；方法2：打乱同shape不同batch，KV保持一致，token在batch中的相对位置不变，标记token |

## 相同token，不同batch shape一致性

| 黑盒表现               | 测试模式 | 测试方法                                                     |
| ---------------------- | -------- | ------------------------------------------------------------ |
| token，不同batch shape | 模式4    | 在现有batch第一个token前面增加或者删减最前面的token，对比增减前后的token对应的输出值 |

复用模式2代码

# 支持的白盒场景

## level-1

-   s2_b / 32 == 128
-   s2_b / 32 < 128
-   s2_b / 32 > 128， s2_b % 32 == 0
-   s2_b / 32 > 128， s2_b % 32 != 0
    计算方式：在区间内random出s2_b，根据cmp_ratio和maskmode倒推出ORI和CMP

## level-2

1.  所有token都在单核上规约（每个都不分到多个核）
2.  有token多核规约（1. 根据S2_b选取符合条件的token，S2_b至少有两个规约块 2. 随机挑选其中给定多个，可以指定跟当前token分到同一个核上的token数目区间 3. 安排到不同核 4. 通过调整数量可以实现单核规约和多核规约混合）
3.  有核有多token规约数据（1. 根据S2_b挑选符合条件的token，S2_b至少有三个规约块 2. 随机挑选其中给定多个“token对”，可以指定token对之间相隔的token数 3. 安排它们及它们之间的token分到一个核上做FA）

## 核的视角分析规约情况

-   某核上有一个token的单核规约数据（1覆盖）
-   某核上有一个token的跨核规约数据（2覆盖，指定同核token数为0）
-   某核上有两个token的跨核规约数据（3覆盖，指定token对之间相隔0个token）
-   某核上有token单核规约数据+1个跨核规约数据（2覆盖，指定同核token数大于0）
-   某核上有token单核规约数据+2个跨核规约数据（3覆盖，指定token对之间相隔大于0个token）

# 自动化手段

## S2生成模块

控制S2_b至少能被切分到2个核

## 固定基准

采用level-2的1模式，保留各token的md5，作为一个固定基准

## 不固定基准

采用黑盒场景的手段，执行两次，相互比较

## 白盒情况自动分析

通过解析metadata结果，计算level-2覆盖率

## 自动场景覆盖

使用修改metadata的形式，在给定区间内随机出S2_b，实现给定规约情况

# 覆盖保障方式

1.  黑盒场景脚本自动实现，配置模式即可覆盖

2.  白盒场景kernel侧代码只能通过修改metadata结果实现，但是可以统计白盒覆盖情况

3.  metadata覆盖情况只能通过人工用例实现，但是可以统计白盒覆盖情况

    实现顺序：

不固定基准+手动分析用例 => 代码实现 => 白盒情况自动分析模块实现 => 自动场景覆盖用来泛化测试

# 2026-07-19 实现校准

本文件正文是早期开发草案，保留用于追溯。当前发生冲突时，以 `SMLA batch一致性.md` 为
第一需求源，以两算子 README/golden/源码和 PR8737 为接口与算法证据；最终实现与状态见
`../requs.md` 和 `../SMLA_batch一致性测试开发计划.md`。

## 黑盒模式校准

1. 模式 1：同一组 batch 重排，所有 batch-aligned 输入同步重排；
2. 模式 2：pytest 参数层构造独立子调用，规范示例是 `B=8 -> B=3+B=5`，正式验收使用显式
   groups；这不是算子内部按 seed 分核；
3. 模式 3：对一个 batch 做完整、不重叠 token partition，例如 `16K -> 4K+12K`；
4. 模式 4：按主需求比较 `300+100` 与 `300+200` 的共同前 300 token。

模式 3/4 对 RightDownCausal/Band 不能只改变 Q。当前实现分别使用：

```text
Mode 3 [start,end): new_ori_S2 = old_ori_S2 - old_Q_length + end
Mode 4:             new_ori_S2 = old_ori_S2 + new_Q_length - old_Q_length
```

随后 actual-input 语义 oracle 校验 Q、可见 ori/cmp KV、mask window、sparse indices、top-k
和 sinks；不能证明等价的 case 返回 `INVALID_TRANSFORM`。

## 原 pytest 流程校准

当前在两个算子各自的 `tests/pytest_v2` 中保留：输入/CPU golden -> PT save/load -> 参数变换
-> metadata -> 主算子 -> CPU 精度比较，同时增加 baseline/derived MD5、metadata 分析和 level
恢复。命名 `--modes`/`--profile` 是正式接口；`-M 1,2` 仅为迁移兼容。

## 白盒自动分析校准

“白盒情况自动分析”已经实现，不修改 metadata 伪造覆盖，而是解析主算子同一次调用实际消费
的 1024-int32 metadata：

- FA `36x9`、FD `72x8` 和 reserved tail；
- `bn2 -> (batch,n2)` 与 query-token/S2-block 展开；
- 同时报告需求 `align_up(ceil(actual_S2/32),128)` 与 PR8737 head 的整数 floor 公式；结构
  展开暂按 PR head，公式冲突不静默隐藏；
- split-G 重复 FA range 去重；
- FA/FD workspace 关联、G 行覆盖和结构错误；
- Level-1 四类、Level-2 token/core 视角与 missing coverage。

自动生成目标 shape、STC 001-027 manifest 和真实 level-3 coverage matrix 仍属于下一阶段。
本机 runtime 当前在第一条 NPU 操作处拒绝 level 3，所以 level-0 metadata 只能做接口/布局
smoke，不能作为 FD、固定调度或主算子 batch consistency 的通过证据。

# 2026-07-20 可离线推进项与完成状态

NPU level3环境等待不再阻塞以下工作。详细代码和命令见`../level3两段式完整调用示例.md`，冲突
见`../SMLA_MQSMLA需求冲突与对齐清单.md`。

| 原大纲模块 | 无NPU能否做 | 当前状态 | 证据/剩余项 |
| --- | --- | --- | --- |
| 给定参数/Excel输入 | 能 | 已完成小shape双算子表 | `../cases/SMLA_MQSMLA_batch一致性离线预检.xlsx`；原STC仍需owner修订 |
| 当前golden | 能 | 两算子真实运行通过 | 复用原golden，不另造简化数学模型 |
| PT save/load | 能 | baseline + 6个derived round-trip通过 | `consistency_prepare`，digest before/after一致 |
| 黑盒场景覆盖 | 能 | 模式1-4均已实现 | BSND/TND/PA变换、TokenOrigin、actual-input oracle |
| 两次执行间random | 能 | seed可复现；默认不随机模式2 | 仅显式`--randomize`启用fuzz |
| 一致性对比 | 大部分能 | origin映射/MD5/CPU精度代码与单测完成 | 真NPU bitwise结果仍待level3环境 |
| metadata输入摘要 | 能 | 已完成 | PT来源、shape/dtype/device/byte/digest/sample/小tensor完整值 |
| metadata算子调用 | 不能完整验收 | 两算子executor已接通 | runtime拒绝level3，状态为`UNSUPPORTED` |
| 白盒自动分析 | 算法能，真实覆盖不能 | parser/analyzer/公式审计完成 | 真实FA/FD coverage matrix待NPU metadata |
| S2生成/Level-1 manifest | 能 | 分类器完成，STC manifest未完成 | floor/ceil公式和STC错误需owner裁决 |
| 自动场景覆盖 | 部分能 | 暂不修改metadata伪造覆盖 | 后续应生成合法输入，再观察真实metadata |
| graph/ACLNN | 部分能 | 原graph保留，一致性executor未完成 | 等eager level3先闭环，再做STC 021/023 |

## 离线链的验收边界

```text
Excel
  -> 原input normalization
  -> 原CPU golden
  -> baseline PT save/reload
  -> modes 1-4 transform + semantic oracle
  -> 从baseline cpu_output按TokenOrigin物化比较token预期
  -> derived PT save/reload
  -> schema + full-byte digest + JSON input report
```

模式4额外token没有`TokenOrigin`，所以离线PT中会明确记录`unmapped_tokens`；这些位置不进入
batch一致性比较，零模板也不宣称是完整CPU golden。若未来需要验证额外token，必须给原golden
新增“对已有tensor重新计算”的正式接口。

## 不再采用的早期策略

1. 不把metadata-only临时probe放在正式`batch/`；该文件已删除。
2. 不通过手工改metadata伪造白盒PASS；analyzer只读主算子实际消费的metadata。
3. 不用level0普通精度PASS替代level3一致性PASS。
4. 不再扩展`-M 1,2`数字含义；正式配置只用命名mode/profile。
