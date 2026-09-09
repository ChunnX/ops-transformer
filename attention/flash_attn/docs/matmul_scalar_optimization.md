# FlashAttn Matmul / L0 buffer 下沉分析

基线：upstream/master `d5c0617b6`；分支：`perf/flash-attn-local-matmul`。验证日期：2026-09-09。

## 本次改动

ND/DN cube block 使用算子内 `op_kernel/arch35/flash_attn_matmul.h`。
common 头文件保持原样，其他 attention 算子的调用不受此次修改影响。

| 修改 | 意义与开销来源 |
| --- | --- |
| BufferManager + Policy + Buffer 改为固定地址和槽位编号 | 热循环不再选择和复制带 LocalTensor、size、event、mutex、cross-core ID 的通用对象；地址由 `slot * 32768` 得到，mutex 由固定基址加 slot 得到。ND/DN block 只保存 `l0aIdx_`、`l0bIdx_` 两个整数，matmul 通过引用手动更新索引；不再保留 L0Buffers 类。 |
| A/B 转置方向改为模板参数 | ND BMM1、DN BMM1 都是 false/true；ND BMM2 是 false/false；DN BMM2 是 true/false。使 LoadData 参数和 L1 切片偏移的专用路径在编译期确定。 |
| MMParam 从 11 个字段缩减至 6 个 | 移除未使用的 cmatrixInitVal、固定为 true 的 isOutKFisrt、固定为 0 的 unitFlag，以及两个转置字段；切 K 时仍只在首块清零累加器。 |
| 直接调用 Mutex / LoadData / Mmad | 同步顺序在本算子内可直接审查，避免多层策略模板；保持原同步关系，不额外合并或删除依赖。 |
| 删除 MatmulBase 并裁剪调用点分支 | DN 小 D 的 BMM1 直接 Full，DN BMM2 直接 Full；ND 小 D 仅 S2=256 的 N 尾块可能切 N，ND BMM2 只保留可能的切 K。见下文边界证明。 |
| 去掉 namespace 导入 | ND/DN 不使用 `using namespace FlashAttnMatmul`；所有相关类型、函数与常量显式限定命名空间。 |
| 删除 BufferManager 初始化和 ResetIdCounter | 去掉固定地址、固定 ID 场景的多余初始化；这是每个 kernel 的开销，通常没有热循环优化重要。 |
| 去掉 MX、量化、多架构及未用接口 | 缩小维护范围。原来的不可达模板分支往往已经被编译器消除，不能把删除源码行数直接算作 scalar 收益。 |

## 必须保留的语义

- 双 buffer 的首次使用是 pong，之后 ping/pong 交替，轮转状态跨 BMM1/BMM2 和 task 保留。
- ND `s2BaseSize == 256 && dBaseSize > 128` 的 L0B 保持单 buffer，始终地址 0、mutex 8。
- L0A 使用 mutex 6/7，L0B 使用 8/9；槽位地址 0/32768，各 32 KiB。
- Full 路径保留两组搬运到计算的同步；切 K/N 路径保留通过 L0B mutex 覆盖两次 MTE1 搬运的原顺序。
- `realM`、M=1 时按 16 执行的处理、计算 K/N 与搬运 loadK/loadN 的区分均保留。
  例如 D=72：Mmad 使用 K=72，搬运按 80 寻址，不能把两者混成同一个尺寸。
- 不改变 L1/UB/L0C 布局、跨核 flag 编号、Fixpipe、softmax 或 mask 逻辑。

## 编译产物结果

CANN 9.1.0，`Ascend950PR_9599`，相同 `high_performance,optional`、auto-sync off 编译选项。
分别编译 upstream 源码和修改后的源码；FP16/BF16 均覆盖 BSND、连续 KV、无 mask 的
ND config 0..7 与 DN config 0/2/6，共 22 个 dtype/tiling-key 组合，包含 AIC/AIV 两侧。

仅删除并原样展开 MatmulBase 时，22 个组合的 AIC `.text` 与第一版下沉实现逐字节一致。
说明编译器原本已内联，单纯展开没有额外指令收益。进一步按调用点裁剪不可达分支后，结果如下。

下表为最终版 FP16 单个 AIC ELF 的 `.text` 字节数（`readelf -SW`），不是整个文件大小：

| 模板 | config | tiling key | 基线字节 | 最终字节 | 减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ND | 0 | 0 | 13096 | 10824 | 17.35% |
| DN | 0 | 131072 | 13200 | 10768 | 18.42% |
| ND | 1 | 33554432 | 14368 | 11832 | 17.65% |
| ND | 2 | 67108864 | 13192 | 10792 | 18.19% |
| DN | 2 | 67239936 | 13280 | 10776 | 18.86% |
| ND | 3 | 100663296 | 14480 | 11824 | 18.34% |
| ND | 4 | 134217728 | 12032 | 10976 | 8.78% |
| ND | 5 | 167772160 | 11400 | 10704 | 6.11% |
| ND | 6 | 201326592 | 11784 | 10944 | 7.13% |
| DN | 6 | 201457664 | 11936 | 11016 | 7.71% |
| ND | 7 | 234881024 | 12496 | 11344 | 9.22% |

静态代码体积减少 6.11%～18.86%，不能直接当作动态 scalar 指令数降幅。
本次使用硬件 PMU 的 scalar 时间衡量运行时收益，没有统计动态 scalar 指令条数。

## 手动维护 L0 索引

随后删除了 `L0Buffers`，ND/DN block 直接保存两个 `uint32_t` 成员 `l0aIdx_`、`l0bIdx_`。
Full 每次翻转一次 A/B，切 K 每轮翻转 A/B，切 N 仅在进入时翻转 A、每轮翻转 B。
索引通过引用跨 BMM/task 保留；双 buffer 首次使用 pong，单 L0B 不翻转且寻址显式使用常量槽位 0。
地址、mutex ID 和 LocalTensor 都在 matmul 内直接构造，不再通过 buffer 对象调用。

本次重构重新编译了 FP16/BF16 的 22 个组合，并逐字节对比 44 个 AIC/AIV ELF 的 `.text`：
**44/44 与删除类之前完全一致**。因此这一步主要精简源码结构，没有额外的机器指令收益；上述性能数据对应的代码未改变。

## 调用点分支裁剪的依据

- `UpdateTilingKeyConfig()` 将实际 D=64 映射至 dBaseSize=64，D=72/128 映射至 128，D=192/256 映射至 192/256。
  因此 BMM1 的 `singleK=actual D <= dBaseSize`；D=72 的 `loadK=80` 仍独立保留。
- `CalcParams()` 中 `actMSize` 初始为 mBaseSize，尾块只会减小。DN 支持的 config=0/2/6 都是 M=128、S2=128；
  BMM1 的 N=actMSize<=128，小 D 路径不可能需要切 K/N；大 D 保留切 K。
  DN 类增加 `static_assert(mBaseSize <= 128 && s2BaseSize == 128)`，未来添加不同块长时必须重新评审。
- ND 的 BMM1 N=actSingleLoopS2Size<=s2BaseSize。小 D、S2=128 时全载；S2=256 时按当前 N>128 决定切 N。
- BMM2 的外层已经将 N 按 128 切分。DV<=128 时实际 N<=DV<=dVBaseSize，因此无需内层切 N。
  DN 的 K<=128 直接全载；ND 仅在原先需要切 K 的 S2=256、DV<=128 分支保留 `singleK > 128` 判断。

## 硬件 scalar 与总耗时

设备 Ascend950PR，CANN 9.1.0。FP16/BF16 × 5 组 D/DV × 4 组 (S1,S2,G)，共 40 例：
D/DV=(64,64)、(72,72)、(128,128)、(192,128)、(256,256)；
(S1,S2,G)=(1,257,3)、(17,129,3)、(128,513,8)、(2048,4096,8)。B=N2=1，BSND、连续 KV、无 mask。
所有版本共用 host/metadata、输入 seed、dtype、shape 和编译选项，只更换 kernel 包。

### PMU

`msprof --aic-metrics=PipeUtilization`；按 base→final→final→base 顺序完成两轮对照。
每 case 连续调用 30 次，剔除前 10 次，取后 20 次的中位数，再对两轮中位数取均值。
严格筛选 `OP Type=FlashAttn`，metadata 与随机数生成不计入。指标读取 `aic_scalar_time(us)`、
`aiv_scalar_time(us)`、`Task Duration(us)`；这些是 msprof 汇总的核/流水线指标，不把 scalar_ratio 的变化当作时间降幅。

| 范围 | AIC scalar 时间变化中位数 | AIC scalar 时间变化范围 | PMU kernel 耗时变化中位数 |
| --- | ---: | ---: | ---: |
| 全部 40 例 | -6.28% | -17.51%～+2.12% | -0.40% |
| 30 个小/中用例 | -4.65% | -9.61%～+2.12% | -2.43% |
| 10 个长矩阵用例 | -13.89% | -17.51%～-10.85% | +0.24% |

若把这 40 个 case 各执行一次，AIC scalar 时间之和减少 12.10%；这个数受长 case 权重影响，
不代表线上业务的总体收益。相应 PMU kernel 时间之和反而增加 0.17%，说明 scalar 收益没有同比转化为总耗时收益。

### 不采集 PMU 的 Graph replay 复核

为减少 Python 发射间隔影响，捕获 64 次 FlashAttn 的 NPU Graph，预热 replay 3 次；
每 case 计时 9 次 replay，NPU Event 时间除以 64 后取中位数。每个版本独立运行 3 轮，
最终取三轮中位数的中位数。这里是热数据 Graph replay 的每次算子摊销时间，包含摊销后的 Graph/Event 开销，
不宣称等价于所有 eager/冷数据场景。最后一轮基线与最终版各 40 例均检查了 Graph 输出/LSE 与自身 eager 结果完全相同。

最终 40 例总耗时变化中位数 **-0.18%**，范围 **-8.28%～+4.24%**；
长矩阵中位数 **-0.05%**，范围 **-8.28%～+0.65%**。整体接近持平，没有普遍加速。
全部逐例数据、三轮 run 的范围见 [matmul_profile_results.csv](matmul_profile_results.csv)。

### 是否存在劣化

存在需要保留在性能验收集中的小 shape 回退，不能给出“全场景无劣化”的结论：

| D/DV；S1/S2/G | dtype | 基线 Graph μs | 最终 Graph μs | 变化 |
| --- | --- | ---: | ---: | ---: |
| 128/128；1/257/3 | FP16 | 4.6001 | 4.7952 | +4.24% |
| 128/128；1/257/3 | BF16 | 4.6061 | 4.7630 | +3.40% |

FP16 三轮基线范围 4.5133～4.7299 μs，最终范围 4.7081～4.8370 μs；
BF16 基线范围 4.5501～4.6770 μs，最终范围 4.7607～4.7919 μs。
这些结果提示约 0.16～0.20 μs 的小用例回退；FP16 有测量范围重叠，不能将单轮百分比当作精确常数。
另外 FP16 D=64、S1=17 用例中位数 +3.45%，但两版本运行范围明显重叠，波动较大。

PMU 中部分 +4%～+5% 的中等 shape 在 Graph 复核中没有重现：例如 FP16 D=192、S1=128 在 Graph 中约 -0.60%，
BF16 D=64、S1=128 也未保持同方向。不能将每一个 PMU 短任务差异直接归因为代码回退。

对 D=128、S1=1、S2=257、G=3 又做了一轮单独隔离运行（同样 9 次 replay 的中位数）：

| dtype | upstream μs | 已下沉、未裁剪分支 μs | 最终版 μs |
| --- | ---: | ---: | ---: |
| FP16 | 4.8399 | 4.8649 | 4.7104 |
| BF16 | 4.7177 | 4.8475 | 4.7884 |

隔离运行中 FP16 回退没有重现，BF16 最终版约 +1.50%；最终版都没有慢于未裁剪分支的中间版。
这是一轮隔离检查，不能据此声称稳定加速，也不足以把完整矩阵中的 +3%～+4% 直接归因于分支裁剪。
更准确的结论是：**存在小用例回退观测，且对测量上下文敏感；尚不能保证全场景无劣化，也未确认寄存器/调度层面的根因。**

原因分析分为已有证据和待验证推测：

- **有证据：scalar 并非唯一瓶颈，流水线时间可重叠。** 例如 FP16 D=256、S1=2048，AIC scalar 从 98.58 降到 81.31 μs，
  但 MAC 时间约 189.97→191.10 μs，kernel 约 195.36→196.78 μs。减少 scalar 后，总耗时仍接近 cube 路径的时间。
  这是与 cube 限制一致的指标，不能把各流水线时间相加，也未据此证明每个核的关键路径。
- **有证据：小任务对固定开销和状态波动很敏感。** 同一基线独立 PMU 运行也出现约 -5.45%～+6.21% 的逐例变化。
  AIC 平均 scalar 减少也不能保证最慢 AIV、跨核到达时序、或最终完成时刻更早。
- **待验证：对象/分支精简改变了编译器的寄存器分配、指令调度或代码布局，可能让某些 Full 路径略慢。**
  当前没有指令级流水线/寄存器 spill 证据，不能断言具体根因。D=128、小 S1 的 PMU icache miss rate 约为 0，
  也不支持直接把该回退解释成 I-cache miss 增加。

复现脚本：`tests/pytests/profile_local_matmul.py`。分别选择两个算子包后运行：

```bash
msprof --output=/tmp/fa-profile --aic-metrics=PipeUtilization --task-time=on \
  --application="python attention/flash_attn/tests/pytests/profile_local_matmul.py --manifest /tmp/fa-manifest.json"
python attention/flash_attn/tests/pytests/profile_local_matmul.py --graph --manifest /tmp/fa-graph.json
```

## 真机回归

脚本：`tests/pytests/check_local_matmul.py`。选择各自的算子包后运行：

```bash
# upstream 包
python attention/flash_attn/tests/pytests/check_local_matmul.py --save /tmp/fa-baseline.pt
# 本分支包
python attention/flash_attn/tests/pytests/check_local_matmul.py --reference /tmp/fa-baseline.pt
```

算子 Python 扩展需支持 `head_dim_v`，metadata/host 与 kernel 协议需匹配。
本次用同一份 host/metadata 包，分别替换上述编译得到的基线与修改后 kernel，隔离本次 kernel 差异。
覆盖 FP16/BF16，D/DV=(64,64)、(72,72)、(128,128)、(192,128)、(256,256)，
(S1,S2,G)=(1,257,3)、(17,129,3)、(128,513,8)，共 30 个无 mask BSND 用例。
脚本检查输出和 LSE 有限、输出接近 FP32 SDPA，以及相对基线的输出/LSE 完全相同。
最终结果：30/30 用例通过，attention output 和 LSE 与 upstream 基线逐元素完全一致；
FP32 SDPA 输出检查和有限值检查均通过。

编译覆盖不代表每个 config 都在上述 shape 集中实际执行；本次未做全量 PA、TND、mask、sink 回归。长矩阵另有上述 Graph/eager 一致性检查。

## 下一步哪些修改值得做

1. **优先定位小 shape 回退**：将 D=128、S1=1、S2=257、G=3 加入性能验收，继续做局部消融与指令级分析。
   已测得 scalar 减少，但此场景总耗时未保证改善；不应以静态代码更小替代性能验收。
2. **扩大布局回归**：MatmulBase 已删除，边界可证明的分支已经裁剪；下一步覆盖 PA/TND/mask/sink，
   以及更多真实业务 shape，避免仅凭当前 40 例宣称全局收益。
3. **检查循环不变量是否已被提升**：LoadData 的固定字段、Mmad 的 M、L1 stride、尾块对齐值可能适合移出切 K/N 循环。
   先看生成代码和寄存器压力，避免增加长生命周期临时值导致 spill。
4. **有条件做满块特化/展开**：常见两轮 K/N 可考虑满块与尾块分离，但需对比代码体积和指令缓存代价；不宜全面展开。
5. **同步优化独立评审**：当前同步与基线等价。合并 mutex、跨核 flag 或缩减 buffer 数量会影响流水重叠和死锁条件，
   应作为另一项改动单独验证，不能混在纯 scalar 精简中直接删除。

优先级较低的是单纯搬文件、改名、手写常量模运算或移除空 inline 函数；编译器通常已处理这些开销。

## PR 前白盒与边界补充验证

运行 `python attention/flash_attn/tests/whitebox/local_matmul/run.py --mutations`：
直接编译实际 matmul 头文件，用 host recorder 记录指令参数，20,160 组契约检查通过。
覆盖 FP16/BF16、Full/K/N、ND/DN 使用的转置组合、M=1 与 16/32/64/128 边界、
S2=127/128/129/255/256、D=72 非对齐和 D=192 切 K 尾块。
检查 NZ 地址/stride、Mmad 初始化/累加、连续调用的 L0 slot 与 mutex 完整顺序。
5 类临时故障注入均被检出；测试不会修改生产头文件。

上板脚本新增 `--whitebox`：基线和最终版各运行 130 个确定性用例，包含多 batch、
多 KV head、连续多行和长 S2 任务。130/130 输出及 LSE 与 upstream **逐元素完全一致**，
有限值和 FP32 SDPA 输出检查均通过。复现方式见
[白盒说明](../tests/whitebox/local_matmul/README.md)。

Host recorder 不模拟硬件异步流水，也不替代 caller/host tiling 分支覆盖。
本轮真机范围仍为 BSND、连续 KV、无 mask；不宣称覆盖 PA/TND/mask/sink。
此前 22 个 dtype/tiling-key 编译组合和 44/44 `.text` 等价结果仍适用：本轮仅补充测试及文档，
没有进一步改变生产 kernel。
