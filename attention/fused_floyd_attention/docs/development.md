# FusedFloydAttention 开发文档索引

## A3→A5 迁移文档

以下文档记录了将 FusedFloydAttention 算子从 Atlas A3（DAV_2201）迁移到 Atlas A5（DAV_3510）的完整过程。

### 文档说明

| 文档 | 内容 |
|---|---|
| [fused_floyd_attention_a3_to_a5_migration.md](fused_floyd_attention_a3_to_a5_migration.md) | **完整迁移分析**。包含架构映射（A3→A5 的 NpuArch/SocVersion/编译宏对应关系）、`__DAV_C310_CUBE__` 宏存在性验证、DAV_3510 架构差异（Buffer 容量/核数/4:2 稀疏/数据通路/同步机制/数据格式）、本算子与架构差异的关联点分析（tiling 自适应/Buffer 硬编码/GM workspace/TPosition/同步机制/SSBUF 宏）、修改清单（P0 阻塞/P0 验证/P1 优化/P2 深度优化/测试） |
| [fused_floyd_attention_a3_to_a5_todo.md](fused_floyd_attention_a3_to_a5_todo.md) | **待修改点清单**。精简版，按 P0 阻塞（不改无法运行）/ 测试 / 已就绪 / 架构速查 四段组织，每条含文件:行号和具体改法 |
| [fused_floyd_attention_a3_to_a5_issues.md](fused_floyd_attention_a3_to_a5_issues.md) | **遗留问题**。记录 `DropOutBitModeInit` 在 A5 缺失的根因（A5 Regbase 架构不提供该函数）、影响评估（本算子 hasDrop=false，无实际影响）、处理方式（架构宏条件编译）、遗留风险（若未来支持 dropout 需改用 Regbase 实现） |
| [fused_floyd_attention_a3_to_a5_status.md](fused_floyd_attention_a3_to_a5_status.md) | **当前代码状态总览**。含编译/安装/tiling/kernel/端到端各阶段状态表、与初始代码的 7 处差异（每处含文件:行号/改法/原因）、tiling 测试结果（A3/A5 UT 全部 PASSED）、A3 vs A5 tiling 对比结论、待解决问题（kernel 执行 hang） |
| [fused_floyd_attention_a3_a5_tiling_diff.md](fused_floyd_attention_a3_a5_tiling_diff.md) | **A3 vs A5 tiling 对比**。用相同 shape 在 A3/A5 平台参数下执行 tiling，对比结果：仅 5 处差异（workspace 大小 + 4 个 Bmm 的 shareL0CSize），全部源于 L0C 扩容（128KB→256KB），tiling 逻辑正确 |

### Tiling 数据

| 文件 | 内容 |
|---|---|
| [fused_floyd_attention_a5_tiling_dump.txt](fused_floyd_attention_a5_tiling_dump.txt) | A5 平台 tiling 原始数据（28 cube / 56 vector, UB=248KB, L0C=256KB），shape: B=1,H=1,N=16,M=128,K=128,D=32，232 个 int64_t 字段 |
| [fused_floyd_attention_a3_tiling_dump.txt](fused_floyd_attention_a3_tiling_dump.txt) | A3 平台 tiling 原始数据（24 cube / 48 vector, UB=192KB, L0C=128KB），相同 shape，用于对比 |

## 迁移要点速览

- **架构映射**：A3=DAV_2201/`__DAV_C220_CUBE__`，A5=DAV_3510/`__DAV_C310_CUBE__`
- **主要修改**：算子注册 ascend950、Cube 宏分支扩展、DropOutBitModeInit 架构条件编译、文档更新
- **tiling 验证**：A3/A5 结果几乎一致，差异仅来自 L0C 扩容
- **待解决**：kernel 在 A5 仿真时 hang，需进一步排查同步机制/数据通路适配

## 目前所作的尝试

1. 以过编译为目标，进行简单适配
结果：过编译后kernel侧运行卡死，对比tiling数据处L0C大小和workspace大小外均相同。
结论：tiling侧无需太多适配，代码迁移工作量主要再kernel侧。
隐患：芯片核数增加若原始tilingdata中数组长度小于核数可能会出问题（待确认），其他内存大小A3->A5均增大，应该不会有问题。

2. 保留当前Process模板，仍使用高阶MATMUL API
结果：matmul的tiling计算有问题，无法正常运行。查阅AscendC开源文档没有支持A5的说明，在ops仓中目前只知道mc2在A5上使用matmul高阶API。对matmul高阶api的使用和A5的支持度不是很了解。
结论：需要深入了解matmul高阶API，一方面如果选择使用高阶API需要明确A5的支持度以及和A3的差别，另一方面如果需要改基础API需要明确高阶API中执行了哪些操作，以复刻算子原计算逻辑。

3. 保留当前Process模板，使用agent适配高阶api相关操作
结果：精度异常，目前定位到是tiling的适配没有做全，agent仍在开发，但不抱太大希望。
结论：agent自主适配试错成本相对较高，另外不能使用高阶api个人感觉不如按照基础api适配。

总结：迁移工作的困难主要是kernel侧高阶api的适配，特别是matmul高阶api（含有同步相关封装）；另外计算ATK测试工具的使用和ATK中调试方法差异也造成了较大阻碍；高阶API在A5上不能直接使用让迁移工作中的重构工作量大大增加。

## 测试用例

与A3测试进行对其，精度测试用例存放在 fork_cann/ops-transformer/attention/fused_floyd_attention/tests/ST 当中，共5000条，使用agent筛选出47条作为典型用例，能实现基本的黑盒用例覆盖，白盒用例覆盖度有待考量。

## 进度安排

1. ATK工具调试方法封装（1.5人天）：ATK工具中使用printf和dumptensor，并封装成skill以供后续agent调试参考，不然每次调试不规范又开始摸索很浪费时间。
2. matmul高阶API适配（3人天）：理解matmul高阶API在A3中的行为，基于mc2的使用示例明确A5可行操作，使用agent对高阶api进行适配，另外得先明确高阶api的tiling运行报错的原因。
3. 基础API适配（5-9人天）：将高阶API主指令映射到基础API，将高阶API封装的同步关系单独配置，根据主指令映射关系配置API参数。目前感觉比较稳定的方式是分段验证，用单基本块C1V1C2V2分步进行，按每个步骤2天来算，根据V侧高阶API能否直接使用需要4-8天。
4. STC用例调试（2+人天）：用agent对STC中不通过的用例进行调试，必要情况转回黄区用model根据日志调试。
总计：11.5-15.5人天；后续算子有此次适配经验，合理总结skill可缩短3.5-6人天。