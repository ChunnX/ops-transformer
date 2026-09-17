# 950 MegaMoe 批跑 MTE 故障维测

本补丁用于当前分支 `investigate/mega-moe-batch-symm-buffer`，未确认根因。
用户进一步确认从 004 开始批跑时 008 通过、009 失败，因此 009 具有独立复现路径。
首轮保持该失败序列和输入不变，不增加 launch blocking，不替换 SymmBuffer。

## 构建与启用

在有匹配 CANN/AscendC 工具链的机器上，用本分支构建：

```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe
```

本定位分支默认开启维测，不需要额外编译开关。
沿用现有算子包部署方式安装本次 `build_out` 下的 run 包，并使其在运行环境中生效。
必须更新 **Kernel 算子包**，仅更新 Torch wheel 不会替换这些 Kernel 改动。
构建脚本的自定义包安装步骤见 `docs/zh/install/compile.md`。
先单跑一次 009，确认输出出现 `[MM_DIAG_INIT_V1]`，再新建进程跑原失败批次。
没有该标记时，先检查包优先级、是否安装了本次构建产物及是否用 `ASCENDC_DUMP=0` 禁用了打印。

默认过滤：

- 仅 `maxOutputSize=2310` 的 MTE Kernel；前序 case 保持算法和参数不变，不输出这些维测日志。
- 普通日志仅物理 AIV `GetBlockIdx()==5`；同一条日志打印 `sub`，确认是否为 AIV1。
  设备故障日志的 core 编号未必使用同一命名方式，应以两边实际打印对应为准。
- Dispatch、Combine 各最多 64 条地址日志/核/launch；达到上限后打印提示。
- 非法索引、越界检查在所有核持续生效，不受普通日志核过滤和条数限制。
- 不新增同步，不覆盖输出，不重新分配通信区。printf 和检查会扰动时序，这轮不作性能测量。

调整过滤时，额外编译宏为 `MEGA_MOE_MTE_DIAG_MAX_RECV`（0 表示全部容量）、
`MEGA_MOE_MTE_DIAG_CORE`（-1 表示全部 AIV）、`MEGA_MOE_MTE_DIAG_RECORDS`。
默认值在 `op_kernel/arch35/common/mega_moe_mte_diagnostics.h`，可直接修改默认值再构建。
不建议首轮对 008 的 1000 次调用全部开打印，以免日志缓冲耗尽。
排查完恢复使用不含本维测补丁的正常算子包。

## 运行与获取信息

保持原有 Excel、起止行、批跑模式、循环次数等参数，只给原运行命令加 `-u` 和输出保存：

```bash
set -o pipefail
python3 -u /home/oiseau/codes/moe_test/mega_moe/batch_run_once/mega_moe_once.py \
  <原来的运行参数> 2>&1 | tee mega_moe_diag.log
```

以上 `<原来的运行参数>` 要替换为原命令参数，不是字面 shell 参数。
分别保存单跑009和从004开始批跑的完整日志。失败后用新进程复现；不要用同一故障进程继续得出独立结论。
脚本异常处理可能继续尝试后续 case，本次先收集009的第一处故障，不改其批跑控制逻辑。

可先提取下面的小日志回传，不需要手工整理 traceback：

```bash
rg -n 'MM_DIAG|invalid GM|cross-device|MTE|Case FAILED|Stack Trace|Reason:|最大CCL|cases=' \
  mega_moe_diag.log > mega_moe_diag_excerpt.log
```

若进程故障导致设备 printf 未完整导出，原始 plog 仍有价值。若环境使用常见的
`$HOME/ascend/log` 目录，可执行以下只读搜索；若配置了其他日志路径，替换为该路径：

```bash
rg -n -i 'mega_moe|mte|invalid.*address|cross.device|trap|task_id|stream_id|pc_start|pc_current' \
  "$HOME/ascend/log" > mega_moe_plog_excerpt.log
```

优先提供 `mega_moe_diag_excerpt.log`，保留完整 stdout/stderr 以便补查。
单跑日志中的 `MM_DIAG_INIT_V1` 同时验证维测 Kernel 确实加载，避免在旧二进制上继续排查。

## 打印位置与解释

| 标记/位置 | 内容 | 能回答的问题 |
| --- | --- | --- |
| `MM_DIAG_INIT_V1` / `LAYOUT` / `PEER` | 本次 shape、上限、prefetch、combine 模式，context/workspace 地址和 workspace 总大小，通信分区及 peer window 地址 | 单跑/批跑是否实际使用相同 tiling、地址表是否异常 |
| `count_read_enter`、`dispatch.count_read` | count 表的 GM 地址和读取字节数 | 是否在初次 count 搬入前后出错 |
| `MM_DIAG_EPOCH_WAIT` | 未匹配 epoch 的首次读值和期望值 | 是否在等前序通信到达；不在轮询中反复打印 |
| `MM_DIAG_COUNT` | 每个槽的原始编码、解码 count、epoch | 设备路由计数是否与 CPU 统计一致；slot 按 expert-major/source-rank 排列 |
| `dispatch.route_read` | route index 的地址、长度和逻辑分区边界 | compact route 下标读取是否超出槽/分区 |
| `dispatch.token_read` | 来源 peer、量化 token 记录基址、offset、单次 MTE2 字节数 | 跨卡 token 读取的地址链 |
| `dispatch.data_write/scale_write/meta_write` | 本段 workspace 写入起点和总长度 | 接收行号是否越过本次 maxOutputSize |
| `combine.meta_read/gmm2_read` | metadata 和 GMM2 结果的读取范围 | Combine 是否读取越界 workspace |
| `combine.remote_write` | 目标 peer、combine 区基址、offset、MTE3 字节数 | metadata 导出的跨卡写地址是否越界 |

`MM_DIAG_ADDR` 中地址为 `base + offset`；`limit` 是当前**逻辑分区**字节数，不是实际 HCCL 注册容量。
`PEER` 的地址已经跳过 60 KiB 异常 dump 前缀；不要再次加该前缀。
`workspace_bytes` 是同源布局推导出的所需空间，不是运行时 allocator 的分配记录。
本补丁不能证明非零 peer 地址仍有有效映射，也不能检查远端实际 bs（检查统一容量 N）；当前009各 rank bs=N=256。

检查失败时先打印 `MM_DIAG_INVALID` 或 `MM_DIAG_ADDR ... valid=0`，再调用 `AscendC::Trap()`，
避免已识别的非法访问继续执行。此时看到 Trap/其他 rank 超时属于诊断主动终止的后果，
应定位它前面的第一条非法记录。故障时 printf 可能不完整，不能把没有非法记录解释为全部检查通过。
诊断还会在 count 总量超过上限时中断，区别于原 Kernel 的裁剪处理；这正用于核对当前009“实测未超限”的前提。

所有地址日志位于对应搬运**提交前**，不是该指令的硬件完成证据。
`*_return` 标记仅表示 Scalar 执行到函数返回处，不能等同于异步流水完成。

## 边界审计与验证范围

| 路径/角色 | 异步流水 | 打点语义/现有边界 |
| --- | --- | --- |
| MTE A8W8 Dispatch，AIV1，prefetch 开/关 | MTE2 / MTE3 | 搬运前日志；route 解析已有 MTE2_S，未新增事件 |
| MTE A8W8 Wave Combine，AIV1，非量化/量化 | MTE2 / MTE3，量化额外有 V | 搬运前日志；metadata 解码已有 MTE2_S，未把 MTE3_MTE2 当完成边界 |
| 最后专家的 Combine，可能包含 AIV0 | 同上 | 与 AIV1 使用同一检查，核过滤只限制普通日志 |
| 主流程阶段，各角色 | 多流水 | Scalar 位置标记，无阶段耗时/完成时间推断 |

此次仅引入故障维测，没有新增 trace 协议、计时报告或输出缓冲复用。
URMA 不属于本次目标，不启用这些检查。

本机可运行：

```bash
python3 mc2/mega_moe/tests/test_mte_diagnostics.py
git diff --check
```

CPU 测试编译实际诊断 helper，验证恰好到边界、越界、整数回绕、空基址、核/容量过滤、
日志限额耗尽后仍检查、宏关闭不产生诊断字符串。它使用 CPU stub，不代替 AscendC 编译和上板测试。
当前机器没有 CANN 工具链/NPU，尚未完成设备编译或复现。
