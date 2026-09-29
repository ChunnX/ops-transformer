# 验收摘要与 PR 范围

本 patch 仅交付源码、脚本、说明和流水图，不携带历史运行日志、设备 dump、构建清单或完整反汇编产物。
以下保留本次实测结论、ELF hash 和关键 PC；需要原始现场时，请运行文末命令重新生成。

## 环境与构建来源

2026-09-29，Ascend 910B3 / NPU 0，Linux aarch64，CANN 9.1.0，Python 3.11.14，
torch 2.7.1、torch_npu 2.7.1.post5，GCC 11.4。仅用 CANN 9.2 objdump 解码 dav-c220，
没有切换到 9.2 运行时。仓库基线 commit：`11e7d2716d04e051677cf727892a5dc82d5ecddf`。

本次从新仓库源文件和本机 SDK custom-op 模板生成 Host、ACLNN、FP16/BF16 设备 ELF，
没有输入旧 OPP 安装包，没有修改 CANN 文件。torch 适配层从新仓库 JIT，运行使用唯一 custom OPP 路径和新进程。
设备回归只覆盖指定 BF16 场景；FP16 已编译但未做上卡回归。生产整网、其他 SoC/SDK 不在本次验证范围。

构建证据在生成目录各配置的 `manifest.json` 中：包含输入源码 hash、Host/ACLNN/设备产物 hash、
BF16 审核摘要。实际 ELF 与迁移前相同配置相同：

| 配置 | BF16 ELF SHA256 |
|---|---|
| pipeline | `a736764eb8b52b828dceee399a7890fc4c60182062c030a904aab5a73b15cb87` |
| same_r1 | `438ac7e87c303ddf605f2d472f7f040c421cdf7235887ec84bcaefbaf4ee72e0` |
| split_r1 | `1b2022ebfc92ea95061fbf08d840f3695907dfb1877695433de414f5a353654c` |
| same_r2 | `7dd4b48e8a053102e2c2afb17248276ea182be82372fd3a94e33647eb109f24f` |
| split_r2 | `8d907f4d23a679d24fc696a70d990c15f5ecfb202b4a9756c48cf07c35f80d76` |
| same_r1024 | `797a853a43ec235c3a99b89a3b1e0354cf1ff61bda2cce746bb91f61b0bad4c1` |
| split_r1024 | `a5ca18027e3f2b51433803a532a0b8f57a587728d22ba27cf899afa6305b0731` |

这里对齐的是**相同配置迁移前后**的 ELF，不是声称 same/split 两个不同配置的 ELF 相同。
默认 pipeline 的同步源代码未调整；Host 仅将外部 opbase 日志/检查宏替换为本地等价诊断检查，
torch 适配按新仓库目录/初始化接口迁移，并对齐 SDK 生成的 ACLNN `double scaleValue` 签名。

## 核心观测

最终扩展验收共 **14 个独立进程、7 种配置**，矩阵退出 0。
每个通过进程完成 1000 次调用；每个故障进程均在首次调用报设备 507014，
不是 Host watchdog。以下偏移均为实际 ELF 的 `.text` 节内地址。

| 配置 | 第 1 个进程 | 第 2 个进程 | 实际故障 Wait PC |
|---|---|---|---|
| pipeline | 超时 | 超时 | 第 1 次 AIC `0x2b4`；第 2 次 AIC `0x2b4` + AIV `0x718` |
| same_r1 | 1000 次通过 | 超时 | AIV `0xf4` |
| split_r1 | 1000 次通过 | 1000 次通过 | 无 |
| same_r2 | 超时 | 超时 | 两次均为 AIV `0x104` |
| split_r2 | 1000 次通过 | 1000 次通过 | 无 |
| same_r1024 | 超时 | 超时 | 第 1 次 AIC `0x70` + AIV `0x118`；第 2 次仅 AIC `0x70` |
| split_r1024 | 1000 次通过 | 1000 次通过 | 无 |

7 个超时进程生成的运行时 ELF dump，SHA256 **全部与该次构建的 BF16 ELF 一致**。
现场匹配到了哪个核就记录哪个核，不补写未采样到的 PC，也不将不同进程拼成一张全核快照。
三组 split 共完成 6000 次调用；这支持本实验的隔离对照，不代表完整训练算子永久安全。

重新构建后，可在 `.klloss_repro/<配置>/kernel.disasm` 逐条核对指令，
同目录的 `manifest.json` 记录该次构建的 hash 和审核结果。关键指令也已摘录在客户说明中。

### pipeline 的寄存器 Wait 怎样对应到事件表

反汇编中的 `WAIT_FLAG_DEV X19/X25` 没有立即显示 ID，需结合相同 ELF 的 `.rodata`，
按常量地址和 load 链核对。本次实测 ELF 的对应关系如下：

- AIC：`.text+0x270` 装入偏移 `0x5e4`，加 `.text+0x280` 的 PC 得 `0x864`；
  `.text+0x298` 从此表按 parity 取一个字节到 X19，`.text+0x2b4` Wait X19。
  `.rodata+地址0x864` 的字节是 `08 09`，即 C2 所等 DW 的候选表。
- AIV：`.text+0x51c` 装入偏移 `0x342`，加 `.text+0x52c` 的 PC 得 `0x86e`；
  `.text+0x6f8` 从此表按 parity 取 X25，`.text+0x714` 发 mode 0，`.text+0x718` Wait X25。
  `0x86e` 的字节是 `00 01`，即 V2 首道 mode 0 的候选表。
- pipeline wrapper 在 `.text+0x168` 装入 `0xf21`，随后从 MTE3 Set，
  这是 mode 2 / flag 15 的隐式 workspace 通知。独立 minimal 的 AIC 没有这个 Set。

这些是静态候选表，不是现场 X19/X25 的寄存器采样值；不能把某次报错直接断言为 DW=8 而非 9。

## 关键反例：单轮不是安全基线

分析必须同时保留下面两条观测，不能只挑符合先前判断的结果：

- 迁移前的同 ELF 单轮实验连续 1000 次通过；最终 14 进程矩阵也出现单轮 1000 次通过。
- 新仓库扩展矩阵也观察到相同 SHA256 的单轮 ELF 首次调用超时。
  `pc start=...0074`、`current=...00f4`，对应唯一的 AIV `WAIT_FLAG_DEV #0`。

当时脚本把单轮误设为“应通过”，所以按规则返回失败并停止，**这个停止结果被保留为反例**，
没有包装成一次成功矩阵。后来修正的是验收假设，不是设备协议：单轮 same 改为探索项，
接受完整通过或准确匹配 Wait 的设备超时，仍拒绝普通异常/watchdog。

这明确否定了“至少两轮才会触发”“必须跨轮累积”的必要性。相同输入/同一 ELF 可有不同结果，
也正是这类整网偶发问题复现困难的直接例子。后续需按实际参与核、管线和事件到达时序分析。

## 如何判定复现，避免假阳性

`reproduce.py` 要求：

1. 构建清单完整、实际安装产物 hash 一致、已反汇编审核，并使用独立进程/唯一 custom OPP。
2. 故障组已进入设备调用，退出非零，日志明确有设备 `507014`。
3. 解析日志的 AIC/AIV `pc start/current`，按对应符号起点换算节内偏移，匹配本次 ELF 的 Wait。
4. 对照组正常返回 0，并通过脚本的输出形状/设备/零值检查。
5. 外层 watchdog（124/137）、导入/编译异常、无 PC 或 PC 不匹配均不计为目标复现。

若 CANN 自动生成故障 ELF dump，矩阵还会校验 dump hash 与构建 ELF 一致；不一致立即停止。
每个进程的 dump 放在该配置 `runtime/<PID>/`，不把临时二进制混入 PR。

默认核心矩阵严格要求 pipeline / same_r2 复现、split_r2 通过；扩展矩阵还要求 same_r1024
复现、split_r1 / split_r1024 通过，单轮 same 单列为探索项。任何非预期结果均停止，
不自动重置设备、清除现场或无限重试。它是本机回归期望，不是跨平台必现保证。

11 个 CPU 单测全部通过，覆盖：数组跨 mode 候选、宏覆盖、未知运行时 ID、注释过滤、默认 Wait 防误报、
实际 pipeline 的 0/1 候选、精确 PC 换算、watchdog/普通异常拒绝、分离组通过判定及单轮探索策略。
同时检查 Set 消息字面值跟踪和未知寄存器写入后的失效行为。
skill 通过 `skill-creator` 的结构校验；扫描器只生成风险候选，不是安全证明。

## 全新输出目录补验

额外以相对路径 `SLIG_WORK_DIR=.klloss_repro/clean_verify` 从空目录重编 split_r2，
Host/ACLNN、FP16/BF16 ELF 和独立 torch-extension JIT 均完成；BF16 hash 仍为上表中的 `8d907f…`。
随后又完成 1000 次调用，退出 0，不计入上面的 14 进程矩阵。

验收后 NPU 0 的 `npu-smi` Health 为 OK、无运行进程；未执行设备 reset。
Shell 语法、Python 编译、文档本地链接、归档日志 hash 和无设备二进制混入待提交文件的检查均通过。

## 证据如何重新生成

```bash
bash scripts/klloss/reproduce.sh --extended --device 0 --runs 1000 --repeats 2
# 使用上一个命令输出的 results/<时间戳>，选择新的归档目录，不覆盖旧证据。
python3 scripts/klloss/export_evidence.py \
  .klloss_repro/results/<时间戳> docs/klloss_sync/evidence/<新的证据目录>
```

上述命令可按需另行生成日志、摘要、manifest 和反汇编，本 patch 不携带这些运行产物。
不要将 `.o` / `.so` 提交到 PR。
归档器校验原日志 hash 与矩阵绑定的 ELF hash，将本地仓库/Python 路径规范化为 `<REPO>` / `<PYTHON_ENV>`，
分别保存原始与归档日志 hash。失败/中止现场必须显式使用 `--allow-partial`，并标为反例而不是验收通过。

## PR 内容和审查重点

- 新算子诊断目录、最小实验、算子内 torch-extension 与固定场景测试。
- 仓库相对路径的 SDK 源码构建、反汇编审核、单次运行、自动验收及证据导出脚本。
- 来龙去脉、正常/候选卡死流水 PNG/SVG、绘图源码和用户参考图。
- 仓库内 flagId 审查 skill、保守扫描器和 CPU 单测；不改全局技能配置。
- 仅对根 README / gitignore 增加入口及生成目录忽略规则；不改其他生产算子。

此 PR 是复现与诊断基础设施，不应以“完整 KL Loss 修复”“所有场景不再卡死”或
“已证实具体一条 mode 0 通知被错误消费”为标题/结论。建议标题：
`[diagnostic] Add reproducible KL-loss cross-mode flagId timeout examples`。
