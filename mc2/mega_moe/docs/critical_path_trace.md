# MegaMoe 关键路径打点

这套诊断用于定位 GMM1/GMM2、ready fan-in、Dispatch、Combine 和末尾同步之间的实际依赖。
它只在事件发生时更新每核的少量 Scalar 字段，kernel 有效工作结束后再统一写入 `y`，避免在热路径上
反复写 GM。诊断运行会占用 `y` 的开头，因此只用于性能和时序分析，不做输出精度比较。

## 当前分支默认配置

不需要传 `--ops-compile-options`。runner 在执行前会检查源码默认值，运行后还会检查 kernel 写出的
trace header，从而识别旧安装包、错误 tiling 或错误编译路径。

| 配置 | 默认值 |
|---|---|
| `MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE` | `1` |
| `MEGA_MOE_WEIGHT1_INTERLEAVED` | `1` |
| `MEGA_MOE_GMM2_READY_MODE` | `MEGA_MOE_GMM2_READY_PER_EXPERT` |
| `MEGA_MOE_GMM2_FANIN_MODE` | `MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE` |
| `MEGA_MOE_GMM2_READY_CORE_STRIDE` | `16` 个 `int32`，即 64B |
| `MEGA_MOE_GMM2_FINAL_BATCH_COMBINE_ONLY` | `0`，所有 wave 都执行 Combine |
| `MEGA_MOE_GMM2_GM_READY_SMOKE_TEST` | `0` |
| `MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG` | `0` |

`expertsPerWave` 没有被编译宏写死，仍由 tiling 根据 tile 数和实际 AIC 数计算。固定 case 的 trace
header 应报告 `aic=36`、`experts_per_wave=5`；runner 会对此做运行时校验。

## 构建和运行

在算子仓根目录直接构建，不附加额外编译选项：

```bash
bash build.sh --ops=mega_moe --soc=ascend950
```

按原有流程安装这次构建的算子包后，仍在算子仓中运行：

```bash
python3 mc2/mega_moe/tools/run_critical_path_trace.py \
  --test-script /home/mace/workspace/moe_test/mega_moe/mega_moe.py
```

`--test-script` 的上述值也是默认值，所以路径不变时可省略。runner 不修改原测试脚本，而是在内存中
固定本次分析 case：`bs=512`、`topk=8`、每 die 16 专家、2 die、`h=4096`、GMM1 的
`2n=4096`、全 36 AIC、balanced token。可先只检查源码宏：

```bash
python3 mc2/mega_moe/tools/run_critical_path_trace.py --check-only
```

源码宏检查只检查当前 checkout，不代表已安装的 kernel。runner 会依次从自身路径和当前工作目录向上
寻找源码仓；若 runner 被单独复制到服务器，会跳过该检查，并在 kernel 执行后以 trace header 验证
实际安装包。也可以显式指定源码仓，或者主动跳过预检：

```bash
python3 /path/to/run_critical_path_trace.py --repo-root /path/to/ops-transformer
python3 /path/to/run_critical_path_trace.py --skip-source-check
```

`--check-only` 必须能够访问源码；它不能用于检查已安装包。要使用原测试脚本当前的 shape 而不是上述
固定 case，增加 `--keep-test-config`。

## 输出

每个 rank 会在测试脚本目录生成两份文件：

- `mega_moe_critical_path_rankN.html`：自包含的 SVG timeline，AIC、AIV0、AIV1 各 36 行；鼠标悬停
  可查看阶段名、起止 cycle 和耗时。
- `mega_moe_critical_path_rankN.json`：每核的原始命名时间点（相对 AIC0 `process_begin`）和四类累计
  Scalar 等待 cycle，便于进一步分析。

HTML 中高不透明度实线块的起止边界都由真实打点给出；半透明虚线块只表示两个稀疏关键点之间推断的
大阶段。例如 ready fan-in、Combine Scalar 下发/最终完成和末尾同步是精确区间，而完整 GMM、SwiGLU
和 Dispatch+Combine rolling body 是推断区间。这样能看到每个核大致在做什么，同时不会把未打点的
异步流水伪装成精确时间线。

`publish_end` 是 `FIX->S` 完成并调用 bypass GM store 后的 Scalar 边界，不等价于其他核已经观察到
该 GM 值；可见性延迟会自然落在 publish 到 AIV1 `ready_end` 的 gap 中。类似地，
`combine_scalar_end` 只是六槽 drain 已下发的 Scalar 边界，只有最后一轮的
`last_combine_actual_end` 通过 `MTE2->S` bridge 表示远端发送真正完成。

终端同时打印 header、各类核的时间点和 span 的 min/median/max、累计等待，以及 publish 到 ready、
ready 到 Combine 等派生 gap。请把完整终端输出和两个 rank 的 HTML/JSON 一起返回。

注意 profiler 看到的算子总耗时还包含末尾 trace 汇合和统一写出；分析真实工作尾部时优先看内部
`process_to_work_end`，不要把诊断写回开销误认为 Compute/Combine 尾巴。
