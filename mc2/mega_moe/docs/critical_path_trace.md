# MegaMoe 关键路径打点

这套诊断用于定位 GMM1/GMM2、ready fan-in、Dispatch、Combine 和末尾同步之间的实际依赖。
trace v3 会为每个物理 AIC/AIV0/AIV1 记录前 4 个 wave 的阶段起止，kernel 有效工作结束后再统一写入
`y`，避免在热路径上反复写 GM。诊断运行会占用 `y` 的开头，因此只用于性能和时序分析，不做输出
精度比较。

每个 wave 只保留 4 个具名 Scalar，而不是动态数组：AIC 将它们解释为 GMM1/GMM2，AIV0 解释为
SwiGLU（另两个字保留为 0），AIV1 解释为 Dispatch/Combine。固定 case 有 4 个 wave，刚好完整覆盖；
其他 shape 超过 4 个 wave 时，header/JSON 的 `wave_trace_truncated` 会明确提示只捕获了前 4 个 wave，
不会动态扩容或产生设备端索引数组。

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

每个 rank 会在当前运行目录生成三份文件：

- `mega_moe_critical_path_rankN.html`：自包含的 SVG timeline，AIC、AIV0、AIV1 各 36 行；鼠标悬停
  可查看 wave、阶段名、起止 cycle、耗时和完成同步语义。
- `mega_moe_critical_path_rankN_triplets.html`：使用同一组数据和同一个 cycle 横轴，但每个逻辑索引按
  `AIC-i / AIV0-i / AIV1-i` 三行相邻排列，便于直接检查一组核之间的交接和掩盖关系。
- `mega_moe_critical_path_rankN.json`：每核的 `waves[].stages[]`、原始全局命名时间点（相对 AIC0
  `process_begin`）、每个阶段的 begin/end 边界语义和四类累计 Scalar 等待 cycle，便于进一步分析。

两个 HTML 只改变纵向行序，不会分别归一化时间；同一 rank 的 AIC/AIV0/AIV1 都使用硬件
`GetSystemCycle` 和同一个 AIC0 `process_begin` 参考点，因此 triplet 内的相对时间可以直接比较。

HTML 中五类高不透明度实线块都是逐物理核、逐 wave 的真实边界。没有分到该 wave 任务的物理核保持
零槽，不会画出伪任务。`setup` 和少量末尾区域仍可能是两个稀疏点之间的推断区间。

逐阶段边界语义如下：

| 阶段 | begin | end |
|---|---|---|
| Dispatch / AIV1 | 当前 wave 首个专家进入 Dispatch | 末专家返回；已有每专家 `MTE3->S` 已保证本核 token/triple GM 写完成 |
| GMM1 / AIC | 本核首个 dispatch-ready 轮询通过；若存在 pingpong `PIPE_FIX` 等待，再经 `FIX->S` 桥接到 Scalar；首个 MMAD 下发前 | wave 末尾新增一次定向 `FIX->S`，表示本核实际 FIX 完成 |
| SwiGLU / AIV0 | 本核首个 AIC→AIV0 `PIPE_V` 等待通过，再经 `V->S` 桥接到 Scalar；首个 vector 指令下发前 | 末任务返回；算子内部已有每 tile `MTE3->S`，表示输出 GM 写完成 |
| GMM2 / AIC | 本核首个 swiglu-ready 轮询通过，首个 MMAD 前 | ready publication 中已有 `FIX->S`，表示直写 GM 完成 |
| Combine / AIV1 | ready fan-in 和 triple `MTE2->S` 均通过后的首条 row | 六槽 drain 后每 wave 一次 `MTE2->S`，表示远端 MTE3 发送实际完成 |

因此没有在每个点前插入 `PIPE_ALL`。v3 只在每核每 wave 的首次 GMM1/SwiGLU begin 增加一次相应的
`FIX->S`/`V->S` 因果桥，并在 GMM1 end、每 wave Combine end 保留完成桥。诊断版会增加少量串行化；
这是换取可信 begin 和 actual completion end 的明确代价，不能把诊断版总耗时当作无打点版本性能。
这些静态 Tensor 事件统一使用已完整 set/wait 回收的 `EVENT_ID0`，不使用可能与内部实现冲突的 6/7。

`publish_end` 是 `FIX->S` 完成并调用 bypass GM store 后的 Scalar 边界，不等价于其他核已经观察到
该 GM 值；可见性延迟会自然落在 publish 到 AIV1 `ready_end` 的 gap 中。类似地，
旧的 `combine_scalar_end` 仍只是六槽 drain 已下发的 Scalar 边界；v3 中每个 wave 的
`combine_end` 都通过 `MTE2->S` bridge 表示远端发送真正完成。

终端同时打印 header、各类核的时间点和 span 的 min/median/max、累计等待，以及 publish 到 ready、
ready 到 Combine 等派生 gap。请把完整终端输出和两个 rank 的两版 HTML/JSON 一起返回。

目标 shape 的 v3 trace 至少需要 `(20 + 36 * 3 * 42) * 4 = 18224` 字节输出空间。shape 太小导致
`y` 不足时，kernel 不会写不完整记录，runner 会报告无效 magic。

注意 profiler 看到的算子总耗时还包含末尾 trace 汇合和统一写出；分析真实工作尾部时优先看内部
`process_to_work_end`，不要把诊断写回开销误认为 Compute/Combine 尾巴。
