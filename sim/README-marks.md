> 原仿真仓说明保留供协议参考。本次 A/B 的配置、标记范围和验证状态以 [shared-start A/B](README-shared-start-ab.md) 为准。

# MegaMoE 默认 COND 打点

kernel 源码内直接使用 `set_cond(tag)`，**默认开启，无须修改 case 或增加编译参数**。
执行 `./mega-sim prepare`，或 `build` 后 `attach`，生成的 AIC/AIV bin 都包含各自路径的打点。
编号和宏集中在 `mc2/mega_moe/op_kernel/arch35/common/mega_moe_sim_marks.h`。

这是指令日志中的 Scalar 进度标记。宏只调用 `set_cond`，不分配 GM/UB，不写 workspace/tiling，
不调用 printf，不增加 barrier、SetFlag 或 WaitFlag。原 kernel 的同步和内存布局保持原样。
它与 release 参考仓的 `asc_mark_stamp<PIPE_MTE2>` / DFX START/STOP 不是同一种机制。

## 标记协议和覆盖范围

每个阶段有固定 ID：`BEGIN = 0x01000000 | ID`，`END = 0x02000000 | ID`。
例如 `0x01000209` / `0x02000209` 分别是 `WAIT_DISPATCH_DATA` 的进入/退出。
日志也可能以十进制或补零的十六进制显示这个值。rank、AIC/AIV、core/subcore 从日志路径区分，
不额外读取 rank/core 并拼入 tag。不同核、不同 rank 的同一 tag 不能相互配对。

| ID 范围 | 当前覆盖 |
|---|---|
| `0x001..0x00E` | kernel、tiling 初始化、op.Init/Process、输入准备、共享 GMM1/GMM2、MoE 主阶段、count 输出、出口同步、Unpermute、三组收尾 ACK |
| `0x101..0x108` | reset、输入打包、下标/count 发送、A8W4 count 准备、首 wave Dispatch、后续 slice Dispatch、GMM1 slice、GMM2/Combine wave |
| `0x201..0x212` | 输入 ready/UB 释放、全 AIV 输入完成、count 消费/轮次/ready、Dispatch 数据、激活数据、GMM1/2 结果及 ACK、共享 UB 交接、peer 握手及本卡同步 |

详细编号和描述直接从 header 读取，不需要维护第二份手写编号表：

```bash
python3 sim/marks.py --list
python3 sim/marks.py 0x01000209 0x02000209
```

主要针对 **730152、MTE A8W4 MoE + A8W8 shared、arch35 1c2v**。
公共 MTE 阶段也能标记其他 Wave 路径，但 A8W4 的细分 Dispatch/GMM 标记不等于其他量化或 URMA 路径的完整覆盖。
在 A8W4 中，`GMM1_SLICE` 对应 AIC 计算、AIV0 权重转换、AIV1 激活；`GMM2_WAVE` 包含各核的 GMM2/权重转换/Combine。
这些是流水交织的调用范围，不能把所有核的区间相加作为 kernel 总耗时。

## 看运行进度和卡点

内部 debug npusim 需要启用其指令日志输出；具体开关和日志目录以内部版本为准。
在实际输出的 `log_ca` 等目录找到对应 rank/core 文件：

```bash
# 单核已有记录
grep COND core0.veccore0.instr_log.dump | grep -v FLAG

# 持续观察；仿真器本身也需要及时刷新文件
tail -n 100 -F core0.veccore0.instr_log.dump \
  | grep --line-buffered COND | grep --line-buffered -v FLAG

# 在一次运行目录内找所有核的 COND 记录，保留文件名和行号
rg -n --no-ignore -g '*instr_log.dump' 'SPR:COND,' /path/to/run/logs
```

遇到问题先定位 rank，再同时对照相关 AIC、AIV0、AIV1。我们每卡 32 AIC + 64 AIV，
只看 `core0.veccore0` 无法判断整个算子是否前进。

- `KERNEL BEGIN` 后没有 `INIT END`：检查 tiling 加载、地址和初始化处的报错。
- `WAIT_REMOTE_COUNT BEGIN` 长时间没有 END：检查远端下标/count 发布、epoch，以及通信窗口。
- `WAIT_DISPATCH_DATA BEGIN` 没有 END：检查负责 Dispatch 的 AIV1、对应输入 ready flag。
- `WAIT_ACTIVATION_DATA BEGIN` 没有 END：检查 GMM1 和负责激活的 AIV。
- `WAIT_GMM1_ACK` / `WAIT_COMBINE_ACK` 没有 END：检查消费端是否完成并发布 ACK。
- `WAIT_RANK_PEER` 没有 END：对照其他 rank 的 Combine、共享计算和出口握手。

上述判断是缩小范围：还要排除日志未刷新、仿真器已异常退出等情况。
阶段可以嵌套、在多个专家/wave 重复；对同一 rank/core 按执行顺序看 BEGIN/END，
不要只按“最后一个 tag”推断所有核状态，也不要把一次 BEGIN 和别核的 END 配在一起。

**异步流水的边界：** `CrossCoreWaitFlag<..., PIPE_MTE2/FIX/V/MTE3>` 是向指定流水提交依赖。
其后 Scalar 的 `set_cond` 可能已出现，而那条流水仍在等待。
因此 `WAIT_GMM1_RESULT END`、`WAIT_GMM2_RESULT END`、`WAIT_INPUT_UB END` 等，
只能表示调用处的 Scalar 已前进，不能证明异步等待解除或数据完成搬运。
GM Scalar 轮询（如 `WAIT_DISPATCH_DATA`）的 END 则在轮询返回后执行。
真正的流水停顿与重叠仍须结合原始 WAIT/SET、MTE、MMAD 指令时间线和 PC。
已有 `temp_storage/sim_dump_to_trace.py` 将 `SPR:COND,` 识别为 `USERMARK`；
使用其转换结果时，上述含义同样适用。本工具只提供编号解码，不假定内部日志的完整格式。

## 开销和性能对比

打点在阶段/slice/wave 边界及关键等待调用前后。轮询和 backoff 的每次重试不打点，
ACK 收尾循环整体打点，也不在逐 token、逐 K 搬运循环里额外打点。
GMM 与激活/Combine 的结果交接、额度 ACK 原本按 tile 发生，所以这几组等待标记也会按 tile 重复；
大 shape 下数量会增加，不能把它们当作每个 kernel 仅一次的标记。

默认一直开启，便于日常排错和版本性能对比。新增标量指令及日志记录仍有开销，
既可能影响模型内的调度时序，也可能增加仿真器墙钟时间；目前未测得内部包上的开销比例。
不同版本对比时保持打点位置和粒度一致，记录是否新增/移动了标记。
必要时可在编译参数中显式定义 `MEGAMOE_ENABLE_MARKS=0`，或修改 header 默认值后重编译做对照。
平时不需要关闭。宏不能用于 VF/SIMT 函数内部，也不应与依赖 COND 状态的自定义逻辑混用。

## 更新 kernel 后继续使用

1. 在本仓 merge/rebase 上游代码，保留 header 和调用点。不要用整目录覆盖来代替合并。
2. 保留已有 ID 的含义；新增阶段只追加新 ID，不重用旧编号。若阶段废弃，保留历史说明。
3. 新增调用使用下面的形式；有提前 return 的路径也要考虑 END。不要为了让 END 看起来更准确而添加额外 barrier。

   ```cpp
   MEGA_MOE_MARK_BEGIN(WAIT_DISPATCH_DATA);
   WaitUntilGmFlagEquals(flagValueAddr, targetValue);
   MEGA_MOE_MARK_END(WAIT_DISPATCH_DATA);
   ```

4. 运行 `python3 sim/marks.py --check` 和 `./mega-sim test`，检查更新后的阶段语义和核角色。
   `build/prepare/all` 也会检查 ID 唯一、名称合法、每个已定义 ID 存在配对调用点。
   这是静态漏点检查，不能证明条件分支、提前返回和运行时的事件配对正确；移动调用点仍需审查。
5. 执行 `./mega-sim prepare`，或 `./mega-sim build` 后重新 `attach` 已准备的输入。
   源码 hash 包含打点 header，不能用旧 bin 验证新增打点。单纯加点不改变输入协议，
   `attach` 仍会重新核对 tiling/workspace 布局。

新构建的编号表和源码位置会写入 manifest，并随 case 的 `build.json` 保存。
更新代码后查旧日志，应使用该次 case 的表：

```bash
python3 sim/marks.py --build-json /path/to/case/build.json --list
python3 sim/marks.py --build-json /path/to/case/build.json 0x01000209
```

打点加入之前生成的 `build.json` 没有此表，应使用对应旧源码。完整 kernel 的内部 debug 编译及 8p 实跑
仍需在服务器验证；本机独立 AIC/AIV 编译小样例只验证 `set_cond` 宏可编译，不替代完整仿真。
