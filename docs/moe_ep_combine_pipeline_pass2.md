# Combine 第二轮：metadata 预取与冗余 mask 清理

## 基线与结论边界

修改基于 `0f88036e0c`。本轮用户反馈性能继续改善、continue 120/120 通过，但贴出的六项计时
与此前 `ce1e81d7` 那组完全相同（Dispatch total 1356.577 us、Combine total 1986.270 us）。
因此不把这组重复数据当成本轮代码的性能增益证据；本文件描述的是后续源码改动，不是设备测速报告。

约束保持：32 Jetty、SQ4096、CQ64；每个 sender 独占一个 Jetty；epilogue 使用平台实际全部计算核。
没有修改通信窗口、slot 布局、完成协议、TopK 累加顺序或 public tiling/context ABI。
Dispatch 的 `DedupAndSend` 后 `V_S → S_V` 修复不动。

## 检查代码排查

扫描范围是四个 `moe_ep_* / op_kernel`、共同同步 helper，以及本地 `asc-comm` 的实际 batch 调用链。

| 类别 | 结果与处理 |
| --- | --- |
| 算子 kernel 的 assert / Trap / printf / 返回值包装检查 | 未发现；未加入新的设备诊断 |
| Combine 的 `(void)` 未用参数表达式 | 改为未命名形参；本身不是性能收益 |
| `dumpMetadata` / dump defs include | 仅保留 tiling 数据结构字段，没有设备 dump writer；不为改名破坏 ABI |
| Hcomm `static_assert` | 编译期检查，没有运行时分支；保留 |
| Hcomm 内部 SGE、batch、SQ 参数分支与错误日志 | 依赖库仍存在；不把“算子未检查返回值”误报为“整个调用链无检查” |
| Host shape/window 校验 | 不在 AIV 热循环内；本轮不删除 |
| 有效 TopK / 本核 token 范围 / 空分片过滤 | 功能逻辑保留；冗余 mask 构造改为直接判断 expert ID |
| batch 容量判断、完成轮询、cache 发布与必要同步 | 属于数据安全/协议，不按调试代码删除 |

本地依赖依据：`D:/code/asc-comm/src/aicore/hcomm/impl/platform_v310/hcomm_aiv_urma.h`
的 `GetHandleRef`、`BatchPostSend`、`BatchCommit`。实际安装 SDK 是否相同，仍需目标构建确认。

## 已实现：跨 peer 的 metadata 双缓冲

旧 `ProcessRemoteMetadataRange` 对每个 peer 的 chunk 先搬入、等待，再逐条读取 metadata 构造 WQE。
典型 8P、BS4096、K6 且均衡时，每源 rank 约 3072 条 expert row，分给32个 sender 后每个约96条，
小于原2048行 chunk；只在同一 peer 的后续 chunk 上做预取，典型场景可能完全用不上。

新流程保持原来 `(sender + peerOffset) % worldSize` 的错峰遍历：

1. `NextMetadataChunk` 生成下一个 peer/chunk 描述，包含是否为该 peer 最后一块。
2. `PrefetchMetadata` 提前把下一块（可以属于下一 peer）搬入另一块 UB。
3. `SendMetadataChunk` 消费当前块时才绑定其 Hcomm peer，并编码原 payload WQE。
4. 仅 peer 的最后一块发送原 completion flag；空 peer 和本 rank 仍各处理一次 flag。

每块1024行、每行5个 int32，两块最大40 KiB，与旧单块2048行的最大 metadata 预算相同。
这是最大预算不变；较小 actualA 下，两块缓冲的实际占用会高于旧单缓冲，不声称所有 shape 的占用相同。
最大1024 ranks 时，声明的 Combine UB 合计为62496 B；不把该静态求和当作目标编译器分配证明。
大于1024行的单 sender/peer 分片会增加 chunk 次数，需要在大 BS/偏斜场景比较实际收益。

两路 `MTE2_S` event 用 `AllocEventID` 持有，不写死 ID。非空块分别 Set/Wait，空块不产生事件；
全部消费后释放。复用前保留 `S_MTE2`，防止 DMA 覆盖上一轮仍在标量读取的 metadata。
事件成对且从框架分配的原则见 [Ascend C SetFlag/WaitFlag 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0270.html)。

这使下一块搬入具备与当前块 WQE 构造重叠的条件；它不减少网络 payload 字节，也不保证硬件一定完全重叠。

## 已实现：Combine epilogue 不再单独生成 mask

旧 mask 链是 int32 expert ID → float → 比较 >=0 → select → uint8 mask；
每路输入再读取 mask 和同一个 expert ID。现在直接读取 expert ID，并在其小于0时跳过。
int32 转 float 的精度损失不会改变符号，所以有效项谓词等价。

删除五条 mask 向量 API 调用、对应同步与专用 UB，删除 host 中 kernel 从未使用的额外 mask workspace。
系统 workspace 保留。输入预取、缺失 local entry 的过滤、每 token 的有效 TopK 顺序及 FP32 累加不改。
空核仍参与原 completion/barrier；不是把 epilogue 固定为32核。

## 带宽空间与后续实现方向

只有总时延，不能分离 URMA 吞吐、发送供给、接收等待与 GM 归约开销。
若 BF16、均衡路由、每个 expert row 各回传一次，Combine 单卡远端 hidden 的估算为：

`BS × K × H × 2 × (P - 1) / P`。

8P/4096/6/7168 对应308281344 B；假设400 GB/s是每卡单向有效带宽上限，纯 payload 理想时间约770.7 us，
不含协议、weights、padding和本地reduce。用旧1986.270 us总时延相除约155.2 GB/s，
用此前另一组1425.085 us相除约216.3 GB/s；两者都是端到端有效 payload 速率，不是网卡计数器利用率，
更不是本轮新结果。其他P应使用实际远端 row 数，不写死8P。

还有实现空间，优先级如下：

- **减少发送供给空隙**：本轮跨 peer 预取已经落地。后续可比较固定1/2-SGE的 WQE 编码成本，
  但不要在算子里复制 SDK 私有 SQE/doorbell 实现来跳过分支；它会引入 ABI、内存发布和环形队列风险。
  优先在明确 SDK 版本与所有权契约下做库级 fast path，并用完全相同 payload 做独立 A/B。
- **减少 epilogue 重复 metadata 扫描**：当前每个计算核扫描本 rank metadata 来构建自身 lookup。
  共享逆向索引有潜力，但必须明确生成方、workspace 生命周期、跨核发布和 cached/standalone 兼容性，
  不能仅删除扫描或其边界过滤。
- **让接收计算更早开始**：现在仍等待全部 sender completion 再归约。BS/chunk readiness 有更大重叠空间，
  但这会改变完成协议，需要连同多轮清零、空分片和异常路径设计及验证，本轮没有悄悄修改。
- **减少网络字节不是免费优化**：同卡多 expert 先预加后发送会改变原 TopK 的浮点累加顺序，
  不能在要求原顺序的前提下作为等价优化加入；压缩/量化同样需要新的数值约定。

参考 `D:/code/cannbot-skills/ops/ascendc-perf-optimize` 的单核访存资料用于识别小搬运与流水机会。
没有隔离通信/计算数据或阶段时间线，未宣称 GM memory-bound 或 URMA saturated，也未据此重切计算核数。

## 验证

- `git diff --check` 通过。
- 新增 `test_combine_pipeline.cpp`，MSVC C++17 `/O2 /W4 /WX` 编译运行通过8192组调度模型。
  覆盖 P=1/2/3/8/16/33/64/1024、所有32个 sender、空/少量/不均衡/大尾块，chunk=1/7/1024/2048；
  对照旧按 peer 顺序的独立 oracle，检查行覆盖、completion顺序及每路event的Set/Wait账目。
- 另检查 int32 正负边界的 mask 谓词等价。CPU断言只在测试中，不进入设备 kernel。
- 逻辑、Clean Code、架构、完整性独立审查；采纳命名清理，`maskTokenNum_` 改为 `localTopkCount_`。
- 未执行 CANN/NPU/远端操作。模型不能覆盖DMA异步时序、硬件事件资源或真实带宽。
  需重新编译并验证单跑、continue、cached、weights开关、小BS、不同P、偏斜与多chunk，之后再对比六项计时。
