# D/C 流水与通信下发优化

## 约束和范围

32 Jetty 只对应32个发送 owner。Dispatch prepare 继续使用其余 AIV，两个 epilogue
维持原有全部可用计算核分片，不固定成32核，不改变 host tiling 或窗口排布。
保留已通过单跑/continue 的 A 点 V_S → S_V，不增加 assert、printf、返回值校验或队列深度校验。

本轮基线为用户报告的最新均值：Dispatch total 1150.660 us，Combine total 1425.085 us。
以下是源码实现及静态验证，不是新版本的设备性能结果，不能承诺 Dispatch <50 us。

## Dispatch metadata 批处理

原 ProcessMetaSlot 对每个 token 分别搬入 TopK、weight、scale，写 source rank/token，
再将一个完整 meta slot 写到 stash。新版本最多8个连续 token 一批：

- 用 strided DataCopyPad 直接填充原 meta slot 格式；GM 输入仍为紧凑 TopK/weight/scale。
- 批量写 stash；命中本卡的 meta 仍按原压缩 slot 序逐项写本地窗口。
- 复用 SetSrcTokenIdxAndDstSlot 已结束使用的 topkIdsBuf，不新增数据 UB，删除原 metadata queue。
- 每批复用前等待 V/S/MTE3 到 MTE2 的依赖；每批搬入结束后等待 MTE2_S。
  group 结束的 MTE3_V 保证下一轮 routing scratch 不覆盖尚在写出的 metadata。
- 实际 tile 大小不超过 scratch 容量和剩余 token；不依赖 BS4096、8P 或固定总 AIV 数。

GM→UB 的目的地址步长保持为 align32(fieldBytes)+dstStride*32=metaSlotBytes。
典型 K6、BF16、带 weight 时，完整8-token tile 的 TopK/weight 搬入调用由16次减为2次，
stash 写出由8次减为1次。本地命中写出仍保留；这只是静态指令数量变化，不是耗时倍率。
tile=8是本轮候选值，不宣称已由硬件测得最优。

## Combine epilogue 输入预取

原路径每路 DataCopy→EnQue→立即 DeQue→Cast/Add。新路径最多保持两路有效输入在队列中：

- x/weight 一起预取，FIFO记录对应 topkId；跳过无效或缺失本地输入的原有语义保持不变。
- 严格按原 TopK 顺序执行 FP32 累加及最后的输出 cast，不做树形归约或重排。
- 使用已有 CalcBufferNum 结果；仅一组 buffer 时只保持一路，不新增数据 UB 分配。
- 输入队列和 weight 队列的逻辑深度改为2，实际 pending 不超过 min(bufferNum,2)。
- token 切换前队列排空，避免 weight 与下一 token 配对错误。

该改动意在让下一路搬入与当前路计算重叠；是否真的覆盖等待须由设备运行验证。

## 本地验证和待测范围

- git diff --check。
- CPU调度模型覆盖 K=0..10 所有有效位组合、深度1/2，共4094组，检查 FIFO、空输入、尾部与容量。
- metadata地址模型覆盖 K=1..32、7种 scale 字节数、tile=1..8，共1792组，检查字段stride和scratch容量。
- 独立逻辑、Clean Code、架构、完整性审查。移除审查发现的废弃 topkOffset 参数。
- 未运行 CANN 编译、NPU 或远端操作；模型不覆盖设备事件分配、DMA异步行为或编译器实现。

## 通信下发增量

Dispatch 的 GetTokenRange 合并起点/终点的计数前缀扫描：先累加到起点的 prepare，
再延伸至终点，不重新读取起点之前的所有计数。边界落在 prepare 起点时仍直接返回前缀，
不读取尚未 ready 的下一分片；空 BS、空 sender、尾部规则保持。

Dispatch 每个远端 peer 只在入口解析一次 GetHandleRef，payload/flag 使用同一个
payloadBatch_.handle；Combine 在 BeginPreparedWrites 切换 peer/window 时解析，
同 peer 后续请求和 flag 复用 activeBatchHandle_.handle。没有复制 handle/cursor，
没有为一个物理 SQ 建立多个独立提交状态，跨 peer 的 batch 仍然复用。

依据：host RegisterDeviceBuffer 注册整个 MoE window，ResolveRemoteWindows 用匹配的
bufferTag 取得远端基址及覆盖容量；payload 和 flag 均位于该注册窗口。
本地 asc-comm 的 GetHandleRef 解析注册信息到内嵌 handle；BatchPostSend 编码当时 peer 信息，
BatchCommit 提交已编码 WQE 并更新 cursor，不重新解析 peer。因此解析可移出逐 WQE 热循环。
这依赖当前单注册窗口约定；未来若把 payload/flag 拆成不同注册内存，必须重新按段绑定。

独立 CPU 调度测试新增共享前缀模型，与旧双边界算法及完整 sorted-list oracle 对照，
覆盖 prepare=1..68、BS=0/1/7/31/32/33/4095/4096/4097、P=2/4/8/16/33；
MSVC C++17 /O2 /W4 /WX 编译和运行通过。测试中的 assert 仅存在于 CPU 测试，
没有加入设备 kernel。部署的 Hcomm SDK 类型/语义仍需通过目标环境编译和回归核实。

设备回归应先跑既有单跑/continue，再覆盖 cached、权重开关、支持的 FP8/scales、空输入、
尾块和多group。通过后对比六项计时及慢rank，不只看提交kernel。

## 尚未实施

本轮没有修改 Dispatch epilogue 的 hidden 预取，也没有加入 chunk-ready、接收侧增量消费、
共享反向索引、全局prefix表重构或改变batch提交阈值。它们仍有优化空间，但会引入不同生命周期/
完成协议或调参变量，不能把当前增量描述为“全部优化完毕”。

参考 D:/code/cannbot-skills/ops/ascendc-perf-optimize 的单核访存资料：合并小搬移、
区分通信等待与本地访存。现有汇总计时不足以证明 memory bound，本轮不据此调整计算核数。
