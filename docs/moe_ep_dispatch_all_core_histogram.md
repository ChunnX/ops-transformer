# Dispatch 前置计数：全 AIV 分工与寄存器直方图

> 第一阶段记录：寄存器直方图仍沿用，但文中按 prepare 交集原子累加的调度已被 tile 独占计数和共享前缀替代。
> 当前实现与验证见 [Dispatch tile 流水实现](moe_ep_dispatch_tile_pipeline.md)。本文保留原设计与当时验证背景。

## 修改范围

计数优化最初基于 `84d0aba660` 的 epilogue 同步测试版本，只修改 Dispatch 前置计数及进入 metadata 准备阶段的分片切换。
后续按要求撤销该提交的全 CQE、逐批 Drain 和 sender 入站等待，恢复由 epilogue 等待 payload 完成。
32 Jetty、SQ4096/CQ64、两个 epilogue、窗口布局和 Python API 均不变；详见
[Dispatch 尽早退出方案](moe_ep_dispatch_early_exit_plan.md)。
host tiling 只更新 workspace 注释，不改变大小、字段或偏移。

`BufferInit()` 原本就在所有 AIV 上执行。原 `CalSendCnt()` 只让 `aivNum - 32` 个 prepare 核参与；
本轮改为全部 AIV 按 BS 均分计数，计数结束后再恢复原来的32个发送核和其余 metadata 准备核分工。
BS 小于 AIV 数时，只有有 token 的核计算，不为使用全部核重复计算 token。

## 算法与数据依赖

计数语义不变：expert count 对每个有效 TopK 项计数；rank count 对同一个 token 的同一目标 rank 只计一次。
同卡多个专家仍分别计入 expert count；未缓存路径保留最早命中的 TopK 位置，重复/无效项写为 `-1`。
cached 路径沿用 `cachedSlotIdx >= 0` 的选择语义，并原样复制 slot index。

### 全核计数但保留 prepare 行

每核得到 `Partition(BS, aivNum, aivId)`。这个区间与原 prepare 分区相交后，逐段计算：

1. 处理交集中的完整 token，生成 dedup rank 数据；token 不跨计数核切开。
2. rank 局部计数原子累加到 `sendCounter[prepare][rank]`，并累加全卡 rank count。
3. expert 局部计数跨交集累计，每核最后一次性原子发布。

原 prepare 核先在 `ResetCounters()` 中清零自己的计数行；沿用 `BufferInit()` 末尾已有全核屏障，
保证初始化完成后才开始原子累计。沿用 `CalSendCnt()` 后的全核屏障，保证后续通信和 slot 前缀读到完整计数。
发布一个交集的 rank count 后，用 `MTE3_V` 等待搬出完成，再清零/复用该 UB 计数。

`SendPhase()` 显式重新设置原 prepare BS 区间。`GetSlotStartNum()`、`GetTokenRange()`、
source-token 列表排序及 prepare-ready 覆盖关系不需要改变，也不新增 workspace。
这是保留后续布局的必要步骤，不能只把原来的 `if (!isPrepareCore_) return` 删除。

### 直接以寄存器掩码做直方图

原代码已在底层调用 `Reg::Histograms`，但外围经过掩码生成、`GatherMask` 压缩、
`V_S` 取有效项数、UB 的 int16→uint8 转换，再调用直方图。
新实现将 rank/expert 两套外围逻辑合并为 `AccumulateSendCounts()`：

- 每次装入两组128个 int16，减当前256-bin页的基址；用无符号范围比较同时排除负数和页外数据。
- 在寄存器内分别 Pack 数据及其掩码的低/高半区，得到256个 uint8；无效项不能因截断混入 bin255。
- 使用 `Histograms::FREQUENCY`；当前页不超过128个 bin 时只调用 BIN0，其他情况调用 BIN0 和 BIN1。
- 输出 histogram 扩为32位并累加到原计数向量。循环处理全部页，不限制为8P或256个专家。

尾块的 `UpdateMask` 只修改自己的剩余计数变量，循环次数预先计算。每次最多处理
`floor(16384 / K) * K` 个 TopK 项，uint16 histogram 单 bin 最大值不超过16384；每块及时扩32位，
因此大 BS 的多块累加不会在 uint16 中累计溢出。

新实现删除计数路径的 `GatherMask` 压缩、相关 `V_S` 标量往返、uint8 中间 UB 和两套重复的分桶代码。
原去重算法只去掉最后一列的无效比较轮次（该列没有更晚项可去重）；没有用原始 rank 的直方图冒充去重计数。
此前修复 BUG 的 `V_S → S_V` 和必要的 MTE 同步保留。没有新增设备日志、断言或返回值检查。
主代理复核还将去重入口已有的 `MTE3_V` 移到 scratch 的 `Duplicate` 之前：下一组较小时，scratch起点可能
落在上一组尚未搬出的 rank 数据内；必须先等搬出再覆写，不能只在后续 `Div` 之前等。

## API 依据

用户提供的 hiascend `latest/.../Histograms.md` 链接本次抓取失败，交叉核对采用官方 asc-devkit 仓自动生成的文档：

- [Histograms](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/histogram_compute/Histograms.html)：
  uint8 输入、uint16 计数、BIN0/BIN1 各128个 bin、掩码过滤和累计溢出约束。
- [数据 Pack](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/data_compression/Pack.html)
  与 [掩码 Pack](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/MaskReg_compute/Pack.html)：
  数据和有效位必须按相同半区一起打包。
- [Compares](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/compare_and_select/Compares.html)
  与 [MaskReg](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/register_data_types/MaskReg.html)：
  比较的未选中位为零，UpdateMask 会递减剩余长度。

这些是开发分支文档，不代替本机 CANN 头文件和编译器兼容性验证。

## 性能预期与限制

以56个 AIV、BS4096、K6为例，原24个 prepare 核每核最多171个 token（1026个 TopK项），
现在56核每核最多74个 token（444个 TopK项）。这是工作量变化，不是实测加速比。
新寄存器路径减少中间 UB 访问及标量同步；P8的 rank 计数不再执行无用的 BIN1。

代价是 count 分区跨 prepare 边界时产生额外小分段及计数原子累加。实际收益受小块固定开销、
atomic 竞争、全核屏障和 K/P/E 分布影响，不能保证所有小 BS 都加速。BufferInit 的全局初始化屏障没有删除，
因为后续使用共享原子计数；去重与 metadata 准备也没有改成新的协议。

当前已恢复异步发送，Dispatch 仍包含计数、metadata 准备、WQE 提交和接收计数等待，
不再主动等待 payload 完成。不能把总耗时直接当成 CalSendCnt 单独耗时；对照基线应为
`84d0aba660^` 的异步路径，不应直接用同步测试版本单项计时推算计数算法加速比。

## 验证

- `python -B mc2/moe_ep_dispatch/tests/ut/op_kernel/test_dispatch_count_histograms.py -v`：8项CPU算法模型测试通过。
  覆盖 AIV33/40/48/56/64/80/100、P2/3/8/33/256/1024、K1/6/7/32、cached/uncached、
  小BS空任务、全无效/重复专家/同卡多个专家、128/256边界、尾掩码与UB范围、大BS分块和计数行/slot前缀一致性。
  cached另有独立slot选择oracle，覆盖同rank多有效项及任意混合slot掩码；小BS空任务只指空核片，host禁止BS0。
- 初版曾通过9项同步完成队列模型测试；它们不适用于恢复后的异步语义。
  当前回归入口为 `python -B mc2/common/tests/test_moe_ep_async_completion.py -v`，12项通过。
  其中DoCpuSync条件/唯一actualA发布与cached slot原样拷贝仅做源码契约验证，不是host pinned/NPU运行验证。
- `git diff --check`通过；新增和修改的生产函数均不超过50个非空非注释行。
- 源码契约检查确认全AIV计数、prepare分工恢复、原子发布与初始化屏障、cached slot透传、原去重同步及
  scratch覆写前MTE3等待；host tiling去除注释后与基线一致，epilogue/context未变，
  Combine仅回退同步测试模式，恢复为 `84d0aba660^` 的实现。
- CPU模型不模拟 Reg 指令、编译器调度、DMA 可见性或真实 atomic 行为，不是 NPU 精度/性能证明。
- 本地 PATH 未找到 CANN `ccec`/`bisheng`，没有执行 CANN 编译/NPU验证，也未操作远端。
- 初版独立 DeepSeek reviewer 调用及一次重试均被服务拒绝，未完成独立审查。
  本轮按用户明确要求改用 `gpt-5.6-sol` 对累计改动独立复核；结果记录于尽早退出方案文档。

上机仍需验证 standalone/cached/continue 的 expert/rank计数、slot索引、recv metadata、数据精度及性能，
特别覆盖此前的BS4096/H7168/K6/8P，以及非2次幂专家分布和大于256的bin范围。
