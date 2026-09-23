# Dispatch tile 流水实现

## 已实施范围

实施早退出方案 A/B/C：全 AIV 计数并准备独立 metadata；tile 独占计数、共享前缀和 expert partial 归并；
剩余计算核按 tile 准备索引/本地 metadata，32发送核边等待 tile 边提交。不是仅把 ready 通知提前。

保留32 Jetty、SQ4096/CQ64、原注册窗口与远端 slot/metadata格式、cached语义、多SGE、最终sender-peer完成flag。
Dispatch/Combine仍为异步提交，payload等待在epilogue；没有新增Drain、设备assert、printf或返回值检查。
Combine与两个epilogue没有因本轮tile方案新增改动。GetRecvCount和可选host actualA仍在Dispatch内，D方案未做。

## 执行顺序与核角色

```text
全AIV：初始化tile ready → SyncAll
全AIV：独占tile计数/去重 + 复用TopK准备注册stash + 每核expert partial → SyncAll
全AIV可分工：tile rank前缀 + expert partial归并 → SyncAll
  ├─32发送核：Communication(count) → 逐tile等待/读prefix[t:t+2]/提交payload → 每peer末flag → 返回
  └─其余核：逐tile准备索引/slot → 本地metadata搬运 → 发布tile ready
              └─最后一核完成GetRecvCount → 返回
epilogue：原协议等待全部完成flag → 重排/归约
```

计数owner为 `tile % AIV`，准备owner为 `tile % (AIV-32)`，发送owner为 `tile % 32`。
peer不是分核维度，不固定8P，也不要求64 AIV；现有host前提仍是至少33 AIV。
计数/独立metadata阶段所有有任务的AIV参与，rank前缀按8个peer列成组分摊，expert归并按8个expert分摊。
小矩阵不会为了用满核而重复计算：P8的rank scan由1个核处理，其余核可做自己的expert归并。

tile大小在host按 `min(max(floor(BS/AIV),1), floor(4096/K), floor(16384/metaSlotBytes))` 决定。
route上限限制UB索引scratch，16KiB限制metadata staging和int16 scatter offset范围；不是按卡数选择。
当前metadata slot按512B对齐：典型BS4096/K6/AIV56/bf16时，tile为32 tokens，共128 tiles，每sender4 tiles。
floor而非ceil保证BS>=AIV时不会因为切片进位导致部分计数核无任务；余数负载差以一个tile为粒度。

## A：metadata复用与UB生命周期

`CalSendCntGroup()` 只读取一次原始int32 TopK。`PrepareMetadata()` 在histogram复用TopK UB之前：

1. 用向量索引生成 `(tokenInTile * metaStride + k) * 4`，Scatter原始TopK到原metadata槽。
2. weights/scales按原32B对齐子区域直接搬入；无weights及非FP8分支沿用模板裁剪。
3. 写原格式的src rank/token字段，整tile写到原注册 `payloadStashWinAddr_`。
4. 等MTE3完成才让直方图/去重复用scratch；原始TopK仍保留在输入UB供后续Cast，不再二次读取GM TopK。

metadata使用既有dstExpBuf前16KiB；int16 index/token和int32 scatter offsets使用tempBuf不重叠区域；
除数放在topkIdsBuf的32KiB偏移，原始TopK最多16KiB。没有增加计算侧UB分配。
FP8 scales、TopK、weights、rank/token的有效区域互不重叠；padding不作为有效输出。

本地接收metadata仍依赖slot前缀，由准备核从已完成的stash按本地索引搬入原本地slot窗口。
这部分新增一次stash读取，换取所有AIV分担独立metadata并避免再读取/构建TopK。

## B：计数与共享前缀

计数行从旧prepare交集变为tile独占写，删除rank/expert全局原子累计。
每个AIV累计自己的expert histogram并写一行partial，空计数核也写零行。
全核屏障后，不重叠的8-expert块由各核跨partial行归约到原最终expert count。

rank前缀采用8-peer列块、每批最多128 tile的Hillis–Steele inclusive scan；每批末行作为下批carry。
prefix[0]为零，prefix[t+1]是到tile t为止的累计值；最后一行同时发布全局per-rank count。
准备核/发送核直接使用prefix，不再各自读取前面prepare行归约，也不再二分src-token列表。
新增一次全核屏障确保两类最终count在Communication读取前完成；这是实际固定开销，不声称没有代价。

workspace依次为：

| 区域 | 大小/布局 |
| --- | --- |
| tile rank count | tileCount × align512(P×4) |
| 最终rank/expert count | align512(P×4) + align512(E×4)，保留原通信数据格式 |
| tile prefix | (tileCount+1) × align512(P×4) |
| expert partial | AIV × align512(E×4) |
| tile ready | align512(tileCount×32)，本地通知、每轮初始化 |
| dedup rank | align512(BS×K×2) |
| src-token列表 | P × align512(BS×4)，仍按peer内token升序 |

host tiling和device tiling结构同步增加显式offset；workspace扩容不改变注册窗口，也不改变公开ACLNN/Torch参数。
必须重新构建/安装匹配的host tiling和kernel算子包，不能混用旧tiling结构与新kernel；本轮没有wheel接口改动。

## C：tile就绪与发送

准备核读prefix[t]后生成该tile每peer的src-token列表与原slot index，再搬本地metadata。
全部相关GM写完成后才发布该tile的32B ready；零命中tile同样发布。
ready发布后列表和metadata不再由生产者覆写，不使用循环队列或多轮复用同一tile槽。

“描述符”采用静态tile身份和prefix[t]/prefix[t+1]两行，不额外复制peer/start/count表：
tile ID确定BS范围/owner，peer列确定目标rank，两个prefix确定源索引和目标slot的起点及数量。
sender只在ready后读取这两行；索引预取也严格限制在该范围，没有保留对未就绪列表的二分查找。
相邻peer的索引仍使用原双缓冲，切peer后立即GetHandleRef并构造对应WQE。

每tile末强制提交尾批，避免等待下一个tile时把已经可发的数据留在UB。
所有本sender的tile提交后，才对每peer发送一次原有最终完成flag，包含无tile sender和零payload peer。
flag不按tile增加；epilogue仍等待原 `[srcRank][senderJetty]` 矩阵，不改远端完成协议。

## 性能收益与限制

源码层面减少了重复TopK读取、metadata串行准备负载、原子累计、重复前缀计算和scalar GM二分查找；
tile允许准备和WQE提交更早重叠。但128 tiles会增加小块固定开销和部分尾SGE，新增prefix归并屏障、
expert partial流量、本地metadata stash读取也有成本。不能从上述改动直接推出50us或固定加速比。

典型8P/BS4096/K6时，每sender128 tokens；即使每条route都独立使用2 WQEBB，
加最终flag和count交换的保守单轮上界仍小于SQ4096。专项CPU模型检查了这个算术界，
它不证明多轮异步源生命周期，不证明更大BS/K/P下的队列安全，也不替代真实Hcomm版本验证。
本地BatchCommit没有累计SQ credit控制，cqe=0不解决SQ覆盖问题；tile尾SGE可能增加WQEBB，需上机观察。
incoming完成也不证明本rank outgoing源可复用，X、count workspace、stash继续受既有异步生命周期约束。

## 静态验证与独立审查

回归入口：

- `python -B mc2/moe_ep_dispatch/tests/ut/op_kernel/test_dispatch_count_histograms.py -v`
- `python -B mc2/moe_ep_dispatch/tests/ut/op_kernel/test_dispatch_tile_pipeline.py -v`
- `python -B mc2/common/tests/test_moe_ep_async_completion.py -v`
- `mc2/moe_ep_dispatch/tests/ut/op_kernel/test_dispatch_schedule.cpp` 是host-only C++ tile选择器测试，需本地C++编译器。

CPU模型覆盖tile所有权、cached/uncached计数、跨128行prefix carry、乱序producer就绪、slot/索引完整性、
metadata scatter与FP8/weights布局、workspace不重叠、典型单轮WQEBB上界及async源码契约。
不模拟Reg实际代码生成、DMA/atomic可见性、Hcomm硬件进度或性能；未获远端执行授权，不执行远端/NPU操作。

本地结果（包含下面的编译窄化与Pack修复回归）：histogram 11项、tile pipeline 9项、
async completion 12项，共32项Python测试通过；
`git diff --check`通过。设备Dispatch/Combine源码中未发现assert、printf、CheckSharedJettyResult、
`(void)hcomm`或Drain。Dispatch函数长度扫描均不超过50个非空、非注释行。
当前环境PATH中没有CANN编译器及可用C++编译器，未执行CANN构建和host-only C++测试。

四项独立`gpt-5.6-sol`审查覆盖逻辑、Clean Code、功能完整性、架构；未留下确定性的静态阻塞项。
主代理处理的具体意见：

- 将source-token写入抽成独立小函数，保持UB与MTE3同步位置，满足函数长度限制。
- 删除仅测试使用的生产TileRange辅助函数，C++测试明确区分实际host选择器与device范围模型，
  并增加源码契约检查，防止两处device范围公式漂移。
- 补充有效cached slot从计数到prefix、乱序prepare和sender消费的完整CPU模型；
  cache依据uncached语义的独立oracle构造，并非实际device运行产生，
  不把人工伪造的重复正slot当作有效API输入；补充测试已由功能审查代理复核。
- 保留FP8/weights真实DMA、同步可见性、SQ累计在途和异步源生命周期的上机验证限制，
  没有把模型通过解释为设备正确性或加速证明。

审查中的搬运单位争议已由主代理查官方具体重载解决：这里调用DataCopyPad，blockLen和GM stride为字节，
UB stride为32B；不能套用DataCopy参数单位。见[官方DataCopyPad表5](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0265.html)。

## 2026-09-23 直方图 Pack 覆盖修复

用户NPU日志暴露了此前静态审查和CPU模型未覆盖的寄存器语义错误。本次附件是P8、BS4096、K6、
E384（不是此前性能例的E256）；rank7的每src payload count为1024，期望3072；
48个本地expert中只有16项计数正确，输出metadata仅8192行，期望24576行。

`RouteHistogramsVf` 原先对同一RegTensor和同一MaskReg依次Pack LOWEST/HIGHEST。
官方 [数据Pack图](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/data_compression/Pack.html)
和 [掩码Pack图](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/MaskReg_compute/Pack.html)
明确标明每次Pack会将目标的另一半清零，而不是保留前一次结果。
主代理实际查看了这两张图，不能仅依据文字“写入高/低半区”推断原位合并语义。

当前32-token tile有192条route，分成128和64两组。第二次Pack覆盖第一次后，只剩64条有效输入，
即64/192=1/3。按日志头尾展示的均匀route规律构造CPU复现，可同时得到：

- 每src payload 1024而非3072，累计offset为0/1024/.../8192。
- rank7本地expert仅16..23及40..47为512，其余为0；附件展示的头尾和通过比例与此吻合。
- PrepareTile仍处理完整的24次peer命中，但下一tile的prefix仅前进8，
  相邻tile会有16个source-token槽位重叠，解释为什么不仅输出变短，还会有索引重复/缺失。

修复仅在该VF：低/高Pack使用独立RegTensor和MaskReg，再用两个
[Or重载](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/logical_compute/Or.html)
配合全uint8 mask显式合并。每256条输入的BIN0/BIN1调用次数不变；
不修改tile/workspace/通信窗口/Jetty/完成标志/epilogue，不增加printf、assert、Drain或PIPE_ALL。

原CPU模型直接向同一列表的两个半区赋值，错误地隐含了“Pack保留另一半”，所以旧测试没有发现本次缺陷。
现模型将每次Pack建模为独立的全宽零填充结果，再显式Or；另有旧覆盖路径的负例、
1/64/127/128/129/192/255/256/257/384/4096尾长与多repeat、分页/无效ID和源码契约测试。
新增源码契约在修复前失败、修复后通过；这仍不是执行CANN指令或NPU回归。

验收需在原失败用例上确认每src count=3072、48个本地expert均512、总metadata=24576、
recv_x_idx一一映射，并继续覆盖standalone/continue、cached/uncached与不同K/P/E。
本轮未执行远端编译或NPU；不能把CPU复现/修复后的32项测试通过表述为设备验收通过。

修复后按本次授权使用`gpt-5.6-sol`完成逻辑、架构、功能覆盖与Clean Code独立复审。
逻辑复审也实际查看了两张官方Pack图，确认必须同时合并数据和mask；
架构复审核对了错误prefix导致source-list切片重叠的链路，确认无需改golden或增加Drain。
Clean Code发现另一份文档仍写初版8项测试，已同步为11项；未留下确定性的静态阻塞项。
