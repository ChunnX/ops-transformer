
## Phase 3 续 [撞墙·需 SE/版本答案] 权威配方已应用，仍挂 CCU kernel ReadNb
应用了 4-agent workflow 综合出的 80% 置信配方：
- send_buff → HcclGetHcclBuffer 通信域内存(commBuf, 两卡同 VA 0x12004ca00000 对称、400MB、框架已 UB 注册)
- 删 HcclCommMemReg + channelDesc.memHandles(坐实 memHandles 是 AIV-only、CCU 不支持)
- CcuReduceTaskArg length 改字节(malloc_kSize)非 count
结果：两卡均建链→注册→launch→GeneArgs 发出有效参数(commBuf+有效token+1024字节)→仍挂 aclrtSynchronizeStream(CCU kernel ReadNb 跨卡执行不完成)。

### 命中的 20% 风险(配方自标)
HcommReadNbiOnThread.md 是 CPU-TS 的 ReadNb 路径；CcuReduceMesh1D 走 CCU kernel ReadNb——两者内存注册要求不同源。HcclGetHcclBuffer 满足 CPU-TS 路径，但对 CCU kernel ReadNb 实测仍不够。配方兜底 HcclCommSymWinRegister 已坐实 950 门控(死路)。

### 唯一缺口(收敛的关键事实)
0609 CANN 上 CcuReduceMesh1D 的 kernel ReadNb 要让对端 commBuf 可解析，到底还差什么(channel 怎么把 HcclBuffer 数据面建起来 / CCU kernel ReadNb 对内存的确切要求)——此事实在现有 0609 文档/源码里查不到(0609 无现成 CCU kernel 数据-mover 范例，moe setup 用更新 API ccuUbAttr 编不过)。需 SE/版本 owner 给确切答案，或换带匹配 CCU 数据面 API 的 CANN 版本。

### 当前最佳状态文件
/home/developer/workspace/hcomm-test/src/ccu_kernel_test/hcomm_ccu_reduce_test.cc (commBuf+byte-len 版, 编译通过, 跑到 launch+GeneArgs, 挂 ReadNb sync)
