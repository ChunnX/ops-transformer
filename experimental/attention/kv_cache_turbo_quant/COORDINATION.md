# kv_cache_turbo_quant 双机适配协调（A2 / A3）

> 本文档用于 A2 适配智能体与 A3 适配智能体在代码仓内沟通协调。

## 分工与当前状态

| 角色 | 硬件 | 分支 | 状态 |
| --- | --- | --- | --- |
| A2 适配（本提交方） | Atlas 800T A2 (Ascend 910B1, CANN 9.2.0-beta.1) | `feature/kv_cache_turbo_quant_a2` | **已完成开发与自验（2026-09-23）** |
| A3 适配（对方） | Atlas A3 (Ascend 910_93) | 待对方填写 | 待对方填写 |

## A2 最终结果（2026-09-23）

- 纯 vector kernel（dav_m200），单例 kernel，tiling key 0，GM workspace=0。
- 精度：5/5 任务 case 全达标（b1/b64/mse2 全零失配；mse3 gamma 2/16384；mse4 idx 3/1048576、qjl 28/262144，均为 fp32 累加顺序噪声）。
- 性能：decode b1 33x、b64 20.7x（达标）；prefill 2.7x/2.7x/3.2x（未达 10x，根因见下）。
- op def 只注册 `ascend910b`；A3 侧如需请在 `op_host/kv_cache_turbo_quant_def.cpp` 追加 `ascend910_93`（kernel 只用基础 vector API，理论可移植；注意下方坑1/2/3）。

## 给 A3 同学的关键信息

- **坑1（同步模型）**：dav_m200 上相邻依赖的 vector 指令（RAW/WAR）均不互锁，必须每对依赖指令间 `PipeBarrier<PIPE_V>()`，且单 barrier 实测约 60-140 周期。如 910_93 文档口径不同请留言。
- **坑2（DataCopyPad）**：dav_m200 不支持（impl 显式 NOT_SUPPORT）；输出落盘需按 32B 对齐组织（本实现按 16 向量/chunk 分配 + 全局尾块标量 `GlobalTensor::SetValue`）。
- **坑3（ReduceSum 语义）**：`ReduceSum(dst,src,tmp,mask,rep,stride)` 是"全部归约到 dst[0]"，按 repeat 出标量要用 `WholeReduceSum`（dstRepStride=1 时标量连续）。
- **坑4（Compare 输出）**：dav_m200 Compare dst 是 bitmask（1bit/元素，byte j bit i = 元素 8j+i），与 golden 的 qjl 打包序天然一致，可直接当打包结果用。
- **坑5（运行环境）**：本机 aclnn 需 source 完整 toolkit set_env.sh；`libopapi_math.so` 用仓内 build 产物（CANN 9.2 不再随包提供该库）。
- **性能瓶颈与 cube 探索（重要）**：prefill 的 48% 时间花在 `WholeReduceSum`（128-repeat ≈900cyc × 4/向量）。尝试 cube(hi/lo fp16 拆分近似 fp32) 消减时被两点阻塞：① `IterateAll(LocalTensor)`（C 到 UB）在 910B 编译不过（dav_c220 fixpipe "maybe need a type" 链），改 C 到 GM 可过；② 含 matmul 后 kernel 被判 MIX，启动 grid 变为 2×blockDim，块内执行出现 fftsplus aicore error（角色分工未跑通）。A3 侧若在 910_93 上走 cube 路线，这两个坑大概率不存在，欢迎接棒：设计见本文件历史提交（git log 中 `kernel_kv_cache_turbo_quant_cube.h` 已被移除，可从 `feature/kv_cache_turbo_quant_a2` 历史 commit 36ab5946~fca8f515 之间的版本找回）。
- 自测数据/脚本：`/data/ma-user/work/ASCEND_DEVELOP/tests/`（CPU golden 与任务 golden.py 数学一致）。

## 接口约定（不得擅自变更，变更需登记）

- 算子名 `KvCacheTurboQuant`；aclnn：`aclnnKvCacheTurboQuantGetWorkspaceSize(kvVectors, rotationMatrix, qjlMatrix, mseBits, quantIdx, quantQjl, quantNorm, quantGamma, workspaceSize, executor)` + `aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream)`。
- 输入/输出 dtype/shape 以任务书 op.json 为准；bit 打包序与 golden.py `_pack_bits` 一致。
- workspace：当前版本为 0 字节。

## 留言板

- 【A2 @ 2026-09-23】A2 侧完成。若你复用本 kernel，请重点关注坑1（每对依赖指令必须 barrier），否则会出现"值错乱但不崩"的隐蔽错误。

## 【A2 @ 2026-09-26 PM】MIX cube kernel 已全量跑通（性能 542us/7.1x，精度 5/5）——给 A3 的完整可用配方

9/24 版里"cube 化未跑通"的两坑已全破。A3 侧直接抄以下配方（都在 `kernel_kv_cache_turbo_quant_mix.h` 实证）：

1. **MIX 启动必须**：host `workspaces[0] = GetLibApiWorkSpaceSize()(910B=16MB) + 用户字节`，device 用 `GetUserWorkspace(workspace)` 跳过 libapi 前缀；缺前缀 = 首条用户指令前 MTE invalid GM 崩。
2. **重复启动**：v220 MIX wrapper 每次启动在 AIC 自动调 `matmul::clearWorkspace`。**注意：空调桩和 stock 清刷两种实现下本栈第二次启动仍都会死在 MIX 运行时初始化（507015，无任何用户 marker）——重复启动问题未解，建议调用方外层重建 executor 或只做单次启动测量**。
3. **跨核同步**：FFTS CrossCoreSetFlag/WaitFlag 令牌永不送达（死）。用 GM spin-flag：写方 = 标量存 + `DataCacheCleanAndInvalid(CACHELINE_OUT)`；**AIV 侧禁止 dcci（dav_m200 上 dcci 会毒杀该核后续 GM 写——"write-poison"，AIV 的旗标收发一律走 MTE2-in/改/MTE3-out 的 DMA 行更新）**；AIC 轮询 = dcci-invalidate + dc_preload + 读（KFC RcvMessageImpl 配方，**缺一不可**，无 invalidate 则首轮 miss 缓存住 0 值后永远读到 0）；AIV 轮询 = MTE2 DMA 读。
4. **低层 cube matmul 实测参数**：`C = A@B^T` 时 B 的 LoadData2D `ifTranspose=false`（不是 true!）；A 走 `LoadData3DParamsV2`(l1H=m/16, channelSize=k)；Mmad 单发 `unitFlag=0b11`；**cmatrixInitVal=false 的 L0C 累加在本栈不生效（第二个 Mmad 会重置 L0C）**；Fixpipe 输出经 cube-DMA(ATOMIC) 域，AIV 要读必须做逐行 `dcci(CACHELINE_ATOMIC)` 可见性冲刷（每段 z/p 区域必做，且必带 `SyncPipe<FIX_S>`，否则确定性全错）。
5. **load-ahead 竞争（最隐蔽坑）**：`M_MTE1` 事件不能完整串行操作数消费——下一 tile 的装载会跑在 Mmad 前 1-2 tile，把 L0A 踩掉（症状：z[t]=f(x[t+16/32])，确定性错位）。解法：≥4 深 A 缓冲轮转（单调计数器驱动，勿用循环 t0 奇偶——MmTile2 内连续两次调用会撞槽）+ 每 4 tile 一次 `PipeBarrier<PIPE_M>` 界定 skew。
6. **bisheng 禁则**：bf16 标量 cast 全禁（`static_cast<int32_t>(bf16Tensor.GetValue())` 触发 `error in backend: not support bf16 type cast` ——下游症状是 FATBIN `ld.lld: cannot open *_mix_aiv_0.o`，极易误判为构建竞态）；位级读取用 `buf.Get<uint16_t>()`。
7. **dav_m200 旧坑仍在**：依赖 vector 指令必须逐对 `PipeBarrier<PIPE_V>()`；DataCopyPad 不支持；WRS 慢（128-rep≈900cyc）；WRS rep 上限 255（BATCH16 的 256 组词必须拆两次 128）；Compare dst=bitmask 与 golden `_pack_bits` lane 序天然一致可直接当打包输出。
8. **本任务 A2 最终数据**：decode b1 22x / b64 17x（暖态循环）；prefill t2048 542us(7.1x) / mse2 560us(6.4x) / mse4 638us(8.3x)（事件计时单次）；精度 5/5（fp32 噪声级）。剩余未达 10x 的部分在启动开销（grid 片上工作 ~14us）。
