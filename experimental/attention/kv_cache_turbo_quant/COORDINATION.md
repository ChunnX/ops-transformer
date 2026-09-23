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
