# AGENT.md - URMA P2P Benchmark 项目交接文档

> 本文件供后续Agent快速理解项目全貌，无需逐文件阅读即可丝滑接手。

---

## 项目定位

**URMA P2P Benchmark** — 昇腾NPU平台上URMA协议Point-to-Point通信基准测试工具。测量NPU间URMA SQE级（HcclCombinOpParam）单链路通信的Overhead/Latency/Bandwidth三项底层指标。与已有bandwidth_test（AlltoAll多对多聚合带宽）互补，聚焦一对一P2P链路性能基线。

---

## 代码架构（四层结构）

```
urma_p2p_benchmark/
│
├── urma_p2p_benchmark.py          # [Python入口层] 参数解析/HCCL初始化/多进程调度/Profiling采集/报告生成
│   三种测试: overhead(mode=0)/latency(mode=1)/bandwidth(mode=2)
│   两种执行: 单算子(graph_type=0)/AclGraph(graph_type=3)
│   性能数据来源: torch_npu.profiler → op_summary CSV → Task Duration(us)
│
├── op_api/                        # [aclnn API层] Host侧算子调度入口
│   aclnn_urma_p2p_benchmark.h     # 两段式接口声明: GetWorkspaceSize + Execute
│   aclnn_urma_p2p_benchmark.cpp   # 实现: 参数校验 → 调用Inner接口 → 设置HcclServerType=MTE
│   条件编译: #if BUILD_OPEN_PROJECT && HCOMM_VERSION_NUM >= 90000000
│
├── op_graph/                      # [图模式层] GE算子注册与任务生成
│   urma_p2p_benchmark_proto.h     # REG_OP定义: 2 INPUT + 2 OUTPUT + 7 ATTR
│   urma_p2p_benchmark_gen_task.cpp # CalcParam(KFC) + GenerateTask(MC2 V2回调)
│   fallback_urma_p2p_benchmark.cpp # Fallback路径: CPU侧调用aclnn接口
│
├── op_host/                       # [Host Tiling层] 参数校验/通信配置/Workspace计算
│   urma_p2p_benchmark_def.cpp     # OpDef注册: MC2().HcclGroup({"group"})声明 → 框架自动创建HCCL上下文
│   op_tiling/urma_p2p_benchmark_tiling.cpp # Tiling主流程:
│   │   1. 属性校验(group/worldSize/maxBs/mode/dstRankId)
│   │   2. 输入输出Shape校验(x:[bs,h], dst_rank_id:[bs], y:[max_bs*ws,h], receive_cnt:[ws])
│   │   3. 窗口大小检查(CheckWinSize → mc2tiling::GetEpWinSize)
│   │   4. Workspace计算: 16MB系统预留 + worldSize*hcclBufferSize + 2GB
│   │   5. SQE通信配置(SetHcommCfg): Mc2CcTilingConfig + protocal=1(URMA) + AIV引擎
│   │   6. BlockDim计算: ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum)
│   │   7. TilingKey: URMA+A5固定组合
│   config/ascend950/              # 二进制kernel编译配置
│
└── op_kernel/                     # [设备Kernel层] 核心实现
    ├── urma_p2p_benchmark.cpp     # Kernel入口: REGISTER_TILING + GET_TILING_DATA + 模板实例化
    ├── urma_p2p_benchmark.h       # Kernel类: UrmaP2pBenchmark<XType,YType>
    │   ├─ Init(): 两阶段UB分配
    │   │   Phase1: urmaComm_.Init → rank信息 → 对齐参数 → 窗口偏移 → 状态缓冲区
    │   │   双缓冲: InitWinState → dataState_选择窗口半区
    │   ├─ Process(): 根据mode分发
    │   │   mode=0: OverheadTest → 仅DataCopy(x→y)
    │   │   mode=1: LatencyTest → 逐token SQE P2pSend + P2pSendDrain
    │   │   mode=2: BandwidthTest → 批量SQE P2pSend + 统一P2pSendDrain
    │   │   然后: SetStatus → Reset+InitCommBuffer → WaitDispatch → LocalWindowCopy
    │   ├─ SetStatus(): SQE级P2pSend发送状态(token计数+flag)到各rank
    │   ├─ WaitDispatch(): 轮询状态区flag, ReduceSum求和等待所有rank完成
    │   └─ LocalWindowCopy(): 从HCCL窗口读取接收数据到输出y, 设置receiveCnt
    │
    ├── p2p_urma_comm.h            # SQE级URMA P2P通信封装类 P2pUrmaSqeComm
    │   ├─ Init(): GetHcclContext<HCCL_GROUP_ID_0> → 获取HcclCombinOpParam
    │   ├─ InitCommBuffer(): 分配SQ/CQ/SQE/CQE/DoorBell UB缓冲区
    │   ├─ InitRankComm(): 加载SQ/CQ信息 → 获取PICI → InvalidateCQE → PollCQ → 生成SQE模板
    │   ├─ P2pSend(): WriteWithNotify SQE提交(不敲DoorBell不轮询CQ)
    │   ├─ P2pSendDrain(): 敲DoorBell + PollCQ + UpdatePICI
    │   ├─ GetRemoteWinAddr/GetLocalWinAddr: 窗口地址计算
    │   └─ GetRemoteWinStateAddr/GetLocalWinStateAddr: 状态区地址计算
    │
    ├── urma_p2p_benchmark_tiling.h      # Tiling数据结构 UrmaP2pBenchmarkTilingData
    └── urma_p2p_benchmark_tiling_key.h # Tiling Key: URMA+A5
```

---

## 关键数据流

```
Python层:
  构造输入(x:[bs,h], dst_rank_id:[bs]) → npu_custom_op(attrs) → profiling采集 → 解析op_summary

aclnn API层:
  CheckParams → aclnnInnerGetWorkspaceSize → aclnnInnerExecute

Host Tiling层:
  属性/Shape校验 → CheckWinSize → SetWorkSpace → CalTilingKey(URMA+A5) → SetHcommCfg(protocal=1)
  → SetBlockDim → 填充TilingData → context->SetTilingKey

Kernel层:
  GetTilingData → Init(Phase1) → DataCopy dstRankIds → mode分发:
    OverheadTest: GM→UB→GM (本地DataCopy)
    LatencyTest:  GM→UB→localWin → SQE P2pSend → Drain per token
    BandwidthTest: GM→UB→localWin → SQE P2pSend → Drain统一
  → SetStatus(SQE发送token计数+flag) → Phase2 Reset+InitCommBuffer
  → WaitDispatch(ReduceSum轮询flag=1.0f*N) → LocalWindowCopy(win→UB→GM)
```

---

## 内存布局

| 区域 | 偏移 | 说明 |
|------|------|------|
| HCCL窗口数据区 | P2P_STATE_SIZE(350KB) + winDataSizeOffset | 双缓冲: dataState选择半区 |
| 本地状态区 | dataState * P2P_WIN_STATE_OFFSET(350KB) | dispatch状态区 |
| 远端状态区 | P2P_STATE_WIN_OFFSET(950KB) | WaitDispatch轮询此区 |
| 每rank窗口空间 | (max_bs/world_size) * hAlignWinSize | expertPerSizeOnWin |
| Workspace | 16MB系统预留 + ws*cclBufferSize + 2GB | Host侧计算 |

---

## 已修复Bug清单

以下Bug已全部修复（2026-06-11）：

### BUG-1: P2pSend函数签名不匹配 [已修复]
- **位置**: `op_kernel/p2p_urma_comm.h`
- **修复**: 增加4参数重载版本 `P2pSend(dstRankId, localAddr, remoteAddr, length)`，内部自动从远端窗口状态区计算notifyAddr，notifyData使用0x3F800000(1.0f)
- **新增**: `P2P_STATE_OFFSET`和`P2P_NOTIFY_DATA_FLAG`常量、`notifyDataState_`成员变量、`SetNotifyDataState()`方法
- **联动**: `urma_p2p_benchmark.h` Init()中新增 `urmaComm_.SetNotifyDataState(dataState_)`

### BUG-2: tiling.h结构体重复定义 [已修复]
- **位置**: `op_kernel/urma_p2p_benchmark_tiling.h`
- **修复**: 删除第一个不完整的struct定义(原21-23行)和非法的`#include`，仅保留第二个完整定义，`#include "kernel_tiling/kernel_tiling.h"`已在文件顶部第19行正确位置

### BUG-3: binary.json残留context输入 [已修复]
- **位置**: `op_host/config/ascend950/urma_p2p_benchmark_binary.json`
- **修复**: 删除context输入项，仅保留x(index=0)和dst_rank_id(index=1)，outputs和attrs结构完整

### BUG-4: OverheadTest死等风险 [已修复]
- **位置**: `op_kernel/urma_p2p_benchmark.h` Process() mode=0分支
- **修复**: overhead模式新增 `SetStatus()` 调用，确保所有rank都发送状态通知，WaitDispatch能正常完成

### BUG-5: aclnn头文件多余花括号 [已修复]
- **位置**: `op_api/aclnn_urma_p2p_benchmark.h#L57-59`
- **修复**: 删除多余的 `}`

---

## 编译命令速查

```bash
# 仅编译本算子（推荐）
bash build.sh --pkg --ops=urma_p2p_benchmark --soc=ascend950 -j16 -O3

# 仅编译Host侧
bash build.sh --ophost --opapi --opgraph -j16 -O3

# 仅编译Binary Kernel
bash build.sh --opkernel --soc=ascend950 --ops=urma_p2p_benchmark -j16

# 编译整个项目
bash build.sh --pkg --soc=ascend950 -j16 -O3
```

**关键**: SOC必须为ascend950（算子AICore仅配置了950平台）。

---

## 运行命令速查

```bash
# 最简验证（2卡overhead）
python urma_p2p_benchmark.py --test_type overhead --world_size 2 --rank_per_dev 1 --dst_rank 1

# 时延测试
python urma_p2p_benchmark.py --test_type latency --world_size 2 --rank_per_dev 1

# 带宽测试
python urma_p2p_benchmark.py --test_type bandwidth --world_size 2 --rank_per_dev 1

# 全量测试
python urma_p2p_benchmark.py --test_type all --world_size 2 --rank_per_dev 1

# 8卡
python urma_p2p_benchmark.py --test_type bandwidth --world_size 8 --rank_per_dev 8

# AclGraph模式
python urma_p2p_benchmark.py --test_type bandwidth --world_size 2 --rank_per_dev 1 --graph_type 3
```

---

## 依赖模块

算子CMake声明依赖以下模块（需代码完整才能编译）：
- `mc2/common` — MC2公共基础设施
- `mc2/3rd` — 第三方依赖
- `mc2/moe_distribute_dispatch_v2` — MOE分发dispatch v2 kernel及base头文件
- `mc2/moe_distribute_dispatch_v3` — MOE分发dispatch v3
- `mc2/moe_distribute_dispatch_setup` — dispatch setup（提供moe_distribute_base.h SQE级通信接口）

Kernel层直接引用的关键接口：
- `moe_distribute_base.h` — SQE级通信基础接口(GetURMASqInfoTensor, PutCommNotifySQE, SendJFSDoorBell等)
- `moe_distribute_v2_base.h` — InitWinState等窗口状态初始化
- `mc2_kernel_utils.h` — Kernel工具函数
- `HcclCombinOpParam` — HCCL组合操作参数上下文(GetHcclContext)

---

## 关键参数映射

| Python参数 | 算子属性 | TilingData字段 | Kernel变量 |
|------------|---------|----------------|-----------|
| --world_size | world_size | world_size | epWorldSize_ |
| --dst_rank | dst_rank_id | dst_rank_id | dstRankIdAttr_ |
| --dtype fp16/bf16 | x.dtype=float16/bfloat16 | - | input_dtype/XType |
| --test_type overhead/latency/bandwidth | mode=0/1/2 | mode | mode_ |
| data_size→bs,h映射 | max_bs=x.dim0 | max_bs | axisBS_ |
| cal_hccl_buffsize() | ccl_buffer_size | - | ccl_buffer_size(attrs) |
| - | aiv_num | aiv_num | aivNum_ |
| - | group | mc2InitTiling/mc2CcTiling | hcomm_info(Python) |
| 自动计算 | - | total_ub_size | totalUbSize_ |
| 自动计算 | - | total_win_size | totalWinSizeEp_ |

---

## 性能数据计算

P2P带宽（单链路，不除world_size-1）:
```
mean_time_s = mean_time_us / 1e6
mean_bandwidth_gbps = (data_size_bytes / mean_time_s) / 1e9
max_bandwidth_gbps = (data_size_bytes / min_time_s) / 1e9
```

统计指标: mean/min/max/std/p50/p90/p99 (numpy计算)

抖动分析: jitter = times - mean → abs_jitter的mean/max/std

---

## 技术细节速查

| 概念 | 值/说明 |
|------|---------|
| 窗口数据区偏移 | P2P_STATE_SIZE = 350KB |
| 本地状态区偏移 | dataState * P2P_WIN_STATE_OFFSET = dataState * 350KB |
| 远端状态区偏移 | P2P_STATE_WIN_OFFSET = 950KB |
| 窗口地址对齐 | WIN_ADDR_ALIGN = 512B |
| UB对齐 | UB_ALIGN = 32B |
| SQE大小 | WriteWithNotify SQE = 96B |
| CQ深度 | P2P_CQ_DEPTH_256 = 256 |
| 双缓冲阈值 | MAX_UB_SIZE_P2P = 240KB |
| TilingKey | URMA+A5 固定组合 |
| protocal字段 | mc2CcTiling.protocal = 1 (0:UB-MEM, 1:URMA) |
| 通信引擎 | AIV_ENGINE = 3 (SQE级URMA) |
| HcclServerType | MTE (aclnn Execute时设置) |
| HCCL通信域声明 | OpDef中 .MC2().HcclGroup({"group"}) → 框架自动创建 |

---

## 文件修改优先级

1. **必须先修**: BUG-1(P2pSend签名)、BUG-2(tiling.h重复定义)、BUG-3(binary.json context残留)
2. **建议修复**: BUG-4(OverheadTest死等)、BUG-5(aclnn多余花括号)
3. **文档同步**: docs/urma_p2p_benchmark.md参数默认值与代码不一致(详见技术参考文档7.4节)