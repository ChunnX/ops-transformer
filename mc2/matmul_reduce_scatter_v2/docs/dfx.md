# MatmulReduceScatterV2 DFX（Ascend950 / arch35）

## 目的与适用范围

DFX 在异常时保留算子静态配置、各核执行位置及通信进度，配合 CCU 的 XN/CKE 信息，
区分通信故障与 Matmul/ReduceSum/核间同步问题。它不改变数学功能，也不是正常执行时的打印开关。

覆盖 FP16/BF16、MXFP8/per-tensor/per-block 的直接 ReduceScatter 与
AlltoAll + Vector ReduceSum 路径。MXFP8 分别由 quant_bmm_reduce_scatter_fp8_hif8.h
和 quant_bmm_a2a_vec_reduce_fp8_hif8.h 执行。

## 与 AllToAllMatmul 的对照

| 项目 | AllToAllMatmul | MMRS |
| --- | --- | --- |
| 异常注册 | IMPL_OP + ExceptionDumpParseFunc | 相同 |
| 隐式参数 | args[0] 是 HCCL context | 相同 |
| workspace / tiling 参数 | args[11] / args[12] | args[9] / args[10] |
| 执行顺序 | 通信、重排、Matmul；MX 模式另有 scale 通信 | Matmul、ReduceScatter，或 Matmul、AlltoAll、ReduceSum |
| 通信计数核 | AIV0 | A2A 为 AIV0；直接 RS 为 AIC0 |
| 运行记录 | 每核 512B，共 64KB，写后 DCCI | 相同；位置枚举按 MMRS 阶段解释 |

不要照搬 A2AMM 的参数编号、scale 轮次或位置枚举。
MMRS 在 Finalize 返回后更新通信阶段，通过执行位置记录进入 Finalize。

## Workspace 地址契约

段表的 offset/totalSize 包含框架 LIB_API 前缀；kernel 的 workspaceGM 和异常 args[9]
都是已经跳过 LIB_API 的用户地址。因此：

    userOffset = segment.offset - LIB_API.size
    srcAddr    = workspaceGM + userOffset

kernel 运行记录写入和 host dump 必须使用同一公式。不能再加一次 LIB_API，
也不能从用户基址导出框架的 LIB_API 段。

- 非量化：LIB_API、MM_RESULT、可选 RECV_BUF、可选对齐填充、RUNTIME_INFO。
- 量化：另记录 Matmul 用户临时区 MM_WORKSPACE（type=13）。
- 量化总大小保留原 helper workspace 加 cToFloatLen（直接 RS），或加 3*cToFloatLen（A2A），
  再追加对齐填充和 64KB。helper workspace 已包含 LIB_API，不能重复计算。
- 两种 A2A kernel 的 recvBuf 均使用 sendBuf + cfg.cToFloatLen，与段表边界相同。
  不能使用未对齐的 M*N*sizeof(C) 替代 cToFloatLen。
- RUNTIME_INFO 相对用户基址按 512B 对齐，128 个 slot，每个 slot 512B。
- LIB_API 和仅作对齐的 RESERVED 段不落盘。

## 本次导出链路修正

1. MMRS 回调明确包含 common/utils 的 host dump 头文件；入口打印 SoC 和两种 tiling 的实际 sizeof。
2. 单独读取 quantMode/WorkspaceLayoutInfo，按实际 sizeof 读取对应 tiling，不再固定读取 2048B。
3. 完整 tiling 拷贝或落盘失败，不跳过已经独立读取的 workspace 段表。
4. workspace 第一轮仅导出 RUNTIME_INFO，第二轮才导出矩阵/临时缓冲区；某个段失败继续其他段。
5. 导出前创建缺少的 dump 目录，包括 deviceId 子目录。
6. Device 数据用最多 1MB 的 host 缓冲分块 D2H/落盘，长度保持 64 位，
   并处理短写、EINTR、零字节写入和 close 错误。
7. 动态符号查找失败不再永久缓存 null，同时尝试已加载的全局符号。
8. 算子证据导出后再读取 CCU XN/CKE，避免 CCU 访问阻挡 workspace 导出。

失败日志包含文件名、deviceId、GM 地址、段偏移、拷贝长度和返回值。
首次 D2H 失败不创建空的“成功文件”；中间块失败可能留下部分文件，会明确报错，
不会计入 dumpedSegments。该计数仅统计完整成功落盘的段。

## 运行记录解释

- execTurn：当前 1-based tile，0 表示尚未进入 tile，不是完成计数。
  尾块从 tileCnt+1 开始，Finalize 不额外加一轮。
- execPosition：0=通信，1=Matmul（包括等待依赖的 Cube、空闲 AIC 的核间同步、Matmul Reset/End），
  2=Vector ReduceSum，3=Finalize。
- commPhase：255=尚未完成初始化；0=InitV2/SetCcTilingV2 返回；
  2=带 commit 的集合通信 API 返回；3=Wait 返回；4=Finalize 返回。
  AlltoAll<true>/ReduceScatter<true> 合并 Prepare/Commit，不虚构单独可观测的 PREPARED 状态。
- 只有通信控制核计数，其他核的零计数不表示没有通信。
- A2A 会在归约 tile i 前提交 tile i+1，所以 execTurn 随阶段切换可以回退。
- commit > wait 仅说明仍有未完成等待的通信，不能单独据此判定卡死。
- API 返回打点不代替通信内部错误检查，需结合 XN/CKE、异常日志和 PC 分析。

## 部署同步清单

必须用同一版本构建并安装 host/kernel，不能只拷贝 MMRS cpp。
WS_SEG_MM_WORKSPACE 不是 SDK 枚举，定义来自项目 common/op_kernel/mc2_tiling_struct.h。
至少核对以下生产代码，部分文件虽然此次未改动，也是 ABI/初始化依赖：

    mc2/common/op_kernel/mc2_tiling_struct.h
    mc2/common/op_kernel/mc2_runtime_info.h
    mc2/common/utils/mc2_exception_dump.h
    mc2/matmul_reduce_scatter_v2/op_host/op_tiling/matmul_reduce_scatter_v2_tiling_common.cpp
    mc2/matmul_reduce_scatter_v2/op_host/op_tiling/matmul_reduce_scatter_tiling_base.cpp
    mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp
    mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/matmul_reduce_scatter_v2_tiling.cpp
    mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_v2_c_tiling.h
    mc2/matmul_reduce_scatter_v2/op_kernel/common_def.h
    mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h
    mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_a2a_vec_reduce_fp16_bf16.h
    mc2/matmul_reduce_scatter_v2/op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h
    mc2/matmul_reduce_scatter_v2/op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h

解析脚本须与上述 ABI 一致，不需要新增测试代码。公共改动保留 A2AMM 原调用签名及参数默认值。

## 没有 workspace 文件时如何定位

先沿用同机 A2AMM 已验证的异常 dump 配置。正常成功调用不会进入异常回调。
需复核配置时，参考本机 CANN 版本对应的
[ASCEND_DUMP_SCENE 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/maintenref/envvar/envref_07_0164.html)，
不要混用不同版本的环境要求。生产代码未启用任何非法地址故障注入。

| 日志/现象 | 下一步 |
| --- | --- |
| 没有 DFX callback begin | 检查是否发生受支持的异常、是否开启异常 dump、是否加载本次 host 库和回调 |
| callback skipped | 查看 SoC 和异常参数，本适配针对 Ascend950 |
| No exception dump path | 检查 dump 配置、动态库和路径服务 |
| workspaceGM=0 或 args 越界 | 核对 args_info、参数 ABI、host/kernel 版本，不猜测地址 |
| Cannot read quantMode / workspace layout | 查看 D2H 返回值、tiling 地址、offsetof/sizeof |
| wsLayout invalid / No RUNTIME_INFO segment | 检查 GetWorkspaceSize/PostTiling 是否使用本次代码 |
| Cannot create / Failed to open | 检查完整路径、权限及磁盘状态 |
| D2H dump failed | 核对 srcAddr=workspaceGM+userOffset，以及故障后设备内存是否可读 |
| type12 存在、其他段失败 | 先分析运行记录，再查看对应段的拷贝/落盘错误 |
| Workspace dump done | dumpedSegments 是成功数；type12 文件应为 65536B |

文件名为 workspace_seg{idx}_type{type}_MatmulReduceScatterV2...，
不要仅按 .bin 后缀查找，设备回调生成的文件名可能没有扩展名。
实际目录为日志中的 dump 路径/deviceId。

## 已有解析脚本与设备验收

继续使用已有 parse_matmul_reduce_scatter_v2_tiling.py 检查实际导出文件，不新增测试。
脚本解析已有文件，不参与异常回调或文件生成。
ops-transformer_dfx 中的旧脚本可识别 type12 运行记录和无扩展名的段文件，
但其 args 名称表未计入隐式 HCCL context，会把 workspace/tiling 错标为 args[8]/args[9]；
也未定义 MM_WORKSPACE(type13) 和 COMM_UNINITIALIZED(255) 的名称。
这不导致文件缺失。核对地址和新字段时使用 ops-transformer 内已同步 ABI 的版本；
不改动 ops-transformer_dfx。

在 ops-transformer 源码根目录执行：

    python mc2/tools/dump_analysis/parse_matmul_reduce_scatter_v2_tiling.py args_info_file --args
    python mc2/tools/dump_analysis/parse_matmul_reduce_scatter_v2_tiling.py tiling_data_file --ws-layout
    python mc2/tools/dump_analysis/parse_matmul_reduce_scatter_v2_tiling.py workspace_segN_type12_file --workspace --aic-core-num N

每次传入一个实际文件。aic-core-num 取对应 tiling 的 param.aicCoreNum。

最终验收仍需真实 CANN 编译及设备测试：覆盖各量化模式、FP16/BF16、主尾块、
直接 RS/A2A、超时/AIC/AIV 异常；核对 magic、地址、控制核计数，测量规范要求的
不超过 2% 的性能回退。离线解析和源码检查无法验证设备缓存可见性、RTS 回调触发及故障后的 D2H 能力。
