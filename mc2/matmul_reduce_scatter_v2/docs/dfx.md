# MatmulReduceScatterV2 DFX（Ascend950 / arch35）

## 目的与适用范围

DFX 在异常时保留算子静态配置、各核执行位置及通信进度，配合 CCU 的 XN/CKE 信息，
区分通信故障与 Matmul/ReduceSum/核间同步问题。它不改变数学功能，也不是正常执行时的打印开关。

本次只覆盖 MXFP8（x1/x2 为 FLOAT8_E4M3FN 或 FLOAT8_E5M2，x1Scale/x2Scale 为
FLOAT8_E8M0）的直接 ReduceScatter 与 AlltoAll + Vector ReduceSum 路径。分别由
quant_bmm_reduce_scatter_fp8_hif8.h 和 quant_bmm_a2a_vec_reduce_fp8_hif8.h 执行。
MXFP4、HiF8、per-tensor、per-block 和非量化路径不启用本次新增的 workspace 分段、
RuntimeInfo 与定点打印。

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

- MXFP8 记录 MM_RESULT、A2A 可选的 RECV_BUF、Matmul 用户临时区 MM_WORKSPACE（type=13）
  以及 RUNTIME_INFO。
- MXFP8 总大小保留原 helper workspace 加 cToFloatLen（直接 RS），或加 3*cToFloatLen（A2A），
  再追加对齐填充和 64KB。helper workspace 已包含 LIB_API，不能重复计算。
- 两种 A2A kernel 的 recvBuf 均使用 sendBuf + cfg.cToFloatLen，与段表边界相同。
  不能使用未对齐的 M*N*sizeof(C) 替代 cToFloatLen。
- RUNTIME_INFO 相对用户基址按 512B 对齐，128 个 slot，每个 slot 512B。
- LIB_API 和仅作对齐的 RESERVED 段不落盘。

两条 MXFP8 kernel 地址链与 seg 一一对应：

| 路径 | kernel 用户 workspace 地址链 | WorkspaceLayout 中的有效数据段 |
| --- | --- | --- |
| MXFP8 直接 RS | `gmToFloat_=workspaceGM`；`workspaceGM_=gmToFloat_+cToFloatLen` | `MM_RESULT(type8)`；helper 临时区非零时增加 `MM_WORKSPACE(type13)`；`RUNTIME_INFO(type12)` |
| MXFP8 A2A | `sendBuf_=workspaceGM`；`recvBuf_=sendBuf_+cToFloatLen`；`workspaceGM_=recvBuf_+cToFloatLen` | `MM_RESULT(type8)`、`RECV_BUF(type9)`、`MM_WORKSPACE(type13)`、`RUNTIME_INFO(type12)` |

每条路径前面还有 `LIB_API(type0)` 描述，必要时在 RuntimeInfo 前插入 `RESERVED(type255)`。
seg 编号是段表下标，不是类型号；因为 type0 由框架管理且不落盘，workspace 文件通常从
`workspace_seg1_type8_` 开始，这是预期行为，不表示少导出一块用户数据。
x1/scale1 不在上述地址链中，不建立 workspace seg，也不由 workspace 异常导出链路保存。

原生 kernel 的 workspace 切分位于以下 `Init`：

- `op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h`：MXFP8 直接 RS。
- `op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h`：MXFP8 A2A。

host 侧仅在 `IsMxfp8DfxScene()` 为真时调用
`arch35/quant_bmm_reduce_scatter_tiling.cpp::BuildQuantWorkspaceLayout`；其他量化模式保持原生总量并写入空布局。

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

最终验收仍需真实 CANN 编译及设备测试：覆盖 MXFP8 E4M3/E5M2、BF16/FP16 输出、主尾块、
直接 RS/A2A、超时/AIC/AIV 异常；核对 magic、地址、控制核计数，测量规范要求的
不超过 2% 的性能回退。离线解析和源码检查无法验证设备缓存可见性、RTS 回调触发及故障后的 D2H 能力。

## 中间缓冲区分段

与 AllToAllMatmul 相同，MMRS 只给 kernel 数据流中实际位于 `workspaceGM` 的中间缓冲区建段。
x1、x1Scale 是只读输入参数，不是 workspace 中间结果，不进入 WorkspaceLayout，也不追加输入描述 ABI。

| 文件前缀 | 内容 | 地址来源 |
| --- | --- | --- |
| `workspace_seg1_type8_` | Matmul 中间结果 MM_RESULT | user workspace 起点 |
| `workspace_seg2_type9_` | A2A 接收数据 RECV_BUF，仅 A2A 路径 | user workspace + cToFloatLen |
| `workspace_segN_type13_` | MXFP8 Matmul workspace，仅实际长度非零时 | Matmul 输出/接收区之后 |
| `workspace_segN_type12_` | 64KB 运行记录 | 所有数据缓冲区及对齐填充之后 |

非量化和量化 tiling 大小分别保持 1408B、1656B；workspaceLayout 偏移分别为 1200、1448。
必须同时重新构建安装 host/kernel，不能只替换 kernel。

## Kernel 少量数据打印

不增加打印类或打印函数，只在两个 MXFP8 kernel 的已有同步点直接调用 `PRINTF`：

- 直接 RS：全部 ReduceScatter Wait 完成后，由 rank0/AIC0 打印 MM_RESULT。
- A2A：通信与 ReduceSum 流水完成后，由 rank0/AIV0 打印 MM_RESULT 和 RECV_BUF。
- 每个缓冲区固定打印第 0、1 行的第 0～3 列，一行对应一条日志。
- direct 共 2 条；A2A 共 4 条，不在 tile 循环内重复打印。
- 读取每一行前刷新对应 cache line，不新增核间同步；ONLY_CUBE 模式不打印。

日志格式如下：

    [MMRS_DFX] workspace_seg1 row=0 cols[0:3]=[v0, v1, v2, v3]
    [MMRS_DFX] workspace_seg2 row=0 cols[0:3]=[v0, v1, v2, v3]

例如现有 BF16 seg1 的默认点 `[0,0]`，在 `param_seg.py` 所在目录运行：

```bash
python3 param_seg.py 0/workspace_seg1_type8_MatmulReduceScatterV2.61.19.20260828084601038 --dtype bf16 --rows 1 --cols 12288 --row 0 --col 0
```

`--rows 1` 仅作前缀预览，完整矩阵仍为 8192×12288。脚本仍读取整个文件，但只解码 rows*cols 个元素。
若选第 R 行，预览至少指定 `--rows R+1` 的实际数值；`--row R --col C` 再筛选输出。
Linux 的 `param_seg.py` 需同步 BF16 `<H`（2 字节）修正。

编译时设置 `ASCENDC_DUMP=0` 会关闭这些直接打印，但不会关闭异常 workspace 分段导出。
性能回归前应关闭打印并重新编译。
打印有缓冲，不保证设备严重异常时还能输出；没有实机文件对比，不能据此认定数据一致。
SDK 输出行为参考 [printf 文档](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0193.html)。
