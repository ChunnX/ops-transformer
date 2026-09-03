# MC2 Dump 解析脚本使用说明

本文说明 `mc2/tools/dump_analysis` 下 Dump 解析脚本的使用方法。日常测试优先使用统一入口 `parse_mc2_dump.py`，通常只需提供算子名和 Dump 目录。

## 1. 支持的算子通路

| 算子 | 通路 | 推荐别名 |
| --- | --- | --- |
| AllGatherMatmulV2 | CCU | `agmmv2` |
| AllGatherMatmulV3 | AIV + URMA | `agmmv3` |
| AlltoAllMatmul | 主线 CCU | `a2amm` |
| AlltoAllMatmul | Apace CCU | `a2amm` |
| AlltoAllMatmulV2 | AIV + URMA | `a2ammv2` |
| MatmulReduceScatterV2 | CCU | `mmrs` |

AlltoAllMatmul 的主线和 Apace 通路由脚本根据 TilingData 自动识别，无需额外指定。

## 2. 环境和目录

- 使用 Python 3.8 或更高版本。
- 脚本仅使用 Python 标准库，无需安装额外依赖。
- 建议在 `mc2/tools/dump_analysis` 目录执行命令；也可以使用脚本绝对路径。

```bash
cd ops-transformer_dfx/mc2/tools/dump_analysis
```

## 3. 按算子名解析整个 Dump 目录

统一入口格式：

```bash
python3 parse_mc2_dump.py <operator> [dump_dir]
```

- `<operator>`：算子名或别名，不区分大小写。
- `[dump_dir]`：Dump 根目录；省略时默认使用当前目录。
- 脚本递归查找匹配文件，并把结果集中写入 `mc2_dump_analysis`。

### 常用命令

```bash
# AllGatherMatmulV2，CCU
python3 parse_mc2_dump.py agmmv2 /path/to/dump

# AllGatherMatmulV3，AIV + URMA
python3 parse_mc2_dump.py agmmv3 /path/to/dump

# AlltoAllMatmul，自动识别主线或 Apace 通路
python3 parse_mc2_dump.py a2amm /path/to/dump

# AlltoAllMatmulV2，AIV + URMA
python3 parse_mc2_dump.py a2ammv2 /path/to/dump

# MatmulReduceScatterV2，CCU
python3 parse_mc2_dump.py mmrs /path/to/dump
```

当前目录就是 Dump 根目录时，可以省略目录参数：

```bash
python3 /path/to/ops-transformer_dfx/mc2/tools/dump_analysis/parse_mc2_dump.py mmrs
```

查看支持的名称和参数：

```bash
python3 parse_mc2_dump.py --help
```

### 自动识别的文件

- `tiling_data_*`：TilingData。
- `args_info_*`：Kernel 参数地址信息。
- `workspace_segN_typeM_*`：Workspace 分段数据。
- `xn_addr_info_*`：XN 地址信息。
- `cke_addr_info_*`：CKE 地址信息。
- `comm_context_*`：URMA 通信上下文。
- `peermem_window_*`、`peermem_seg_*`：URMA Peer Memory 信息。

### 输出目录

默认输出到：

```text
<dump_dir>/mc2_dump_analysis/<CanonicalOperatorName>/
```

主要结果包括：

- `analysis.txt`：TilingData、Args、Workspace 等汇总结果。
- XN、CKE、通信上下文和 Peer Memory 的独立解析结果。

例如：

```text
/path/to/dump/mc2_dump_analysis/MatmulReduceScatterV2/analysis.txt
```

## 4. 单独解析某个文件

需要 JSON、Workspace Layout 或只解析一个文件时，可直接调用算子脚本。

| 算子 | 解析脚本 |
| --- | --- |
| AllGatherMatmulV2 | `parse_all_gather_matmul_v2_tiling.py` |
| AllGatherMatmulV3 | `parse_all_gather_matmul_v3_tiling.py` |
| AlltoAllMatmul | `parse_allto_all_matmul_tiling.py` |
| AlltoAllMatmulV2 | `parse_allto_all_matmul_v2_tiling.py` |
| MatmulReduceScatterV2 | `parse_matmul_reduce_scatter_v2_tiling.py` |

先查看单个脚本支持的参数：

```bash
python3 parse_matmul_reduce_scatter_v2_tiling.py --help
```

### MMRS 示例

解析 TilingData：

```bash
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/tiling_data_MatmulReduceScatterV2.xxx
```

JSON 格式输出：

```bash
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/tiling_data_MatmulReduceScatterV2.xxx --json
```

查看 Workspace 分段布局：

```bash
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/tiling_data_MatmulReduceScatterV2.xxx --ws-layout
```

解析 Kernel 参数地址：

```bash
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/args_info_MatmulReduceScatterV2.xxx --args
```

解析 `type12` RuntimeInfo：

```bash
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/workspace_seg2_type12_MatmulReduceScatterV2.xxx \
  --workspace --aic-core-num 24
```

`--aic-core-num` 应填写当前设备实际使用的 AIC Core 数量。

### URMA 通信信息示例

AllGatherMatmulV3 和 AlltoAllMatmulV2 可能生成通信上下文和 Peer Memory 文件：

```bash
python3 parse_all_gather_matmul_v3_tiling.py \
  /path/to/comm_context_AllGatherMatmulV3.xxx --comm-ctx

python3 parse_all_gather_matmul_v3_tiling.py \
  /path/to/peermem_window_AllGatherMatmulV3.xxx --peermem
```

AlltoAllMatmulV2 的参数相同，替换成对应解析脚本即可。

## 5. 按行列查看 Workspace 数据

`parse_matrix.py` 用来按照指定数据类型、矩阵形状和行列范围读取 Workspace 原始数据。

### 查看前 10 行、每行第 0～63 列

```bash
python3 parse_matrix.py \
  /path/to/workspace_seg1_type8_MatmulReduceScatterV2.xxx \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row-range 0 10 --col-range 0 64
```

范围采用左闭右开规则：`--row-range 0 10` 表示第 0～9 行，`--col-range 0 64` 表示第 0～63 列。

### 查看离散行列

```bash
python3 parse_matrix.py workspace.bin \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row 0 8 16 --col 0 1 63 64
```

### 查看头部或尾部行

```bash
python3 parse_matrix.py workspace.bin \
  --dtype fp16 --rows 8192 --cols 12288 --head 64

python3 parse_matrix.py workspace.bin \
  --dtype fp16 --rows 8192 --cols 12288 --tail 64
```

### 转置显示或输出 JSON

```bash
python3 parse_matrix.py workspace.bin \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row-range 0 10 --col-range 0 64 --transpose

python3 parse_matrix.py workspace.bin \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row-range 0 10 --col-range 0 64 --json
```

支持的 `--dtype`：`bf16`、`fp16`、`fp32`、`int8`、`uint8`、`fp8_e4m3`、`fp8_e5m2`、`e8m0`、`fp4_e2m1`。

## 6. Workspace Segment 怎么看

文件名中的 `segN` 是 Workspace 布局中的段序号，`typeM` 是段的语义类型。段序号会随实际布局变化，判断数据含义时应优先看 `typeM` 和 TilingData 中的 Workspace Layout。

| type | 当前枚举含义 |
| ---: | --- |
| 0 | Library API Workspace |
| 1 | ND2NZ |
| 2 | GATHER |
| 3 | GATHER_SCALE1 |
| 4 | GATHER_SCALE |
| 5 | COMM_OUT |
| 6 | PERMUTE_OUT |
| 7 | BIAS |
| 8 | MM_RESULT |
| 9 | RECV_BUF |
| 10 | COMM_INT8 |
| 11 | DYNAMIC_QUANT |
| 12 | RuntimeInfo |
| 13 | MM_WORKSPACE |
| 255 | RESERVED |

不同分支中的枚举可能存在差异。最终以当前代码的 `WorkspaceSegType`、Tiling 侧 `BuildWorkspaceLayout()` 和 Kernel 侧 Workspace 地址偏移为准。

统一入口对普通 Workspace 段显示前 64 个原始字节的十进制值，并同时打印 `dataMeaning`、`previewType` 和字节范围。`type13/MM_WORKSPACE` 是 Matmul 内部临时区，不是结果矩阵；MMRS 的 Matmul 输出对应 `type8/MM_RESULT`。由于 `type13` 没有固定 dtype 和逻辑 shape，脚本不会把它强行解释成 BF16 或 FP16。

## 7. 如何确定 dtype、rows 和 cols

1. 在 Tiling 代码的 `BuildWorkspaceLayout()` 中确认段长度和类型。
2. 在 Kernel 初始化代码中确认对应 GlobalTensor 的数据类型和地址偏移。
3. 根据实际逻辑形状确定 `rows` 和 `cols`，不要仅凭文件名猜测。
4. 检查文件大小是否与矩阵形状相符。

常见大小关系：

```text
BF16/FP16 文件大小 = rows × cols × 2 字节
FP32 文件大小       = rows × cols × 4 字节
FP8/INT8 文件大小   = rows × cols × 1 字节
FP4 文件大小        = ceil(rows × cols / 2) 字节
```

文件大小不匹配时，常见原因有：

- 使用了全局形状，但文件实际保存的是单卡或单 Tile 数据。
- 段内存在 Padding 或对齐空间。
- 数据类型判断错误。
- 文件属于 RuntimeInfo、Scale 或其他非普通矩阵段。

## 8. 常见问题

### 只想快速解析，使用哪个命令？

```bash
python3 parse_mc2_dump.py mmrs /path/to/dump
```

把 `mmrs` 替换为目标算子的别名即可。

### 没有找到 Dump 文件

检查以下内容：

- 路径是否指向包含 rank 子目录或 Dump 文件的根目录。
- 文件名是否包含正确的算子名。
- 算子名或别名是否正确。
- Dump 配置是否开启了 TilingData、Args 或 Workspace 分段导出。

### `workspace_seg2_type12` 为什么不是矩阵结果？

`type12` 是 RuntimeInfo，记录每个 Core 的通信位置、阶段、提交次数和等待次数等运行状态，不能按普通矩阵解析。

MMRS 解析器会按预期核逐行输出 `commit`、`wait`、`pending=commit-wait` 和状态：

- `BALANCED`：commit 与 wait 相等。
- `PENDING`：commit 大于 wait，仍有未完成通信。
- `COUNT_ERROR`：wait 大于 commit，计数关系异常。
- `NOT_WRITTEN`：magic 无效，该 slot 尚未被 RuntimeInfo 初始化。

如果所有预期核都是 `NOT_WRITTEN`，说明 Kernel 没有写入这段内存，或者 Dump 的 RuntimeInfo 起始地址/采集时机不正确。脚本还会检查 magic 是否出现在非 512B 对齐位置，并提示可能的段起点错位。

### 如何与 Kernel 中的数据对比？

- 根据 Tiling Workspace Layout 确认 Segment 地址偏移和长度。
- 根据 Kernel GlobalTensor 类型确认 `dtype`。
- 使用相同逻辑形状和相同行列范围查看数据。
- 确认比较的是通信、转置或 ReduceScatter 的同一执行阶段。

## 9. 推荐流程

```text
执行算子并生成 Dump
        ↓
使用 parse_mc2_dump.py 按算子名解析目录
        ↓
查看 analysis.txt，确认 Tiling、参数地址和 Workspace Layout
        ↓
使用 parse_matrix.py 查看目标 Segment 的指定行列
        ↓
结合 Kernel 地址偏移和数据流核对结果
```
