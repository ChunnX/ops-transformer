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
- 所有解析结果统一导出为 NPY，需要 NumPy。
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
- 脚本递归查找匹配文件，并在 `mc2_dump_analysis` 下为每个源文件生成一份独立 NPY；不会再生成合并的 `analysis.txt`，再次运行时会移除该算子目录里遗留的旧版 `analysis.txt`。

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
- `exception_info.*`：原始异常 Dump；由仓内 Python 代码直接解析 protobuf 头和各数据区，不调用 `msaicerr.py`。
- 算子 `.json`：仅在异常头中的 dtype 为 `undefined` 时作为 dtype 补充信息；`.o`、`_host.o` 不需要参与数据解包。

### 输出目录

默认输出到：

```text
<dump_dir>/mc2_dump_analysis/<CanonicalOperatorName>/
```

主要结果包括：

- 每个 Tiling、Args、Workspace、XN/CKE、通信上下文和 Peermem 输入文件各生成一个同名 `.npy`，并保留原 rank 子目录。
- Tiling、Args 和 `type12/RUNTIME_INFO` 为对象型 NPY；XN/CKE 为二维 `uint64`。
- MMRS `type8/MM_RESULT`、`type9/RECV_BUF` 根据同次 TilingData 自动保存输出 dtype 和 `(rankM, rankN)`；BF16 以 `uint16` 原始位模式存储。
- 无法可靠确定 dtype/shape 的其他 Workspace/Peermem 仍完整保存为 `uint8`。
- `exception/<exception_info文件名>/input.N.<dtype>.npy`：每个异常输入独立保存。
- `exception/<exception_info文件名>/output.N.<dtype>.npy`：每个异常输出独立保存。
- `exception/<exception_info文件名>/workspace.N.raw_bytes.npy`：每个异常 Workspace 独立保存。
- `exception/<exception_info文件名>/exception_metadata.npy`：异常头、地址、shape、dtype 和分段偏移等元数据。

## 4. 解析 exception_info 并导出 NPY

统一入口直接使用仓内解析器完成：

```text
exception_info.*
    -> 读取 8B protobuf headerLength
    -> 解析 DumpData protobuf 头
    -> 按 input/output/buffer/workspace 的 size 切分连续 payload
    -> 每段单独导出 NPY
```

不需要 CANN Python 环境、`ms_interface`、`libascend_dump_parser.so` 或 `msaicerr.py`：

```bash
python3 parse_mc2_dump.py mmrs /path/to/dump
```

也可以只解析单个原始异常文件：

```bash
python3 parse_mc2_exception_report.py /path/to/exception_info.61.154.xxx \
  --operator MatmulReduceScatterV2 \
  --export-npy /path/to/npy_output
```

生成的 `exception_metadata.npy` 会标注每个数组的 `role`、`index`、payload 偏移、地址、dtype、shape 和解释方式：

- 文件名携带已支持 dtype：按该 dtype 保存。
- BF16/FP8 等 NumPy 无法稳定直接表示的格式：用 `uint16`/`uint8` 保存原始位模式，标记 `raw_bits`。
- dtype 为 `undefined` 或未携带 dtype：用 `uint8` 无损保存，标记 `raw_bytes`。
- 异常头中的 shape 与实际元素个数一致时，NPY 直接保存该多维 shape；不一致时按一维无损保存并在元数据中同时保留逻辑 shape。

对象型结果使用 `allow_pickle=True` 查看：

```bash
python3 -c "import numpy as np; d=np.load('/path/to/tiling_data_xxx.npy', allow_pickle=True).item(); print(d)"
```

数值型结果可直接切片，例如查看 XN/CKE 前两行、前八列：

```bash
python3 -c "import numpy as np; a=np.load('/path/to/xn_addr_info_xxx.npy'); print(a[:2, :8])"
```

MMRS `type8/type9` 的 NPY 已带二维 shape，可以直接切片；其中 BF16 NPY 保存的是 `uint16` 位模式。需要把 BF16 解码成浮点数，或按 FP8/E8M0 和指定行列查看时，继续对原始 `workspace_seg*` 文件使用已有的 `parse_matrix.py`。

## 5. 单独解析某个文件

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

## 6. 按行列查看 Workspace 数据

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

## 7. Workspace Segment 怎么看

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

统一入口会把 MMRS `type8/MM_RESULT` 和 `type9/RECV_BUF` 与时间最接近的同 stream/task TilingData 配对，并按输出 dtype、`rankM`、`rankN` 保存二维 NPY。`type13/MM_WORKSPACE` 是 Matmul 内部临时区，不是结果矩阵；由于它没有固定 dtype 和逻辑 shape，脚本仍以一维 `uint8` 保存，不会强行解释成 BF16 或 FP16。

## 8. 如何确定 dtype、rows 和 cols

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

## 9. 常见问题

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

### exception_info 提示 protobuf headerLength 无效

说明该文件不完整，或并非本工具支持的 `DumpData` 异常容器。先核对文件是否传输完整，并检查文件开头 8 字节记录的 headerLength 是否小于 `文件大小 - 8`。该错误与 CANN 环境变量无关。

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

## 10. 推荐流程

```text
执行算子并生成 Dump
        ↓
使用 parse_mc2_dump.py 按算子名解析目录
        ↓
使用 NumPy 查看独立的 tiling/args/workspace NPY
        ↓
使用 parse_matrix.py 查看原始目标 Segment 的指定行列
        ↓
结合 Kernel 地址偏移和数据流核对结果
```
