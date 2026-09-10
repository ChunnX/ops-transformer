# MC2 Dump 解析脚本使用说明

本文说明 `mc2/tools/dump_analysis` 下 MC2 DFX Dump 解析脚本的使用方法。日常测试优先使用统一入口 `parse_mc2_dump.py`：传入算子名和 Dump 根目录后，脚本自动识别对应文件，并为每个源文件生成一份独立、可直接查看的 TXT 解析结果。

统一入口不解析 `exception_info.*`，不生成 NPY，也不生成合并的 `analysis.txt`。

## 1. 支持的算子和通路

| 算子 | 通路 | 推荐别名 |
| --- | --- | --- |
| `AllGatherMatmulV2` | CCU | `agmmv2` |
| `AllGatherMatmulV3` | AIV + URMA | `agmmv3` |
| `AlltoAllMatmul` | 主线 CCU | `a2amm` |
| `AlltoAllMatmul` | Apace CCU | `a2amm` |
| `AlltoAllMatmulV2` | AIV + URMA | `a2ammv2` |
| `MatmulReduceScatterV2` | CCU | `mmrs` |

`AlltoAllMatmul` 的主线 CCU 和 Apace CCU 使用同一个命令，脚本根据 TilingData 自动识别通路。

## 2. 环境要求

- Python 3.8 或更高版本。
- 统一入口只使用 Python 标准库，不需要 NumPy。
- 建议在 `mc2/tools/dump_analysis` 目录执行，也可以使用脚本绝对路径。

```bash
cd ops-transformer_dfx/mc2/tools/dump_analysis
```

## 3. 最简使用方法

```bash
python3 parse_mc2_dump.py <operator> [dump_dir]
```

- `<operator>`：算子名或别名，不区分大小写。
- `[dump_dir]`：Dump 根目录；省略时默认使用当前目录。

常用命令：

```bash
# AllGatherMatmulV2，CCU
python3 parse_mc2_dump.py agmmv2 /path/to/dump

# AllGatherMatmulV3，AIV + URMA
python3 parse_mc2_dump.py agmmv3 /path/to/dump

# AlltoAllMatmul，自动识别主线 CCU 或 Apace CCU
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

## 4. 自动识别的文件

### 4.1 所有 MC2 DFX 通路的核心文件

| 文件 | 是否应有 | 解析内容 |
| --- | --- | --- |
| `tiling_data_*` | 必有 | Tiling 参数、量化模式、通信参数、WorkspaceLayout 及各 Segment 的偏移和长度 |
| `args_info_*` | 必有 | Kernel 参数对应的 GM 地址 |
| `workspace_segN_typeM_*` | 必有 | 保存实际采集到的 Workspace Segment；每段一个文件，文件序号不一定从 0 开始或连续 |

这里的“必有”是指算子已经正确开启 MC2 DFX Dump。若缺少其中一类，应优先检查 DFX 开关、异常回调注册、TilingData 中的 WorkspaceLayout，以及 Dump 目录是否选对。

### 4.2 CCU 通路文件

以下文件只属于 CCU 通路，包括 AllGatherMatmulV2、AlltoAllMatmul 主线/Apace 和 MatmulReduceScatterV2：

| 文件 | 是否应有 | 解析内容 |
| --- | --- | --- |
| `xn_addr_info_*` | 应有 | XN 地址表，按小端 `uint64`、每行 64 个地址解析 |
| `cke_addr_info_*` | 应有 | CKE 地址表，按小端 `uint64`、每行 64 个地址解析 |

AIV + URMA 通路没有 XN/CKE 文件。

### 4.3 AIV + URMA 通路文件

下面文件只有 URMA 通路才会生成，即 AllGatherMatmulV3 和 AlltoAllMatmulV2：

| 文件 | 是否应有 | 解析内容 |
| --- | --- | --- |
| `comm_context_*` | 应有 | URMA 通信上下文，包括 rank、channel 和通信缓冲区地址；结果中也会给出使用 `parse_matrix.py` 查看原始字节的命令 |

CCU 通路没有 `comm_context_*` 文件。

### 4.4 不参与统一解析的文件

- `exception_info.*`：统一入口不再读取或解析。
- `peermem_window_*`、`peermem_segN_typeM_*`：统一入口不读取或解析。
- 算子 `.o`、`*_host.o`、`.json`：不属于本脚本的 MC2 Tiling/Workspace 解析输入，会被忽略。
- 已生成的 `mc2_dump_analysis` 目录：递归扫描时会跳过，避免重复解析输出文件。

## 5. TXT 输出目录和内容

结果默认输出到：

```text
<dump_dir>/mc2_dump_analysis/<标准算子名>/
```

如果原始 Dump 带 rank 子目录，输出会保留相同目录层级。例如：

```text
<dump_dir>/
└── mc2_dump_analysis/
    └── MatmulReduceScatterV2/
        ├── 0/
        │   ├── tiling_data_MatmulReduceScatterV2.xxx.txt
        │   ├── args_info_MatmulReduceScatterV2.xxx.txt
        │   ├── workspace_seg1_type8_MatmulReduceScatterV2.xxx.txt
        │   └── workspace_seg2_type12_MatmulReduceScatterV2.xxx.txt
        └── 1/
            └── ...
```

输出规则：

- 每个识别到的源文件生成一个同名 `.txt`，不同调用、不同 rank 不会混在一起。
- 不生成 NPY。
- 不生成汇总 `analysis.txt`；再次运行时会删除旧版本遗留的该文件。
- 解析结果同时显示在终端，便于直接观察执行情况。
- 统一入口只解析实际扫描到的受支持文件，不会因为缺少某一类文件而单独报错；应根据第 4 节人工核对当前通路预期的文件是否齐全。

各类 TXT 的主要内容：

| 类型 | TXT 内容 |
| --- | --- |
| TilingData | 脚本支持的字段化结果、量化/通路信息、WorkspaceLayout、Segment 偏移和大小 |
| ArgsInfo | Kernel 参数名及对应 GM 地址 |
| 普通 Workspace Segment | `segIdx`、`segType`、含义、文件大小和前 64 字节十进制预览 |
| `type12/RUNTIME_INFO` | 显示每个核的核号、是否已写入、当前执行位置、通信阶段、commit 次数和 wait 次数。从 TilingData 读取到 AIC 核数时，还会标明该核是 C 核还是 V 核；MMRS 另外显示未完成通信数和通信状态。 |
| XN/CKE | 地址表的行列数、非零地址数量和逐行十六进制地址 |
| URMA CommContext | 通信上下文字段，以及使用 `parse_matrix.py` 查看原始字节的命令提示 |

普通数据型 Workspace 可能非常大，因此统一入口不会把整个二进制展开成文本；需要查看指定行列时使用 `parse_matrix.py`。

## 6. 单独解析某个文件

| 算子 | 后端解析脚本 |
| --- | --- |
| AllGatherMatmulV2 | `parse_all_gather_matmul_v2_tiling.py` |
| AllGatherMatmulV3 | `parse_all_gather_matmul_v3_tiling.py` |
| AlltoAllMatmul | `parse_allto_all_matmul_tiling.py` |
| AlltoAllMatmulV2 | `parse_allto_all_matmul_v2_tiling.py` |
| MatmulReduceScatterV2 | `parse_matmul_reduce_scatter_v2_tiling.py` |

### 6.1 入参说明

基本调用格式：

```bash
python3 <后端解析脚本> <bin_file>
```

固定入参：

| 入参 | 是否必填 | 说明 |
| --- | --- | --- |
| `<后端解析脚本>` | 是 | 根据算子选择本节开头表格中的后端解析脚本。 |
| `<bin_file>` | 是 | 要解析的 Dump 文件路径。文件通常没有 `.bin` 后缀，应直接填写实际文件名。 |

解析模式用于指定输入文件的解析方式，写在 `<bin_file>` 后面：

```bash
python3 <后端解析脚本> <bin_file> [解析模式]
```

| 解析模式 | 输入文件/适用范围 | 说明 |
| --- | --- | --- |
| 不加解析模式 | `tiling_data_*` | 默认按完整 TilingData 解析。 |
| `--args` | `args_info_*`；所有脚本 | 按 Kernel 参数顺序解析各输入、输出、Workspace 和 TilingData 的 GM 地址。 |
| `--ws-layout` | `tiling_data_*`；所有脚本 | 只解析 TilingData 中的 `WorkspaceLayoutInfo`，查看每个 Segment 的 `offset`、`size` 和 `type`。 |
| `--workspace` | `workspace_segN_typeM_*`；所有脚本 | 按文件名中的 Segment 类型解析 Workspace；识别到 `type12` 时，直接按每核 RuntimeInfo 结构解析并显示运行状态。 |
| `--comm-ctx` | `comm_context_*`；仅 AllGatherMatmulV3、AlltoAllMatmulV2 | 将文件按 URMA `CommUdmaContext + CommUbmemContext` 解析，显示 rank、channel 和通信缓冲区地址。 |

上述解析模式一次命令只应选择一个。解析完整 TilingData 时不加解析模式。

辅助参数按需追加在命令末尾：

```bash
python3 <后端解析脚本> <bin_file> [解析模式] [辅助参数 ...]
```

| 辅助参数 | 适用范围 | 说明 |
| --- | --- | --- |
| `--aic-core-num N` | 配合 `--workspace` 解析 `type12` | 指定实际 AIC 核数，用于区分 AIC/AIV 核；MMRS 还会据此计算预期 slot 数。`N` 取同次 TilingData 中的 `aicCoreNum` 或 `usedCoreNum`。 |
| `--apace` | 仅 AlltoAllMatmul，配合 `--workspace` | 按 Apace 通路解释 RuntimeInfo 的 position 标签。只影响标签含义，不改变原始数值。 |
| `--json` | 所有脚本 | 将本次解析结果输出为 JSON，便于其他程序继续处理。 |

### 6.2 MMRS 示例

```bash
# 解析 TilingData
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/tiling_data_MatmulReduceScatterV2.xxx

# 只查看 WorkspaceLayout
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/tiling_data_MatmulReduceScatterV2.xxx --ws-layout

# 解析 ArgsInfo
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/args_info_MatmulReduceScatterV2.xxx --args

# 解析 type12 RuntimeInfo
python3 parse_matmul_reduce_scatter_v2_tiling.py \
  /path/to/workspace_seg2_type12_MatmulReduceScatterV2.xxx \
  --workspace --aic-core-num 24
```

示例中的 `seg2` 和核数 `24` 都不是固定值：Segment 序号以实际文件名为准，MMRS 的核数应填写该次 TilingData 中的 `aicCoreNum`。使用统一入口时会自动尝试读取核数，一般不需要手工填写。

### 6.3 URMA 示例

```bash
python3 parse_all_gather_matmul_v3_tiling.py \
  /path/to/comm_context_AllGatherMatmulV3.xxx --comm-ctx
```

## 7. 按行列查看 Workspace 数据或原始字节

`parse_matrix.py` 用于按数据类型、矩阵形状和行列范围读取 Workspace 数据，也可以按 `uint8` 查看其他文件的原始字节。

基本格式：

```bash
python3 parse_matrix.py <bin_file> --dtype <type> --rows <M> --cols <N> [显示范围参数]
```

必填参数：

| 参数 | 说明 |
| --- | --- |
| `<bin_file>` | Workspace Segment 文件路径；查看 `comm_context_*` 原始字节时也可使用该脚本。 |
| `--dtype <type>` | 文件中元素的实际数据类型，不能只根据 Segment 类型猜测，应结合 TilingData 和 Kernel 数据流确认。 |
| `--rows <M>` | 原始文件对应矩阵的总行数，不是本次要打印的行数。例如 `--rows 8192` 表示完整矩阵有 8192 行。 |
| `--cols <N>` | 原始文件对应矩阵每一行的元素个数，不是本次要打印的列数。例如 `--cols 12288` 表示每行有 12288 个元素。 |

`rows` 和 `cols` 共同描述完整矩阵的形状。例如 `--rows 8192 --cols 12288` 表示把文件解释为一个 `8192 × 12288` 的矩阵，总元素数为 `8192 × 12288`。默认按行优先排列，第 `row` 行、第 `col` 列在文件中的元素下标为：

```text
元素下标 = row × cols + col
```

因此，`rows` 和 `cols` 必须根据 TilingData、算子输出矩阵形状以及 Kernel 的 Workspace 数据布局填写。要控制本次实际打印哪些行和列，应使用 `--row`、`--row-range`、`--col` 或 `--col-range`。

显示范围和输出参数：

| 参数 | 说明 |
| --- | --- |
| `--head N` | 未指定行号或行范围时显示前 `N` 行，默认 16 行。 |
| `--tail N` | 在 `--head` 选择的前部行之外，再显示最后 `N` 行，默认不显示。 |
| `--row-range START END` | 显示连续行范围 `[START, END)`。 |
| `--col-range START END` | 显示连续列范围 `[START, END)`。 |
| `--row R [R ...]` | 显示一个或多个离散行号。 |
| `--col C [C ...]` | 显示一个或多个离散列号。 |
| `--transpose` | 文件按列优先布局时使用，逻辑坐标按 `c × rows + r` 取数。 |
| `--json` | 将选中的行列输出为 JSON。 |

`--row` 与 `--row-range` 不要同时使用，`--col` 与 `--col-range` 也不要同时使用。

查看前 10 行、每行第 0～63 列：

```bash
python3 parse_matrix.py \
  /path/to/workspace_seg1_type8_MatmulReduceScatterV2.xxx \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row-range 0 10 --col-range 0 64
```

范围采用左闭右开规则：`--row-range 0 10` 表示第 0～9 行，`--col-range 0 64` 表示第 0～63 列。

查看离散行列：

```bash
python3 parse_matrix.py workspace.bin \
  --dtype bf16 --rows 8192 --cols 12288 \
  --row 0 8 16 --col 0 1 63 64
```

支持的 `--dtype`：`bf16`、`fp16`、`fp32`、`int8`、`uint8`、`fp8_e4m3`、`fp8_e5m2`、`e8m0`、`fp4_e2m1`。

查看 `comm_context_*` 原始字节时，使用 `--dtype uint8 --rows 1 --cols <文件字节数>`；这只是字节视图，结构化字段仍以 `--comm-ctx` 的解析结果为准。

## 8. Workspace Segment 含义

文件名中的 `segN` 是本次 WorkspaceLayout 的段序号，`typeM` 才是数据语义。段序号可能随通路和条件变化，不能只凭 `segN` 判断内容。

只会为实际采集到的 Segment 生成文件。系统预留段、空段或未采集段可能没有对应文件，因此看到的 `segN` 不一定从 0 开始，也不一定连续。

| type | 含义 |
| ---: | --- |
| 0 | `LIB_API`，系统库预留 Workspace |
| 1 | `ND2NZ`，格式转换缓冲区 |
| 2 | `GATHER`，Gather 数据缓冲区 |
| 3 | `GATHER_SCALE1` |
| 4 | `GATHER_SCALE` |
| 5 | `COMM_OUT`，通信输出缓冲区 |
| 6 | `PERMUTE_OUT`，转置输出缓冲区 |
| 7 | `BIAS` |
| 8 | `MM_RESULT`，Matmul 结果/部分通路的通信发送区 |
| 9 | `RECV_BUF`，通信接收区 |
| 10 | `COMM_INT8` |
| 11 | `DYNAMIC_QUANT` |
| 12 | `RUNTIME_INFO`，每核运行和通信状态 |
| 13 | `MM_WORKSPACE`，Matmul 内部临时区，不是结果矩阵 |
| 255 | `RESERVED`，对齐或预留区域 |

最终含义应同时核对：

1. 当前分支的 `WorkspaceSegType` 枚举。
2. Tiling 侧 `BuildWorkspaceLayout()` 的 Segment 顺序、offset 和 size。
3. Kernel 侧 GlobalTensor 的 Workspace 地址偏移和数据类型。

## 9. RuntimeInfo 结果怎么看

`type12/RUNTIME_INFO` 是按核划分的运行状态区，不是矩阵数据，不能按普通矩阵的数据类型和行列数解释。当前每个核占用 512 字节，脚本把这 512 字节称为一个 `slot`。

各算子后端都会解析这些基础字段：

- `magic` / `magicValid`：该 slot 是否已被 Kernel 初始化；有效值为 `0x5A5A5A5A`。
- `coreId` / `coreType`：核编号和核类型。没有取得本次运行的 AIC 核数时，核类型会显示为“未知”。
- `turn`：当前执行轮次。
- `position`：Kernel 记录的执行位置，标签含义与算子及通路有关。
- `phase`：当前通信阶段。
- `commit`：该核已提交的通信次数。
- `wait`：该核已等待完成的通信次数。

多数后端只展开 `magicValid=true` 的 slot；如果 `activeSlots=0`，说明 Dump 中没有找到已初始化的 RuntimeInfo slot，应检查 RuntimeInfo Segment 起始地址、Kernel 初始化是否执行，以及异常 Dump 的采集时机。

MMRS 后端还会显示以下汇总和状态字段：

- `expectedSlots`：预计会使用的核记录数量。当前 MMRS 中，若 AIC 核数为 `N`，对应 AIV 核数为 `2N`，所以预计使用 `3N` 个 slot；未传入 `--aic-core-num` 时只能按文件中的最大 slot 数显示。
- `validExpectedSlots`：预期范围内通过 magic 校验的 slot 数。
- `pending = max(commit - wait, 0)`。
- `BALANCED`：commit 与 wait 相等。
- `PENDING`：commit 大于 wait，仍有通信未完成。
- `COUNT_ERROR`：wait 大于 commit，计数关系异常。
- `NOT_WRITTEN`：magic 无效，该 slot 没有被 Kernel 初始化。

## 10. 推荐排查流程

```text
执行算子并生成 MC2 DFX Dump
        ↓
使用 parse_mc2_dump.py 按算子名解析目录
        ↓
分别查看对应 TilingData/ArgsInfo/Workspace TXT
        ↓
用 TilingData TXT 核对 WorkspaceLayout 的 offset 和 size
        ↓
需要看具体数值时，用 parse_matrix.py 查看目标 Segment 的指定行列
        ↓
结合 Kernel Workspace 地址偏移和数据流确认问题阶段
```

常见检查项：

- 提示“没有找到受支持 Dump 文件”：确认 Dump 根目录和算子名正确，并检查文件名是否包含对应算子名。
- Dump 目录中没有 XN/CKE：先确认当前是否确实为 CCU 通路。
- Dump 目录中没有 CommContext：先确认当前是否为 AIV + URMA 通路。
- Workspace 文件大小与 TilingData 中 Segment size 不一致：检查段偏移、对齐/padding 和 Dump 时机。
- RuntimeInfo 全部无效：检查 Kernel 初始化、Workspace 起始地址以及异常发生阶段。
- RuntimeInfo 中核类型显示为“未知”：从对应 TilingData 中找到 `aicCoreNum` 或 `usedCoreNum`，再使用后端脚本的 `--aic-core-num N` 重新解析。
