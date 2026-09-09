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
| `workspace_segN_typeM_*` | 必有 | 按 WorkspaceLayout 逐段生成；每个有效 Segment 对应一份文件 |

这里的“必有”是指算子已经正确开启 MC2 DFX Dump。若缺少其中一类，应优先检查 DFX 开关、异常回调注册、TilingData 中的 WorkspaceLayout，以及 Dump 目录是否选对。

### 4.2 CCU 通路文件

以下文件只属于 CCU 通路，包括 AllGatherMatmulV2、AlltoAllMatmul 主线/Apace 和 MatmulReduceScatterV2：

| 文件 | CCU 通路 | 解析内容 |
| --- | --- | --- |
| `xn_addr_info_*` | 应有 | XN 地址表，按小端 `uint64`、每行 64 个地址解析 |
| `cke_addr_info_*` | 应有 | CKE 地址表，按小端 `uint64`、每行 64 个地址解析 |

AIV + URMA 通路没有 XN/CKE 文件。

### 4.3 AIV + URMA 通路文件

下面文件只有 URMA 通路才会生成，即 AllGatherMatmulV3 和 AlltoAllMatmulV2：

| 文件 | URMA 通路 | 解析内容 |
| --- | --- | --- |
| `comm_context_*` | 应有 | URMA 通信上下文；解析结果会提示使用 `parse_matrix.py` 查看原始字节 |

CCU 通路没有 `comm_context_*` 文件。

### 4.4 不参与统一解析的文件

- `exception_info.*`：统一入口不再读取或解析。
- `peermem_window_*`、`peermem_segN_typeM_*`：统一入口不读取或解析。
- 算子 `.o`、`*_host.o`、`.json`：不属于本脚本的 MC2 Tiling/Workspace 解析输入，会被忽略。
- 已生成的 `mc2_dump_analysis` 目录：递归扫描时会跳过，避免重复解析输出文件。

## 5. TXT 输出目录和内容

结果默认输出到：

```text
<dump_dir>/mc2_dump_analysis/<CanonicalOperatorName>/
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
- 启动时会检查当前通路预期的文件类别；缺少核心文件或通路专有文件时在终端显示警告，但仍继续解析已经存在的文件。

各类 TXT 的主要内容：

| 类型 | TXT 内容 |
| --- | --- |
| TilingData | 完整字段、量化/通路信息、WorkspaceLayout、Segment 偏移和大小 |
| ArgsInfo | Kernel 参数名及对应 GM 地址 |
| 普通 Workspace Segment | `segIdx`、`segType`、含义、文件大小和前 64 字节十进制预览 |
| `type12/RUNTIME_INFO` | 关联的 TilingData、推导出的 AIC 核数、通路，以及每核 magic、position、phase、commit、wait、pending 和状态 |
| XN/CKE | 地址表 shape、非零地址数量和逐行十六进制地址 |
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

调用格式：

```bash
python3 <后端解析脚本> <bin_file> [解析模式] [辅助参数]
```

| 入参 | 是否必填 | 输入文件/适用范围 | 说明 |
| --- | --- | --- | --- |
| `<bin_file>` | 是 | 所有脚本 | 要解析的 Dump 文件路径。文件通常没有 `.bin` 后缀，应直接填写实际文件名。 |
| 不加解析模式 | 否 | `tiling_data_*` | 默认按完整 TilingData 解析。 |
| `--args` | 否 | `args_info_*`；所有脚本 | 按 Kernel 参数顺序解析各输入、输出、Workspace 和 TilingData 的 GM 地址。 |
| `--ws-layout` | 否 | `tiling_data_*`；所有脚本 | 只解析 TilingData 中的 `WorkspaceLayoutInfo`，查看每个 Segment 的 `offset`、`size` 和 `type`。 |
| `--workspace` | 否 | `workspace_segN_typeM_*`；所有脚本 | 按文件名中的 Segment 类型解析 Workspace。`type12` 会进一步解析为每核 RuntimeInfo。 |
| `--aic-core-num N` | 否 | 配合 `--workspace` 解析 `type12` | 指定实际 AIC 核数，用于计算预期 slot 数并区分 AIC/AIV 核。`N` 取同次 TilingData 中的 `aicCoreNum` 或 `usedCoreNum`。 |
| `--comm-ctx` | 否 | `comm_context_*`；仅 AllGatherMatmulV3、AlltoAllMatmulV2 | 将文件按 URMA `CommUdmaContext + CommUbmemContext` 解析，显示 rank、channel 和通信缓冲区地址。 |
| `--cann DIR` | 否 | 仅 AllGatherMatmulV2、AlltoAllMatmul、MatmulReduceScatterV2 | 指定生成该 Dump 时使用的 CANN 根目录，使脚本按对应版本的头文件计算 TilingData 结构偏移；不指定时从当前 CANN 环境定位。 |
| `--apace` | 否 | 仅 AlltoAllMatmul，配合 `--workspace` | 按 Apace 通路解释 RuntimeInfo 的 position 标签。只影响标签含义，不改变原始数值。 |
| `--json` | 否 | 所有脚本 | 将本次解析结果输出为 JSON，便于其他程序继续处理。 |
| `--verify REF_FILE` | 否 | 仅 AllGatherMatmulV3、AlltoAllMatmul | 将 TilingData Dump 与指定的 Host 参考文件逐字节比较；`REF_FILE` 是参考文件路径。 |

`--args`、`--ws-layout`、`--workspace` 和 `--comm-ctx` 表示不同的文件解析模式，一次命令只应选择其中一个。解析 TilingData 时不加这些模式参数。

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

`--aic-core-num` 应填写该次 TilingData 中记录的实际 AIC Core 数量。统一入口会自动从同次 TilingData 获取，一般不需要手工填写。

### 6.3 URMA 示例

```bash
python3 parse_all_gather_matmul_v3_tiling.py \
  /path/to/comm_context_AllGatherMatmulV3.xxx --comm-ctx
```

## 7. 按行列查看 Workspace 数据

`parse_matrix.py` 用于按数据类型、矩阵 shape 和行列范围读取 Workspace 原始文件。

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

## 8. Workspace Segment 含义

文件名中的 `segN` 是本次 WorkspaceLayout 的段序号，`typeM` 才是数据语义。段序号可能随通路和条件变化，不能只凭 `segN` 判断内容。

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

`type12/RUNTIME_INFO` 不是矩阵数据。其 TXT 会按核显示：

- `expectedSlots`：根据 TilingData 中的 AIC 核数和 Kernel task 类型得到的预期 slot 数；`MIX_AIC_1_2` 为 `3 × AIC`，`MIX_AIC_1_1` 为 `2 × AIC`，AlltoAllMatmul Apace 的 `AIC_ONLY` 为 `1 × AIC`。
- `validExpectedSlots`：预期范围内 magic 校验通过的 slot 数。
- `commit`：该核已提交的通信次数。
- `wait`：该核已等待完成的通信次数。
- `pending = commit - wait`。
- `BALANCED`：commit 与 wait 相等。
- `PENDING`：commit 大于 wait，仍有通信未完成。
- `COUNT_ERROR`：wait 大于 commit，计数关系异常。
- `NOT_WRITTEN`：magic 无效，该 slot 没有被 Kernel 初始化。

若所有预期核都是 `NOT_WRITTEN`，应检查 RuntimeInfo Segment 起始地址、Kernel 初始化是否执行，以及异常 Dump 的采集时机。

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
- 缺少 XN/CKE：先确认当前是否确实为 CCU 通路。
- 缺少 CommContext：先确认当前是否为 AIV + URMA 通路。
- Workspace 文件大小与 TilingData 中 Segment size 不一致：检查段偏移、对齐/padding 和 Dump 时机。
- RuntimeInfo 全部无效：检查 Kernel 初始化、Workspace 起始地址以及异常发生阶段。
- 同一 rank 目录包含多次调用时，统一入口会优先按文件名末尾的 Dump 序号/时间戳关联最近的 TilingData；可在 RuntimeInfo TXT 的 `associatedTiling` 字段确认配对结果。
