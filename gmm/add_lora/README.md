# AddLora

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：AddLora 是分组 MoE 场景下的 LoRA 增量注入算子。按 `indices[b]` 指示的适配器编号，从按 `(W, L)` 堆叠的 LoRA 权重中选取 `(w, layer_idx)` 对应的 `A/B` 矩阵，对每行输入 `x[b]` 计算双矩阵乘增量 `lora[b]`，乘以缩放系数 `scale` 后累加到 `y[b]` 的列片段 `[y_offset, y_offset + s)` 上（in-place，单次 FP32 加法 + 单次 FP16 舍入）。`weightA` 缺省时退化为 BGMV 单矩阵乘路径（`R == H1`）。路由越界（`indices[b]` 不在 `[0, W)`）的行与切片外的列保持 `y` 原值。

- 计算公式：

$$
t_b = x_b \cdot A_{\text{indices}[b],\,\text{layer\_idx}}^{T} \quad (\text{weightA 在场，} R \le 128)
$$

$$
\text{lora}_b = t_b \cdot B_{\text{indices}[b],\,\text{layer\_idx}}^{T}
$$

$$
y_b[\text{y\_offset} : \text{y\_offset}+s] \mathrel{+}= \text{scale} \cdot \text{lora}_b
$$

weightA 缺省（BGMV）时 `t_b = x_b`，`lora_b = x_b \cdot B^T`（K=R=H1 分块）。

- kernel 为 PyPTO-Pro DSL 编写（`op_kernel/add_lora.py`，单 `@pl.jit` MIX kernel，AIC 分组 GEMM + AIV 重排/epilogue，5 次 `sync_all(MIX)` 屏障阶段衔接）；host 侧 tiling / infershape / 算子注册由 `op_host/` 提供。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| y | 输入 | 被累加的基座输出，`[B, H3]`；`y_out` 与其同地址（in-place 绑定）。 | FLOAT16 | ND |
| x | 输入 | 每行输入向量，`[B, H1]`。 | FLOAT16 | ND |
| weightB | 输入 | LoRA B 权重堆叠，`[W, L, H2, R]`。 | FLOAT16 | ND |
| indices | 输入 | 每行的适配器路由编号，`[B]`；取值 `[0, W)` 为有效，越界行跳过。 | INT32 | ND |
| weightA | 可选输入 | LoRA A 权重堆叠，`[W, L, R, H1]`；缺省时走 BGMV 路径（要求 R == H1）。 | FLOAT16 | ND |
| layer_idx | 属性 | 选中的层编号，`[0, L)`。 | INT64 | - |
| scale | 属性 | lora 增量缩放系数，须为有限非 NaN 值。 | FLOAT | - |
| y_offset | 属性 | y 上列片段起点，`[0, H3)`。 | INT64 | - |
| y_slice_size | 属性 | 列片段长度；`-1` 表示取全宽 `H2`，否则 `[0, H2]`。 | INT64 | - |
| y_out | 输出 | in-place 输出，与 y 同地址，shape `[B, H3]`。 | FLOAT16 | ND |

## 约束说明

  - `B ∈ [1, 4096]`；`W, L ∈ [1, 32]`；`R/H1/H2/H3` 均为 16 的倍数且 `≤ 131072`。
  - weightA 在场时 `R ≤ 128` 且其 shape 须为 `[W, L, R, H1]`；缺省时 `R == H1`。
  - `y.dim0 == x.dim0 == indices.dim0 == B`。
  - `y_offset + H2 ≤ H3`；`y_slice_size ∈ {-1} ∪ [0, H2]`。
  - kernel 内部 workspace 段布局（x_g / t / lora / cnt_seg / segbase / rowpos / tbl）由 `op_host/add_lora_tiling.cpp` 按 SPEC 冻结数学填充，对调用方不感知。

## 文件清单

| 文件 | 说明 |
|---|---|
| `op_kernel/add_lora.py` | PyPTO-Pro kernel（单 `@pl.jit` MIX kernel；含 Stage-4 已验证的 D1–D15 机制偏离记载与 ops 接入适配 A1–A4 声明）。 |
| `op_graph/add_lora_proto.h` | GE 算子原型注册（REG_OP(AddLora)，weightA 为 OPTIONAL_INPUT）。 |
| `op_host/add_lora_def.cpp` | OpDef 注册（输入/输出/属性与 AICore ascend950 配置）。 |
| `op_host/add_lora_infershape.cpp` | InferShape（y_out = y 前 2 维 `[B, H3]`）与 InferDataType（DT_FLOAT16）注册。 |
| `op_host/add_lora_tiling.h` | CompileInfo 声明（ParamCheck / CheckShapeDims）；TilingData 结构体由 pypto_codegen 生成的 `AddLoraTiling_tiling.h` force-include 提供。 |
| `op_host/add_lora_tiling.cpp` | host tiling：blockDim（MIX 核）、TilingKey（addLoraFlag = weightA 在场）、TilingData 19 字段填充与 workspace 段偏移（镜像 Stage-4 wrapper 数学）。 |
| `op_host/config/ascend950/add_lora_binary.json` | ascend950 二进制算子配置（bin_filename 哈希为占位，构建时由 `ascendc_bin_param_build.py` 重算）。 |
| `op_host/config/ascend950/add_lora_simplified_key.ini` | opc `--simplified_key_mode` 配置（default=0）。 |

## 调用说明

本仓接入第一段仅交付算子目录结构（`bash build.sh --opkernel --soc=ascend950 --ops=add_lora` 可编译）；aclnn 调用样例在后续接入段补充。
