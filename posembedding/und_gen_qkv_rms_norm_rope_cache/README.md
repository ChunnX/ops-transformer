# UndGenQkvRmsNormRopeCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- 算子功能：多模态大模型推理 prefill/decode 阶段的 Q/K/V 主干融合算子。把 undecoded 段（und_qkv）与
  generated 段（gen_qkv）按 `cat_indices` 间接寻址拼成输出序列，逐 token 完成：

  1. **Split**：按 N 维把每个 token 拆成 Q[Hq, D] / K[Hk, D] / V[Hv, D]；
  2. **RMSNorm**：Q/K 分别用 und/gen 两套权重（按 `src_t < und_len` 选择）做归一化；
  3. **MRoPE**：按 `positions[3, T]` 取三轴（时间/高度/宽度）cos_sin，依 `mrope_section` 合并成一份后
     对 Q/K 做标准 RoPE；V 不参与 norm/rope；
  4. **输出**：Q 作为独立输出；K/V 经 Cast float32→bf16 后按 `slot_mapping` 写入分页 KV Cache。

- 计算全程 float32，KV Cache 以 bf16 存储（FP8/INT8 量化不在本期范围）。

- 输入输出支持以下数据场景：

    ```
    und_qkv:       [und_len, N, D]           N = Hq + Hk + Hv
    gen_qkv:       [gen_len, N, D]           当前必填，gen_len > 0
    und_weights_q: [D]   und_weights_k: [D]
    gen_weights_q: [D]   gen_weights_k: [D]  可选，与 gen_qkv 同进同出
    cos_sin_cache: [max_pos, D]              前半 cos、后半 sin
    slot_mapping:  [T]                       slot = block_num * Bs + row_idx
    positions:     [3, T]
    cat_indices:   [T]                       当前必填（不传的恒等映射路径暂不支持）
    k_cache:       [Bn, Bs, Hk, D]           调用方预分配，原地写入
    v_cache:       [Bn, Bs, Hv, D]           调用方预分配，原地写入
    q:             [T, Hq, D]                T = und_len + gen_len
    ```

- **KV Cache 布局固定为连续 BBND**（`[Bn, Bs, N, D]`），不支持非连续/BNBD 物理布局。
- 索引类张量（`slot_mapping` / `positions` / `cat_indices`）统一 int64。
- `mrope_section` 为空列表时退化为 `[D/2, 0, 0]`，此时等价标准 RoPE。

## 参数说明

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 说明 |
| :----- | :-------- | :------- | :------- | :--- |
| und_qkv | 输入 | BFLOAT16 | ND | undecoded 段 QKV，[und_len, N, D] |
| und_weights_q | 输入 | BFLOAT16 | ND | und 段 Q 的 RMSNorm 权重，[D] |
| und_weights_k | 输入 | BFLOAT16 | ND | und 段 K 的 RMSNorm 权重，[D] |
| cos_sin_cache | 输入 | FLOAT32 | ND | cos/sin 缓存，[max_pos, D] |
| k_cache | 输入/输出 | BFLOAT16 | ND | K Cache，[Bn, Bs, Hk, D]，原地写入 |
| v_cache | 输入/输出 | BFLOAT16 | ND | V Cache，[Bn, Bs, Hv, D]，原地写入 |
| slot_mapping | 输入 | INT64 | ND | 每个 token 的 cache 行偏移，[T] |
| positions | 输入 | INT64 | ND | MRoPE 三轴位置，[3, T] |
| gen_qkv | 可选输入（当前必填） | BFLOAT16 | ND | generated 段 QKV，[gen_len, N, D]，gen_len > 0 |
| gen_weights_q | 可选输入（当前必填） | BFLOAT16 | ND | gen 段 Q 的 RMSNorm 权重，[D] |
| gen_weights_k | 可选输入（当前必填） | BFLOAT16 | ND | gen 段 K 的 RMSNorm 权重，[D] |
| cat_indices | 可选输入（当前必填） | INT64 | ND | out_t → src_t 映射，[T] |
| q | 输出 | BFLOAT16 | ND | 处理后的 Q，[T, Hq, D] |
| num_heads_q | 属性 | INT | - | Q 头数 Hq |
| num_heads_k | 属性 | INT | - | K 头数 Hk |
| num_heads_v | 属性 | INT | - | V 头数 Hv |
| norm_eps | 属性 | FLOAT | - | RMSNorm epsilon，默认 1e-6 |
| mrope_section | 属性 | LIST_INT | - | MRoPE 三轴分段，默认 [] |

## 约束说明

- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>（DAV_3510, arch35）。
- headDim 固定为 128。
- (Hq, Hk, Hv) 支持 (8, 1, 1) 与 (16, 2, 2) 两种组合。
- cache block_size 为 128；T = und_len + gen_len 支持 1 ~ 64K。
- `slot_mapping` 的取值必须唯一且落在 [0, Bn*Bs)，重复 slot 在多核下结果不确定。
- **当前实现要求 `gen_qkv`、`gen_weights_q`、`gen_weights_k`、`cat_indices` 四个可选输入全部提供，
  且 `gen_len > 0`**；缺省的退化路径（纯 prefill、单序列恒等映射）暂不支持，tiling 会直接拦截。
  IR/OpDef 保留 OPTIONAL 声明，后续放开时删掉 `CheckSupportRange` 中对应校验即可。
- `k_cache`/`v_cache` 必须内存连续。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| aclnn 接口 | [test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp](./examples/test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp) | 两段式接口调用样例，含 slot_mapping 预计算 |
| 接口文档 | [aclnnUndGenQkvRmsNormRopeCache.md](./docs/aclnnUndGenQkvRmsNormRopeCache.md) | L2 接口说明 |

## 参数校验

校验统一收敛在 **tiling** 一层（`op_host/*_base_tiling.cpp`），aclnn L2 与 InferShape 只做各自必需的最小校验，
避免同一条约束在三处重复实现、后续改约束时漂移。

| 层 | 校验内容 |
| :-- | :------- |
| aclnn L2 | 仅空指针类：必选张量非空；`genQkv` 提供时 `genWeightsQ`/`genWeightsK` 必须成对提供 |
| InferShape / InferDataType | 仅推导前提：`und_qkv`/`gen_qkv` 为 3D、属性 `num_heads_q` 可取；dtype 透传不校验 |
| **Tiling** | `CheckDtypeValid`（12 输入 + 3 输出逐个查）、`CheckAttrsValid`（头数正数与求和==N、eps>0、mrope_section 长度与求和）、7 组 shape 校验（und/gen 的 N/D 一致、weights 长度、cos_sin_cache、k/v_cache 的 Bn/Bs/Hk/Hv/D 与容量、slot_mapping / positions / cat_indices 长度）、`CheckSupportRange`（headDim=128、block_size=128、T ∈ [1, 64K]、head 组合白名单、4 个可选输入必须全部提供） |

> aclnn 单算子路径和图模式最终都会走 tiling，所以 tiling 侧的校验对两条路径都生效；
> 代价是非法入参在 `aclnnXxxGetWorkspaceSize` 阶段不报错，而是在执行阶段由 tiling 返回失败。

> 数据值域相关的约束（`slot_mapping` 取值唯一性、`positions < max_pos`、`cat_indices` 值域）
> 属于运行期数据，Host 侧看不到 tensor 内容，无法校验；由调用方保证。

## 开发状态

| 阶段 | 状态 |
| :--- | :--- |
| OpDef / REG_OP / InferShape / InferDataType | 已完成 |
| Tiling 参数解析与校验 | 已完成 |
| Tiling 多核 / UB 切分 | **待实现**（见 `op_host/*_tiling.cpp` 的 TODO） |
| Kernel 计算实现 | **待实现**（见 `op_kernel/arch35/*_regbase.h` 的 TODO） |
| aclnn L0/L2 接口 | 已完成 |
| UT（infershape / tiling / kernel 拉起） | 已完成 |
| ST（ATK） | golden 已就绪，待 kernel 实现后跑通 |
