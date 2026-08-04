# ApplyRotaryPosEmbGrad

## 算子功能

双路反向旋转位置编码（ApplyRotaryPosEmb 的反向）。同时计算 **query** 和 **key** 的 rope 反向梯度，融合为一次 kernel 调用，节省 cos/sin 加载和 kernel launch 开销。

## 数学公式（half mode）

对于 Q 路和 K 路分别计算：

```
# Q 路
grad_query = cat(cos1 * grad_q_embed_1 + sin2 * grad_q_embed_2, cos2 * grad_q_embed_2 - sin1 * grad_q_embed_1, dim=-1)
grad_cos += sum(grad_query_embed * query, [reduce_dims])
grad_sin += sum(grad_query_embed * cat(-query_2, query_1, dim=-1), [reduce_dims])

# K 路
grad_key = cat(cos1 * grad_k_embed_1 + sin2 * grad_k_embed_2, cos2 * grad_k_embed_2 - sin1 * grad_k_embed_1, dim=-1)
grad_cos += sum(grad_key_embed * key, [reduce_dims])
grad_sin += sum(grad_key_embed * cat(-key_2, key_1, dim=-1), [reduce_dims])
```

## 支持情况

| 产品 | 支持 |
|:--|:--|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3/A2 系列 | ×（不涉及 membase） |

## 输入/输出

### 输入
| 参数 | 类型 | 说明 |
|:--|:--|:--|
| grad_query_embed | REQUIRED | Q 路输出端上游梯度 |
| grad_key_embed | REQUIRED | K 路输出端上游梯度 |
| cos | REQUIRED | 正向 cos 输入 |
| sin | REQUIRED | 正向 sin 输入 |
| query | OPTIONAL | 正向 Q 输入（非空则计算 grad_cos/grad_sin） |
| key | OPTIONAL | 正向 K 输入（非空则计算 grad_cos/grad_sin） |

### 输出
| 参数 | 说明 |
|:--|:--|
| grad_query | Q 路梯度 |
| grad_key | K 路梯度 |
| grad_cos | cos 梯度（query/key 非空时有效） |
| grad_sin | sin 梯度（query/key 非空时有效） |

### 属性
| 属性 | 类型 | 默认值 | 说明 |
|:--|:--|:--|:--|
| rotary_mode | string | "half" | 旋转模式。仅支持 "half"。 |
| layout | int | 1 | layout 类型。1=BSND, 2=SBND, 3=BNSD(预留), 4=TND。 |

## Layout 支持

OpDef attr layout: {1=BSND, 2=SBND, 3=BNSD, 4=TND} → internal Layout {0=BSND, 1=SBND, 2=BNSD}

| Attr Layout | dy shape | cos shape | Internal Layout | DxTilingKey | 状态 |
|:--|:--|:--|:--|:--|:--|
| 1=BSND, B=1 | (1,S,N,D) | (1,S,1,D) | 0=BSND | 203=BAB | √ |
| 1=BSND, B>1 | (B,S,N,D) | (B,S,1,D) 或 (1,S,1,D) | 1=SBND | 204=AB | √ |
| 2=SBND | (S,B,N,D) | (S,B,1,D) 或 (S,1,1,D) | 1=SBND | 204=AB | √ |
| 3=BNSD | (B,N,S,D) | (B,1,S,D) 或 (1,1,S,D) | 2=BNSD | — | 预留 |
| 4=TND | (T,N,D) | (T,1,D) | 0=BSND | 203=BAB | √ |

> 约束: cos N 维必须 = 1; cos B ∈ {1, grad_query_embed.B}; cos S/D 与 grad_query_embed 一致。
> TND 经维度映射 (B=1, S=T, N=N) 后等价于 BSND → BAB。

## 约束

- 仅 Ascend 950 (regbase) 支持
- 仅 half 模式 (rotary_mode="half")
- D <= 1024, D 必须能被 2 整除
- cos N 必须 = 1; cos B ∈ {1, grad_query_embed.B}
- query 和 key 必须同时非空或同时为空
- GQA 支持: nQ >= nK 或 nK >= nQ 均可

## Tiling Key 编码

四维 tiling key: `REDUCE_TPL_KEY + DxTilingKey(8bit) + DcosFlag(1bit) + Layout(2bit)`

| DxTilingKey | 值 | Split 模式 |
|:--|:--|:--|
| KEY_BAB | 203 | BSND 分核 |
| KEY_AB  | 204 | SBND 分核 |
| KEY_A   | 205 | NO_BROADCAST 分核 |

| Layout | 值 | 说明 |
|:--|:--|:--|
| BSND | 0 | cos (1, S, 1, D) + TND 映射 — BAB |
| SBND | 1 | cos (B, S, 1, D) B>1 — AB |
| BNSD | 2 | 预留，不实例化 |

## 文件结构

```
apply_rotary_pos_emb_grad/
├── CMakeLists.txt
├── README.md
├── examples/
│   └── test_aclnn_apply_rotary_pos_emb_grad.cpp
├── op_graph/
│   ├── CMakeLists.txt
│   ├── apply_rotary_pos_emb_grad_proto.h
│   └── fusion_pass/.gitkeep
├── op_host/
│   ├── CMakeLists.txt
│   ├── apply_rotary_pos_emb_grad_def.cpp
│   ├── apply_rotary_pos_emb_grad_infershape.cpp
│   ├── apply_rotary_pos_emb_grad_tiling.h
│   ├── apply_rotary_pos_emb_grad_tiling.cpp
│   ├── apply_rotary_pos_emb_grad_tiling_bab.cpp
│   ├── apply_rotary_pos_emb_grad_tiling_ab.cpp
│   ├── apply_rotary_pos_emb_grad_tiling_a_and_b.cpp
│   └── config/ascend950/
│       ├── apply_rotary_pos_emb_grad_binary.json
│       └── apply_rotary_pos_emb_grad_simplified_key.ini
└── op_kernel/
    ├── apply_rotary_pos_emb_grad_apt.cpp
    └── arch35/
        ├── apply_rotary_pos_emb_grad_tiling_key.h
        ├── apply_rotary_pos_emb_grad_tiling_data.h
        ├── apply_rotary_pos_emb_grad_common.h
        ├── apply_rotary_pos_emb_grad_dag.h
        ├── apply_rotary_pos_emb_grad_rotary_x_vf.h
        ├── apply_rotary_pos_emb_grad_bab.h
        ├── apply_rotary_pos_emb_grad_ab.h
        ├── apply_rotary_pos_emb_grad_a_and_b.h
        └── apply_rotary_pos_emb_grad_rotary_x.h
```
