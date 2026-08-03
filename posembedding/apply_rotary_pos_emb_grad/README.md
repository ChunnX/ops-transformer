# ApplyRotaryPosEmbGrad

## 算子功能

双路反向旋转位置编码（ApplyRotaryPosEmb 的反向）。同时计算 **query** 和 **key** 的 rope 反向梯度，融合为一次 kernel 调用，节省 cos/sin 加载和 kernel launch 开销。

## 数学公式（half mode, mode=0）

对于 Q 路和 K 路分别计算：

```
# Q 路
dx_q = cat(cos1 * dy_q1 + sin2 * dy_q2, cos2 * dy_q2 - sin1 * dy_q1, dim=-1)
dcos += sum(dy_q * x_q, [reduce_dims])
dsin += sum(dy_q * cat(-x_q2, x_q1, dim=-1), [reduce_dims])

# K 路
dx_k = cat(cos1 * dy_k1 + sin2 * dy_k2, cos2 * dy_k2 - sin1 * dy_k1, dim=-1)
dcos += sum(dy_k * x_k, [reduce_dims])
dsin += sum(dy_k * cat(-x_k2, x_k1, dim=-1), [reduce_dims])
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
| dy_q | REQUIRED | Q 路输出端上游梯度 |
| dy_k | REQUIRED | K 路输出端上游梯度 |
| cos | REQUIRED | 正向 cos 输入 |
| sin | REQUIRED | 正向 sin 输入 |
| x_q | OPTIONAL | 正向 Q 输入（非空则计算 dcos/dsin） |
| x_k | OPTIONAL | 正向 K 输入（非空则计算 dcos/dsin） |

### 输出
| 参数 | 说明 |
|:--|:--|
| dx_q | Q 路梯度 |
| dx_k | K 路梯度 |
| dcos | cos 梯度（x_q/x_k 非空时有效） |
| dsin | sin 梯度（x_q/x_k 非空时有效） |

### 属性
| 属性 | 类型 | 默认值 | 说明 |
|:--|:--|:--|:--|
| mode | int | 0 | 旋转模式。仅支持 0 (half)。 |

## Layout 支持

| Layout | tiling key | 说明 | 状态 |
|:--|:--|:--|:--|
| BSND (1S1D) | BAB (203) | cos shape (1, S, 1, D) | √ |
| SBND (S11D/SB1D) | AB (204) | cos shape (S, 1, 1, D) / (S, B, 1, D) | √ |
| BNSD (11SD/B1SD) | ABA/BA (201/202) | cos shape (1, 1, S, D) / (B, 1, S, D) | 预留 |
| NO_BROADCAST + BROADCAST_BSN | A/B (205/206) | A(205): cos == dy 全同; B(206): cos = 111D 全广播 | √ |

> TND (3D) 不是独立 layout，经维度映射（B=1, S=T, N=N）后归类为上述 4 种之一。例如 TND+cos=T1D → BSND (BAB)，TND+cos=TND → NO_BROADCAST (A)，TND+cos=1ND → BROADCAST_BSN (B)。

## 约束

- 仅 Ascend 950 (regbase) 支持
- 仅 half 模式 (mode=0)
- D <= 1024, D 必须能被 2 整除
- x_q 和 x_k 必须同时非空或同时为空
- GQA 支持: nQ >= nK 或 nK >= nQ 均可

## Tiling Key 编码

四维 tiling key: `REDUCE_TPL_KEY + DxTilingKey(8bit) + DcosFlag(1bit) + Layout(2bit)`

| DxTilingKey | 值 | Split 模式 |
|:--|:--|:--|
| KEY_ABA | 201 | BNSD B 广播 |
| KEY_BA | 202 | BNSD B 不广播 |
| KEY_BAB | 203 | BSND 分核 |
| KEY_AB | 204 | SBND 分核 |
| KEY_A | 205 | TND 不广播 |
| KEY_B | 206 | TND 广播 |

| Layout | 值 | 说明 |
|:--|:--|:--|
| BSND | 0 | cos (1, S, 1, D) — BAB |
| SBND | 1 | cos (S, 1, 1, D) 等 — AB |
| BNSD | 2 | cos (1, 1, S, D) 等 — ABA/BA，预留 |
| NO_BROADCAST + BROADCAST_BSN | 3 | 全同/全广播 — A/B |

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
│   ├── apply_rotary_pos_emb_grad_tiling_aba_and_ba.cpp
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
        ├── apply_rotary_pos_emb_grad_aba_and_ba.h
        ├── apply_rotary_pos_emb_grad_bab.h
        ├── apply_rotary_pos_emb_grad_ab.h
        ├── apply_rotary_pos_emb_grad_a_and_b.h
        └── apply_rotary_pos_emb_grad_rotary_x.h
```
