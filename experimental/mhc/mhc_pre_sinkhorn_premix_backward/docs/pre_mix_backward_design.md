# mhc_pre_sinkhorn_premix_backward 新增 pre_mix 适配设计（A2/arch22）

> 状态：公式已通过 torch golden 验证（6/6 ALL PASS），kernel 改动待实施。
> 约束：只改 A2A3（arch22），不动 A5（arch35）。

## 1. 背景

正向 `mhc_pre` 新增可选输入 `pre_mix` 与可选输出 `pre`：

- 传入 `pre_mix` 时，`y` 不再使用 `pre` 加权，而是使用 `pre_mix` 加权：
  `y = Σ_hc mix_for_y · x`，其中 `mix_for_y = pre_mix（若传入）否则 pre`
- `pre` 输出非 nullptr 时，把 `pre` 结果搬出，供下游（下一处 mhc_pre 的 pre_mix）消费

反向算子需同步适配：识别 `pre_mix` 前向分支，输出 `grad_pre_mix`，接受可选输入 `grad_pre`（下游消费 pre 回传的梯度）。

## 2. 反向公式（已验证）

参考实现：`hc_model/model.py` + `hc_model/kernel.py`。
验证脚本：`test/golden_hc_pre_backward.py`（autograd vs 手写解析，6 case：2 shapes × {v1, v2, v2b}，纯 CPU fp32，全 PASS）。

设 `d_y` 为 y 的梯度、`d_pre_out` 为下游回传的 pre 梯度：

**分支 A：y 折叠反传**

```
d_mix_for_y = einsum('bsd,bshd->bsh', d_y, x)     # Σ_c d_y·x
d_x        += mix_for_y[..., None] · d_y[..., None, :]
```

**分支 B：pre 链（合并点）**

```
d_pre = (has_pre_mix ? 0 : d_mix_for_y) + d_pre_out
d_z   = d_pre · σ(z)·(1-σ(z))          # z 由 h_hat2/scale/base 重构
alphaPreGrad = Σ d_z · pre_raw          # = grad_hc_scale[0]（pre 段）
biasPreGrad  = Σ d_z                    # = grad_hc_base[:n]
```

**分支 C：post 链**：同 pre，系数 2（现有逻辑不变）。

**分支 D：comb 链（sinkhorn 反传）**：输入只有 `grad_comb`，与 pre 无关，不改动。

**分支 E：matmul + RMSNorm**：`d_hc_fn`、`d_xf` 的 RMSNorm 修正项数据源为合并后的梯度，链路自动更新，不单独改。

### 关键场景结论（golden 已确认）

| 场景 | alphaPreGrad/biasPreGrad | 说明 |
|---|---|---|
| v1（无 pre_mix） | 非零 | y→pre 边存在 |
| v2（有 pre_mix，pre 有下游） | **仍非零** | 来自 d_pre_out，y 旁路被切断 |
| v2b（有 pre_mix，pre 无下游） | **恒为 0** | pre 完全无梯度来源 |

## 3. kernel 现有数据流（arch22/mhc_pre_grad_kernel.h，v1 版）

```
ComputeGradPre:    d_mix_for_y = Σ_c(d_y·x)  ──Kahan──▶ dPrePostTempLocal_（pre/post 交错槽）
                   同时落盘 x·d_y → xWorkspace（供 mm2 算 d_hc_fn）
ComputeGradHHat2:  Axpy(dPrePost, grad_post, 2)          ◀ 合并点：d_pre = d_mix + 2·grad_post
                   → sigmoid 反传 → dhat → kahan 累加 → grad_hc_scale[0] / grad_hc_base[:n]
                   → dhat2 落盘 → mm1（grad_x 的 matmul 项）/ mm2（d_hc_fn）
ComputeGradX1:     grad_x = mm1结果 + grad_y·pre + RMSNorm 修正    ◀ y 折叠项用 pre
ProcessMatmul1/2:  AIC 侧 mm1_/mm2_
```

## 4. 改动点

### ① ComputeGradX1：y 折叠项换加权源

- 位置：`ComputeGradX1`（`Mul(gradXCastLocal_, gradYCastLocal_, preBrcbLocal_,...)`），以及 `Process` 中搬入 `pre` 并 Brcb 的位置
- 改法：
  - 新增 `preMixGm` buffer 搬入 + Brcb → `preMixBrcbLocal_`
  - y 折叠项加权源改为 `hasPreMix ? preMixBrcbLocal_ : preBrcbLocal_`
- 说明：sigmoid 反传不使用 pre 值（由 h_hat2/scale/base 重构 z），不受影响
- 对应验证项：`d_x`

### ② ComputeGradHHat2：d_pre 条件合并 + 新增 grad_pre_mix 落盘

- 位置：`ComputeGradHHat2` 的 `Axpy(dPrePostTempLocal_, gradHPostLocal_, 2, ...)` 处
- 改法（三态合并）：
  - 无 pre_mix：`d_pre = d_mix_for_y + d_pre_in`（现状 + 可选下游梯度）
  - 有 pre_mix：Axpy 前将 `dPrePostTempLocal_` 的 pre 段槽清零，`d_pre = 0 + d_pre_in`
  - 在 Axpy **之前**，把 `dPrePostTempLocal_` 的 pre 段按槽位抽取拷出 → 新可选输出 `grad_pre_mix`（= d_mix_for_y，fp32 [bs,n] 连续）
- 注意：`dPrePostTempLocal_` 为 pre/post 交错布局（stride 2 槽），落盘时需按槽抽取
- 对应验证项：`d_pre(merged)`、`d_pre_mix`、`alphaPreGrad`/`biasPreGrad`（累加逻辑不动，自动跟随合并点）

### ③ 接口 / tiling 侧

| 位置 | 改动 |
|---|---|
| kernel Init / apt.cpp | GM_ADDR 增加：可选输入 `grad_pre`、可选输出 `grad_pre_mix` |
| tiling（arch22 data/key 头文件） | 新增 flag：`hasPreMix`、`hasGradPre`（指针判空置位，同正向 hasPreMix/hasPreOut 模式） |
| op_host def / shape 推导 | 注册新增可选输入输出（shape 同 pre 段 [bs,n] fp32） |
| aclnn / torch extension | 接口与透传 |

## 5. 不需要改动的部分

| 模块 | 原因 |
|---|---|
| SinkhornGrad（comb 链） | 输入只有 grad_comb，与 pre 无关 |
| RMSNorm 修正项 | 数据源为合并后梯度，链路自动更新 |
| mm1 / mm2 | 数据源是 dhat2 / xWorkspace，自动正确 |
| grad_hc_scale / grad_hc_base 累加（kahan、ComputeScaleBias） | 由 ② 的合并点自动保证正确 |

## 6. 验证对照表

| golden 梯度项 | 覆盖改动 |
|---|---|
| d_x | ① |
| d_pre_mix | ②（落盘） |
| d_pre(merged) | ②（合并） |
| d_hc_scale / d_hc_base（含 alphaPreGrad/biasPreGrad） | ② 联动（自动） |
| d_hc_fn | 无改动，链路自动 |
| comb/sinkhorn 链 | 无改动 |

后续上板验证计划：v1/v2/v2b 三场景 + `grad_pre_mix` 与 `grad_pre` 数值比对（golden 脚本同款容差 atol=1e-5, rtol=2e-3）。
