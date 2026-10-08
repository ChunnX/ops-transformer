# compressor_grad — A3 (Ascend910_93x / arch22) AscendC 实现

本目录是 `attention/compressor_grad`（Compressor 反向算子）的 **A3/AscendC 移植版**。参考实现为同目录的 A5（Ascend950 / arch35）PyPTO 版本；A5 版依赖 arch35 专有特性（UB↔L1 CV 直通、L0C 256KB、MIX section 等），无法在 A3 直接运行，因此按 A3（DAV_2201）架构重新实现。

## 形态说明

- 独立可构建的 AscendC torch-extension 工程（`setup.py` → whl → `import compressor_grad` 注册 `torch.ops.npu.compressor_grad`），自带 host 编排与注册链路。
- **尚未接入仓库原生 op 构建体系**（op_host def/infershape/tiling + aclnn 注册），以纯新增目录形式提交、不影响任何既有文件；接入原生体系所需的语义/tiling 依据见下文与 README 末尾的验证资产。

## 算子语义

与 A5 版一致：输入 x(B,S,H)、wkv/wgate(C,H)（C=coff*D）、d_cmp_kv(B,⌈S/cr⌉,D)、softmax_score/kv(B,⌈S/cr⌉,coff*cr,D) fp32；attr cmp_ratio∈[2,128]、coff∈{1,2}；仅处理 S//cr 个完整块（尾块跳过）。输出 d_x(B,S,H)、d_wkv/d_wgate(C,H)、d_ape(cr,C) fp32。

计算：① Hadamard 反向 ② 块内 softmax 反向（fp32）③ 梯度路由写 d_new_kv/d_new_score（cur 槽 + coff=2 的 prev 跨块槽）④ d_ape 跨块累加（fp32 原子加）⑤ 双 GEMM 反向（d_x = dn_kv@wkv + dn_sc@wgate；d_w = dnᵀ@x）。

## A3 架构适配要点（相对 A5）

- CV 无直通 → 3-kernel 经 GM workspace 交换：**cast**（x 直写 blocked-NZ 布局 + dn 清零）→ **scatter**（Hadamard+softmax 反向+路由+d_ape）→ **gemm**（dx/dw 双 GEMM，MIX_AIC_1_2）
- L0C 128KB/UB 192KB 容量适配；转置不由独立 kernel 完成，GEMM 经 `LoadDataWithTranspose`（A 侧）与 `load3dv2+SetFmatrix`（B 侧自然布局）在加载时内联完成
- fp16 输入走 **HiLo 双平面补偿**（dn=hi+lo 双 fp16 平面、双 Mmad 同 L0C 累加），保证 fp16 官方容差（rel_thr 2^-10）下 matched_ratio≥0.999

## 性能（Ascend910_9382，CANN 9.1.0，msprof device 侧口径，对比 PyTorch 标杆）

| Case | Shape | dtype | 标杆(µs) | 本实现(µs) | Speedup |
|---|---|---|---|---|---|
| 1 | [2,1024,2048] D=128 cr=32 coff=1 | bf16 | 246 | 68 | 3.60x |
| 2 | [1,4096,4096] D=128 cr=64 coff=1 | bf16 | 655 | 185 | 3.55x |
| 3 | [4,2048,5120] D=128 cr=16 coff=2 | bf16 | 2268 | 696 | 3.26x |
| 4 | [1,8192,4096] D=512 cr=128 coff=1 | bf16 | 2597 | 878 | 2.96x |
| 5 | [2,2050,1024] cr=8（尾块） | fp16 | 337 | 102 | 3.31x |
| 6 | [1,32768,7168] D=512 cr=64（最大） | bf16 | 17136 | 7000 | 2.45x |

**geomean ≈ 3.16x**（6/6 反超标杆；R7 固化态 whl 官方口径实测）。后续优化（dx prefetch + 动态分核重标定，A/B 实测 gemm 再 -14~-31%）已落地待复测更新。

## 正确性验证

- 官方 evaluate（6 个 fuzz 用例）：**6/6 PASS**（matched_ratio≥0.9 判定口径）
- 扩展矩阵 52 项（coff=2 跨块路由边界、尾块、D=512×coff=2、fp16×T_pad>65535、确定性、异常输入拒绝）全 PASS
- fuzz 50 例（官方分布内随机 shape）全 PASS
- 修复过程的关键 bug 与精度红线（fp16 量化噪声、repeatTime uint8 截断、Nd2Nz 位宽等）均在开发记录中留档

## 构建与使用

```bash
source <CANN>/set_env.sh            # 需含 ascend910_93 算子库
export SOC_VERSION=ascend910_93
cd arch22_ascendc
python3 setup.py bdist_wheel && pip install --force-reinstall dist/*.whl
```

```python
import torch, torch_npu, compressor_grad  # 注册 torch.ops.npu.compressor_grad
d_x, d_wkv, d_wgate, d_ape = torch.ops.npu.compressor_grad(
    x, wkv, wgate, d_cmp_kv, softmax_score, kv, cmp_ratio, coff)
```

输入约束（host 侧有 TORCH_CHECK 防护）：D%64==0、H%128==0、cr≤255、S≥cr（至少一个完整块）、x/wkv/wgate/d_cmp_kv 同 dtype（fp16/bf16）、softmax_score/kv 为 fp32。

## 已知限制

- 小 Shape（S<cr）按"无完整块"拒绝（标杆语义为全零输出，差异已文档化）
- d_ape 为跨核原子加，非 bitwise 确定（|Δ|~1e-6，远低于容差）
- fp16 路径因 HiLo 补偿 GEMM 计算量翻倍（精度达标的必要代价）
