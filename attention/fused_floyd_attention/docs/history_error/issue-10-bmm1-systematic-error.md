# Bmm1 系统性数值误差（当前主攻，根因未定）

## 现象

```
Output 0 (softmaxMax): Cosine=0.965, MaxAbsErr≈59 (range[14,74])
Output 1 (softmaxSum): Cosine=0.918
Output 2 (attention):  Cosine≈0.0005
```

softmaxMax(Output0) 只依赖 Bmm1，cosine 仅 0.965、误差高达 59 → **Bmm1 本身就不对**，
这是 Output2 错误的上游根因。

## 关键诊断

- **误差在所有 slice 上完全均匀**（14 个 b-slice cosine 均 ≈0.965）→ 排除分核/AIV 分配 bug，是**系统性 matmul 数值误差**。
- softmaxMax ratio(NPU/CPU) 分布 0.175~2.659，std 0.199 → **不是固定 scale，是 Q·K^T 每个点积值都错**（NPU 一致偏小，累加贡献不完整）。

## 已排除项

1. **transpose 方向**：独立核对 GQA ComputeMm1/Mm2 的 LoadData transpose——
   FFA adapter `ifTranspose = !transB_`：bmm1 transB_=true→ifTranspose=false ✓ 与 GQA Mm1 的 K 一致；bmm2 transB_=false→ifTranspose=true ✓ 与 GQA Mm2 的 V 一致。**transpose 逻辑与 GQA 一致，不是方向问题。**
2. **GM 读取参数**：adapter 的 srcDValue 与 A3 库 `SetOrgShape(orgKa/orgKb=d=64)` 完全一致 → GM 读取层无错。“改 stride 没效果”印证了这一点。
3. **换用库 matmul 后误差形态几乎完全一致**（softmaxMax 0.965x 的“魔法数字”重现，Output2 cos 0.0005x）→ 说明 adapter 的 stride/batch 细节不是根因，**根因在库和 adapter 的共同上游**。

## FFA 双 key/value 路径（复杂点）

case 61 是真正的 Floyd 双 key/value：

```
query        bf16 [2,7,32,128,64]
key1/value1  bf16 [2,7,32,128,64]   ← gS1=32
key2/value2  bf16 [2,7,128,128,64]  ← S2=128
```

- `IterateBmm1k1`(key1)：写 `mm1Res[bmmv2Num*idx*s2Real]`（[s1,s2] 内行偏移，每次 bmmv2Num 行）
- `IterateBmm1`(key2)：写 `mm1Res[batchSizes*s2Real*s1Real]`（n2 batch 维）
- `GetBmm1Result`（line 1252-1255）按 `[s1RealSize, s2RealSize]` 行主序读。
- 两路结果交织进同一个 `[s1,s2]` mm1Res，布局本身设计如此，但合并语义需仔细核对。

## 怀疑方向

- adapter 单次 matmul 的 stride 数值细节（已多次排查，效果有限）；
- 双 key 路径写 mm1Res 的 offset 冲突/合并语义；
- host tiling 算出的 M/N/K/orgShape 是否符合 case 61 的 BNSD 语义；
- A3 原版逻辑在 A5(AIV-only) 上的适配性（如 intra-core flag）。

## 方向转变：改用库 matmul + KFC

**决策**：库 `matmul::Matmul` 在 A5 可用（靠 `REGIST_MATMUL_OBJ`→`REGIST_CUBE_OBJ`：AIC 进 KfcServer 当 server，AIV 当 client，`bmm.IterateBatch` 经 KFC 消息驱动 AIC）。改用库可一次性绕开手写 mini-matmul 的 stride 坑。

**验证结果**：
- ✅ 编译通过、KFC 分核正常、无 timeout/挂死；
- ✅ tiling 未走 fillBasicTiling 兜底（`MatmulApiTiling::GetTiling()` 对 case 61 成功）；
- ❌ 精度未改善：Output0 cos=0.965425(vs adapter 0.965123)、Output2 cos=0.000525(vs 0.000549)。
- → **换库后误差形态与手写 adapter 几乎一致**，坐实根因在共同上游。

## 卡点：拿不到 kernel 内部实测数据

- `DumpTensor` 和 `printf` 在 A5 MIX / ATK 框架下均无输出（详见 [issue-11](issue-11-dump-not-available.md)）。
- 无法 dump Q/K/mm1Res 手算对比，难以直接定位传参是否正确。

## 下一步（待定，倾向 B）

- **A（成本高）**：msopst 单算子直调，DumpTensor 应能输出，拿实测数据。
- **B（成本低，推荐）**：在 A3(910B)上跑 case 61 原版——若 A3 也是 0.965x → 算子/case 本身有 bug；若 A3 精度好 → 确认是 A5 适配问题。**最快排除/坐实“A5 特有问题”。**
- **C（成本中）**：host tiling 代码审查（`SetBmm1TilingInput`/`CaclBmmBatch` 算出的 M/N/K/orgShape 是否符合 BNSD 语义），纯 host 侧，不需 kernel dump。

> A3 测试包已备好：`/home/j00946653/830/a3_test_package`（run_test_a3.sh + case json + executor + 操作步骤）。编译命令 `bash build.sh --pkg --ops=fused_floyd_attention --soc=ascend910b -j16`。首次编译遇 `no member named 'kCoreOffset'` 已修（删除误用 DumpTensor 的 debug 代码块）。
