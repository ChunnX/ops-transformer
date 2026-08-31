# issue_20260829_mm-split-multihead_01：910B3 上多头（N≥2）query/query_rope 输出与 golden 不一致

## 现象
- mla_prolog_v5 在 910B3（20 AI Core）上，非量化 bf16 场景：
  - `query`（上采样输出）与 `query_rope` 的 **head 0 数值精确正确**（cos=1.0000），
    **head ≥ 1 全部错误**（N=2 时整体 cos≈0.51，N=16 时 cos≈0.069，即仅 1/N 正确）。
  - `query_norm`（cQ）、`kv_cache`（cKV）、do_rope=False 的 `kr_cache`、`cacheIndex`
    指定 slot 的 kv 值均与 CPU golden 精确一致（bf16 精度内，5/5 用例 PASS）。

## 排查过程（摘要）
1. C++ aclnn 直调与 pybind 路径现象一致 → 非 torch 适配层问题。
2. 单头（N=1）全链路 cos=1.0000 → 输入/NZ 权重读取/rmsnorm/mmCkvKr/rope 直通均正确。
3. tilingKey 与 binary 选择正常（SelectBin 成功、Launch numBlocks=20）。
4. 尝试将 tiling `FillMatmul3Tiling` 的 N 对齐粒度从 16 改为 (D+Dr)=192（避免 20 核下
   分块 160 跨头边界），现象无变化 → 问题不止 mm3 分块对齐（该适配已保留，注释说明）。
5. 源仓 v1 kernel/tiling 均为 vendored 原样迁移（未改动逻辑），怀疑为源算子在
   910B3（20 核，非 910B 的 24 核）上 mm3/mm4 多核分块协同的兼容性问题，
   需 kernel 级调试（dump/profiler）进一步定位。

## 已知限制（当前测试按分层验证规避）
- 多头 query/query_rope 不作为精度判据（test.py 标注 known-limit）。
- rope 开启时 kr_cache/query_rope 的存储布局与标准 interleaved 假设不同
  （pos=0 处语义已单点验证正确：out[0]=x[1]*cos0+x[0]*sin0），golden 公式待按
  kernel `service_rope_arch22.h` 的 GatherMask 奇偶拆分布局修正。
- PA 模式下未使用的 cache slot 也会被写入（值 0），"untouched slot" 校验不作为判据。

## 后续建议
1. 用 CANN kernel 调试工具（msdebug/simulator）dump mm3/mm4 中间结果，定位 20 核分块 bug。
2. 在 910B（24 核）环境复跑同一用例，验证"核数差异"假设。
3. 修正 golden 的 rope 布局假设后，将 kr/query_rope 纳入精度判据。

---

## ✅ 问题关闭（2026-08-31）：根因为测试 harness 权重布局构造错误，kernel 本身完全正确

### 最终结论
经内核级插桩（tiling 参数转储、mm3 workspace 区域转储、全核执行拓扑记录）逐层实证：

1. **kernel 与 tiling 均正确**：kernel 侧 tiling 参数（mm3SingleCoreN=192 等）与 host 一致；
   MatmulQcQr 以正确参数执行；mm3 输出布局严格跟随 weight_uq_qr 的列布局。
2. **真正根因**：`weight_uq_qr` 的算子契约是**按头交错**列布局 `[Qc_h(D) | Qr_h(Dr)] × N`，
   而测试 harness（golden.py/test.py）按数学直觉将其构造为分离布局 `[Wuq | Wqr]`。
   mm3 输出因此呈分离布局，而消费者（mm4/RopeQr）按交错布局读取 → head≥1 输出错位。
   "head 0 的 query 正确 + 最后一个 head 的 query_rope 正确"正是两种布局下重叠位置的精确特征。
3. **"未使用 slot 被写入"为误报**：-12345 在 bf16 下舍入为 -12352，`== -12345` 判定必然失败。
4. **rope 输出布局语义**（单位探针实证标定）：
   `out[j<Dr/2] = x[2j]*cos[j] - x[2j+1]*sin[j]`，`out[Dr/2+j] = x[2j]*sin[j] + x[2j+1]*cos[j]`
   （标准 RoPE 旋转对，输出为 [前半分量|后半分量] 布局；sin/cos 前后半重复）。

### 修复（仅测试侧，kernel/tiling 保持源仓原样）
- golden.py / test.py：weight_uq_qr 按头交错构造；golden 按交错布局提取 Qc/Qr；
  rope 公式按实证语义修正；cache 初始标记改用 bf16 可精确表示值。
- 排查过程中引入的全部 kernel/tiling 调试插桩已移除，vendored 代码还原为源仓原版。

### 验证
tests/pybind/test.py 全量 5/5 PASS（query / query_rope / query_norm / kv_cache / kr_cache
全部输出、全部用例：TND、PA_BSND、3 维输入、do_rope 开关、scale 校正）。
