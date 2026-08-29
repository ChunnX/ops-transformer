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
