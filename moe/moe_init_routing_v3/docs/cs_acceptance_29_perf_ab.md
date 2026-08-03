# moe_init_routing_v3 计数排序 §2.9 验收用例 A/B 结果

- 日期：2026-08-14
- 芯片：A5 (Ascend950DT)，device 6，串行单用例执行
- 来源：设计文档 §2.9 验收标准（4 shapes）+ §1.2.4 固定参数
- 用例文件：`/home1/l00939308/ops-test-kitB2/transpose/moe_v3_test_acceptance_29.csv`
- CS 侧日志：`/home1/l00939308/ops-test-kitB2/transpose/accept29_logs/ti{0..3}.log`
- BI 侧日志：`/home1/l00939308/ops-test-kitB2/transpose/accept29_bi_logs/ti{0..3}.log`（卸载 vendor 包后运行）
- 执行命令：`TTK_DISABLE_BUILTIN=0 python3 -m ttk kernel -i transpose/moe_v3_test_acceptance_29.csv --ti=N --device-whitelist=6`

## 0. 摘要

- **decode 两条验收用例全部优化**：92B decode（FullLoad 400000）**-20.0%**，505B decode（CutOrigin 500000）**-10.4%**
- **prefill 两条用例 CS 未启用**：active_num=T×8=1048704 超出 CS_MAX_N=1048576（仅超 128），双侧同走 builtin 11000000，无劣化亦无 CS 收益
- 8 次运行全部 PRECISION PASS，无精度退化
- 无任何用例劣化

## 1. 用例定义（§2.9 + §1.2.4）

固定参数：dropless（drop_pad_mode=0）+ expert_tokens_num_type=1（GATHER→COUNT）+ expert_tokens_num_flag=True + quant_mode=-1（unquant）+ row_idx_type=0，K=8。

| ti | testcase_name | T | H | E | LE | active_num |
|----|---------------|------|------|-----|-----|-----------|
| 0 | cs_accept_92b_decode_t1792_h2560_k8_e256_le32 | 1792 | 2560 | 256 | 32 | 14336 |
| 1 | cs_accept_505b_decode_t2560_h5120_k8_e384_le6 | 2560 | 5120 | 384 | 6 | 20480 |
| 2 | cs_accept_92b_prefill_t131088_h2560_k8_e256_le32 | 131088 | 2560 | 256 | 32 | 1048704 |
| 3 | cs_accept_505b_prefill_t131088_h5120_k8_e384_le24 | 131088 | 5120 | 384 | 24 | 1048704 |

注：§2.9 第 4 条原文 E=256 与 §1.2.3/§1.2.4 的 E=384 冲突，按 §1.2.4 取 **E=384**（CSV remark 已注明）。

## 2. A/B 结果

| ti | 用例 | CS 侧 TK | CS 路径 | CS µs | BI µs | Δ% | 判定 |
|----|------|----------|---------|-------|-------|-----|------|
| 0 | 92B decode | 400000 | FullLoad | **33.528** | 41.930 | **-20.0%** | ✅ 优化 |
| 1 | 505B decode | 500000 | CutOrigin | **45.030** | 50.237 | **-10.4%** | ✅ 优化 |
| 2 | 92B prefill | 11000000 | builtin | 2077.805 | 2080.581 | ~0% | ⚪ 双侧同 builtin |
| 3 | 505B prefill | 11000000 | builtin | 2283.355 | 2286.324 | ~0% | ⚪ 双侧同 builtin |

Δ% = (CS - BI) / BI，负值为优化。

## 3. 路由分析

- **ti=0（92B decode）**：T×8=14336 > sortLoopMaxElement，ep_=1（LE×2=64 ≤ E=256，部分专家范围），K=8、unquant、dropless、COUNT、LE=32≤256，UB 可全载 → **FullLoad 400000**。八条 AND 全满足。
- **ti=1（505B decode）**：H=5120 大列导致 FullLoad UB 超限（perCore 40 行 × 5120 × 2B ≈ 400KB > 248KB UB）→ 回落 **CutOrigin 500000**。属合法回退，且满足 CutOrigin 优化三因子（K≥3 + H≥5120 + T≥2048），仍取得 -10.4% 收益。
- **ti=2/3（prefill）**：active_num=T×8=1048704 **超出 CS_MAX_N=1048576（1M）上界**（恰好超 128），且 T 大 UB 无法全载 → 双侧同走 **builtin 11000000**。

## 4. 结论与风险

1. decode 验收用例收益与既有规律吻合：部分专家范围 FullLoad（LE 小→收益大，LE=32 达 -20%）；CutOrigin 三因子（K8+H5120+T2560）。
2. **prefill 用例是 CS 覆盖盲区**：T=131088 是 §2.9 明确的验收 shape，但 active_num 恰超 CS_MAX_N 128 个元素导致 CS 完全不启用。
   - 若 §2.9 验收要求 4 条 shape 全部走 CS 新路径 → 需评估放宽 CS_MAX_N（如 1048704 → 上调至 ≥131088×8）的 UB/性能影响
   - 若验收仅要求不劣于 A3 → 当前 builtin 路由安全（A3 侧同 shape 也走旧路径）
3. §2.9 验收标准为"A5 优于 A3"，本报告仅为 A5 内部 CS vs builtin 对比；A3 侧数据需另行采集。

## 5. 异常记录

- ti2 BI 侧首跑遇驱动瞬时错误（SqCq alloc ret=7，device 6），重跑恢复，非算子问题。
