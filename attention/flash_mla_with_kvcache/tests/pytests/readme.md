# FlashMlaWithKvcache 功能矩阵（pytest）

flash_mla_with_kvcache（DeepSeek MLA，Ascend 950 / DAV_3510）功能矩阵测试，复用 flash_attn 的
`tests/pytests` harness 骨架，按 MLA(m0070) 接口适配，**布局限定 TND/NZ/NTD（R1）**。

## 运行

```bash
cd attention/flash_mla_with_kvcache/tests/pytests
# NPU 全量矩阵
python test_flash_mla_with_kvcache.py
# 指定 case
python test_flash_mla_with_kvcache.py --case_id PREFILL_NO_MASK,DECODE_FD
# 只看 metadata 分核
python test_flash_mla_with_kvcache.py --case_id SECTION_64B --meta_only
# 追加 M9 静态图 vs 动态图 + 确定性检查
python test_flash_mla_with_kvcache.py --case_id PREFILL_LSE --static_graph
```

## 执行矩阵

| 维度 | 覆盖 |
|---|---|
| prefill (FD=false) | PREFILL_NO_MASK / PREFILL_CAUSAL / PREFILL_BF16 / PREFILL_LSE / PREFILL_CAUSAL_LSE |
| decode (FD path) | DECODE_FD / DECODE_CAUSAL_BF16 / DECODE_LSE |
| PA (block_table) | 全部（R1 必选路线：q=TND, kv=PA_NZ, out=NTD） |
| LSE 开关 | return_softmax_lse True/False |
| 静态图 vs 动态图 | `--static_graph`（确定性 + torch.export 静态子图） |
| >1-section | SECTION_64B / SECTION_16B_2K（断言 metadata sectionNum>=2） |
| varlen | VARLEN_NO_MASK / VARLEN_CAUSAL |
| dtype | fp16 / bf16 |
| mask_mode | 0 (NO_MASK) / 3 (CAUSAL/RIGHT_DOWN) |

精度判定：`utils/compare.py`（flash_attn 同款阈值，fp16 rtol=0.5% atol=2.5e-5，
bf16 rtol=0.78125% atol=1e-4，超阈占比 ≤ 0.5%）。golden 见
`core/backends/cpu_golden.py`：S = (Q‖Qrope)·(K‖Krope)^T * scale（576 宽），
O = P·V_nope（512），LSE (N, T)。

## 目录

```
pytests/
├── test_flash_mla_with_kvcache.py            # 入口
├── core/
│   ├── data.py                  # InputSpec + build_flash_mla_with_kvcache_params(m0070)
│   ├── case_loader.py           # 用例加载/规范化（TND/NZ/NTD）
│   ├── orchestrator.py          # 编排（含 NC 自洽/确定性/静态图分支）
│   ├── reporter.py              # 终端/CSV
│   └── backends/{base,cpu,cpu_golden,npu}.py
├── test_cases/
│   ├── functional_stc.py        # 功能矩阵
│   └── functional_nc.py         # 最小 NC 探测子集
└── utils/{compare,data}.py
```