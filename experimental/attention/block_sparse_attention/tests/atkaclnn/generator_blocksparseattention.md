# generator_blocksparseattention.py 参数配置表

## 生成器参数配置表

| 维度 | 生成方式 | 取值/范围 | 说明 |
|---|---|---|---|
| batch B | `random.randint(1, batch_hi)` | 1–16 | `batch_hi=16` |
| Q 序列长度 | `random.randint(1, s1_hi)` | 1–4096 | s1=qs |
| KV 序列长度 | `random.randint(1, s2_hi)` | 1–30000 | s2=kvs |
| Q 头数 qn | `N2 × G` | 1–64 | kvn=N2(1–8),整除约束 qn%kvn==0 |
| head_dim D | `random.choice` | 64 / 128 | |
| 块 X / 块 Y | `random.choice` 各自独立 | 64 / 128 / 256 随机组合 | |
| 布局 | `random.choice` | BNSD / BSND / TND | Q/KV 同布局 |
| 输出 dtype | `random.choice` | bf16 / fp16 | |
| quantMode | `random.choice` | 2(OCP) / 3(CX) | |
| dstTypeMax | quant3: `0.0 或 uniform(6,12)`(各半);quant2: `0.0` | | |
| pad_kind | `random.choice` | both / kv_only / q_only / decode / block_aligned / tight / spread | 仅 BNSD/BSND 且 is_padded=1 |
| is_padded | `random.choice` | 0 / 1 | BNSD/BSND |
| is_varied_len | `random.choice` | 0 / 1 | 仅 TND |
| inner_precise | 固定 | 4 | |
| maskType | 固定 | 0 | |
| blockSize | 固定 | 0 | |
| preTokens / nextTokens | 固定 | 2147483647 | 全窗口 |
| softmaxLseFlag | 固定 | 0 | 不返回 LSE |
| scaleValue | `1 / sqrt(D)` | D=64→0.125,D=128→0.0884 | |
| attenMask | tensor | `range_values=["None"]` | 不物化,按默认处理 |
| blockTable | tensor | `range_values=["None"]` | 不物化,按默认处理 |
| softmaxLse | tensor | `range_values=["None"]` | 不物化(softmaxLseFlag=0 无需输出) |


## 备注

- 规模退避(COST_BUDGET / fit_cost_budget)已注释禁用,生成 case 不再按 `B*N1*S1*S2*D` 回退,大组合可能造成 CPU golden 极慢。
- 当前 yaml `dtype_numbers` 按 2 倍规则生成 case(填 N → 实际 2N)。
