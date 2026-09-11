# 三个 MLA 算子 FD 性能测试用例

本文用于 `sparse_flash_mla`（`smla`）、`quant_sparse_flash_mla`（`qsmla`）和 `mixed_quant_sparse_flash_mla`（`mqsmla`）的共同测试。每个用例都执行 FD=OFF/ON 对照；三个算子只替换各自要求的 dtype、`quant_mode`、descale 和 KV 存储格式。

## 2. 统一场景、模式和布局

### 2.1 参数约定与 FD 对照

| 名称 | 含义 |
| --- | --- |
| `BS1` | batch size `b` |
| `N1` | Query head 数 `num_heads_q`/`q_n`；`num_heads_kv=1` |
| `KV` | 每个 batch 的原始 KV 有效长度 `ori_kv_s` |
| `Q` | Query 长度 `q_s`；prefill 为大于 1，decode 固定为 1 |
| `CKV` | 压缩 KV 长度，满足 `CKV * cmp_ratio + cmp_residual_kv = KV`（按实际压缩定义取整） |
| `K1` | `ori_sparse_indices` 的 TopK（仅全稀疏模式） |
| `K` | `cmp_sparse_indices` 的 TopK（CSA 和 ORI_CMP_SPARSE） |

固定 `head_dim=512`、`num_heads_kv=1`、`topk_value_mode=1`、`return_softmax_lse=False`；所有用例统一传入 shape 为 `(N1,)` 的 `sinks`，不将 sinks 作为性能变量。metadata 和主算子的所有切分参数必须逐项一致。

FD=OFF 使用 `supportFd_=false`，FD=ON 使用 `supportFd_=true`（或当前版本对应的 metadata 开关）。两者均关闭 batch 一致性，不能调用 `torch_npu.npu.set_deterministic_level(3)`；同一输入交替测量两个版本。

三个算子的输入适配如下：`smla` 使用 `float16/bfloat16`；`qsmla` 使用 `quant_mode=1`、hifloat8 Q/K/V 及三个 descale；`mqsmla` 分别测试 `quant_mode=1`、`2`，Q 为 `bfloat16`、KV 使用对应混合 FP8 格式、`rope_head_dim=64`。

### 2.2 五种支持模式

| 模式 | 必传输入 | 参数规则 |
| --- | --- | --- |
| `SWA` | `ori_kv` | 不传 `cmp_kv`、两个 sparse index；`ori_topk=cmp_topk=0`、`cmp_ratio=1`、`cmp_mask_mode=0`；`ori_mask_mode` 覆盖 `0/3/4`。 |
| `HCA` | `ori_kv`、`cmp_kv` | 不传 sparse index；`ori_topk=cmp_topk=0`、`cmp_mask_mode=3`；`cmp_ratio=4` 和 `128`。 |
| `CSA` | `ori_kv`、`cmp_kv`、`cmp_sparse_indices` | 不传 `ori_sparse_indices`；`ori_topk=0`、`K>0`、`cmp_mask_mode=3`；`cmp_ratio=4` 和 `128`。 |
| `ORI_SPARSE` | `ori_kv`、`ori_sparse_indices` | 不传 `cmp_kv`；`K1>0`、`cmp_topk=0`、`cmp_ratio=1`；`ori_mask_mode` 覆盖 `0/3/4`。 |
| `ORI_CMP_SPARSE` | `ori_kv`、`ori_sparse_indices`、`cmp_kv`、`cmp_sparse_indices` | `K1>0`、`K>0`；`ori_mask_mode` 覆盖 `0/3/4`，`cmp_mask_mode` 覆盖 `0/3`；`cmp_ratio=4` 和 `128`。 |

`ORI_SPARSE` 和 `ORI_CMP_SPARSE` 是两个全稀疏路径。`smla` 当前版本即使使用 `ori_sparse_indices`，metadata 的 `ori_topk` 仍按文档约束保持 0；`K1` 由 sparse index 最后一维表达。

### 2.3 主测布局与备份布局

所有主性能用例统一使用 `layout_q=BSND`、`layout_kv=BSND`（L-BB）。表中的每一个 `BS1/N1/Q/KV/TopK/cmp_ratio` 组合都是一个独立 case；不得在一次 case 内通过 `seqused_q`、`seqused_ori_kv` 或 `seqused_cmp_kv` 混合多个长度。

只在重要 case 上增加以下备份布局：

| 编号 | `layout_q` | `layout_kv` | 必要附加输入 |
| --- | --- | --- | --- |
| L-BB（主测） | `BSND` | `BSND` | 每个 case 使用固定 `BS1/Q/KV`；不使用 `seqused_*` 改变同一 case 的工作量 |
| L-TT（备份） | `TND` | `TND` | 每个 batch 的 Q、ori KV、cmp KV 长度分别相等；`cu_seqlens_*` 等间隔递增 |
| L-BP（备份） | `BSND` | `PA_BBND` | 仅重要 case；`ori_block_table`，有 cmp 时增加 `cmp_block_table`；`seqused_*` 在 batch 间取相同固定值 |
| L-TP（备份） | `TND` | `PA_BBND` | 仅重要 case；每个 batch 长度相等，配合 `cu_seqlens_q`、block table 和固定 `seqused_*` |

这四种组合覆盖当前接口的全部合法布局关系；非 PA 场景的 `BSND/TND` 交叉组合不支持。PA 用例固定 `block_size=128`，并测试连续页和随机置换页。L-TT/L-BP/L-TP 只作为 L-BB 重要 case 的备份，不替代主测结果。

### 2.4 TopK 和 KV 扫描集合

Ascend 950 上，CSA 的 `K`、ORI_SPARSE 的 `K1`、ORI_CMP_SPARSE 的 `K1/K` 均使用：

```text
TopK = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
```

每次保证 TopK 不超过对应有效 KV/token 数；若 KV 小于某个 TopK，则使用不超过 KV 的最大集合值，并记录实际值。SWA/HCA 的 TopK 固定为 0。目标产品若对 TopK 有更窄约束（例如只支持 512/1024），保留允许值，其余标记为 `UNSUPPORTED`，不可静默改成 0。

Decode 所有模式都按以下 KV 集合测量，最大值为 128K，且小长度区间更细；prefill 也从该集合取值，但必须满足 `Q>1` 及该算子对 Q/KV 长度关系的约束。`KV=1` 无法构成 `Q>1` 的 prefill case，因此由 decode 用例覆盖。

```text
KV = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
      1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K}
```

`cmp_ratio=1` 只用于 SWA；`cmp_ratio=4`、`128` 用于 HCA、CSA、ORI_CMP_SPARSE。对 ratio=128 的小 KV，按 `CKV` 和 `cmp_residual_kv` 规则构造合法压缩长度，不能因为 `CKV=1` 而跳过该点。

### 2.5 还需单独控制的性能敏感参数

以下参数不能在一个 case 内混合；应从主矩阵中选定一个固定 case，复制出独立的参数变体：

| 参数 | 建议独立取值 | 原因与优先级 |
| --- | --- | --- |
| `N1` / group size | `1,2,4,8,16,32,64,65,96,128` | **P0**。Ascend 950 metadata 在 `groupSize>64` 时进入 split-G 路径并减少 AIC core 数，`64→65` 是必须测的边界；当前只测 32/64/128 会漏掉小 N1 和切换点。 |
| Q 行数（prefill） | `2,8,16,17,32,33,64,128,512,2048` | **P0**。FD 归约按 16 行 chunk 处理，应覆盖 `16→17`、`32→33` 等尾块边界；每个 Q 值为独立 case。decode 的 `Q=1` 单独保留。 |
| KV S2 切分边界 | 在现有集合加入 `127,129,255,257,511,513,1023,1025,131071,131072` | **P0**。Ascend 950 metadata 的 S2 基本切分为 128，边界会改变 split 数、tail 和 FD workspace；128K 点保留。ratio=4/128 时也要检查 `CKV` 的对应边界。 |
| `cmp_residual_kv` | ratio=4：`0,1,3`；ratio=128：`0,1,127` | **P0**。余数会改变 cmp mask 恢复长度和最后一个 S2 block；每个余数单独成 case，并同步传给 metadata/主算子。 |
| `ori_win_left/right` | `left={0,1,16,127,128,512,-1}`，`right={0,1,-1}` | **P1**。窗口决定实际 ori S2 工作量；`127/128` 是切分边界。`ori_mask_mode!=4` 时窗口必须按接口要求设为 `-1`。 |
| 实际 TopK length | 分别测 `actual K = 1、K/4、K/2、K`，sparse index 容量保持固定 | **P1**。相同 index 容量下有效长度不同会改变实际计算量；mask=0 时传对应 topk length，其他 mask 遵守接口限制。每个 actual K 是独立 case。 |
| TopK 索引分布 | 连续/相邻 block、均匀随机、跨页随机 | **P1**。相同 TopK 数量的访存局部性和 PA 页访问数不同；索引值保持合法，`-1` 只放在无效尾部。 |
| PA block size | `64,128,256`（再按产品允许值扩展） | **P1**。block size 改变 block 数、尾块和 block table 访存；每个 block size 独立 case，主矩阵仍以 BSND/BSND 为准。 |
| `return_softmax_lse` | `false`（主测）、`true`（重要大 KV 备份） | **P1**。true 会增加 LSE 写回及 FD 归约路径，不能把两者耗时混在一组。 |

`num_heads_kv`、`head_dim` 和 `topk_value_mode` 当前分别受限为 1、512 和 1，不应人为扩展成性能维度；`softmax_scale` 只改变数值缩放，通常不改变 tiling 或访存路径。

## 3. Prefill 用例

Prefill 固定 `Q>1`，建议使用 `Q={2,4,8,16,32,64,128,512,2048}`；主测全部使用 L-BB，并执行 FD=OFF/ON。表中每个参数组合单独建立一个 case，且只选择满足长度约束的 `Q/KV` 配对（例如 `Q<=KV`；`KV=1` 仅在 decode 测试）。除 mask 专项外，窗口参数固定 `ori_mask_mode=4`、`ori_win_left=127`、`ori_win_right=0`。

| 编号 | 模式 | `BS1` | `N1` | `Q` | `KV` | `cmp_ratio` | TopK |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PF-S | SWA | 1、8、32 | 32、64、128 | 每个 Q 单独成 case | 每个 KV 单独成 case | 1 | 0 |
| PF-H | HCA | 1、8、32 | 32、64、128 | 每个 Q 单独成 case | 每个 KV 单独成 case | 4、128 | 0 |
| PF-C | CSA | 1、8、32 | 32、64、128 | 每个 Q 单独成 case | 每个 KV 单独成 case | 4、128 | `K` 全集合 |
| PF-O | ORI_SPARSE | 1、8、32 | 32、64、128 | 每个 Q 单独成 case | 每个 KV 单独成 case | 1 | `K1` 全集合 |
| PF-OC | ORI_CMP_SPARSE | 1、8、32 | 32、64、128 | 每个 Q 单独成 case | 每个 KV 单独成 case | 4、128 | `K1/K` 全集合 |

主性能矩阵使用 `Q={16,512,2048}`、`BS1={1,8,32}`、`N1={32,128}`、全部 KV；每个笛卡尔积元素都是独立 case。其余 Q/N1/BS1 组合作为 P1 扩展。PF-C、PF-O、PF-OC 的 TopK 必须完整展开，不得只测单个 K。

### 3.1 Prefill mask 专项

对每种模式至少选择 `KV={128,2K,16K,128K}`、`Q={16,512}`、`BS1=1`、`N1=64` 跑 mask 笛卡尔积：

- SWA、ORI_SPARSE：`ori_mask_mode=0/3/4`，`cmp_mask_mode=0`。
- CSA：`ori_mask_mode=0/3/4`，`cmp_mask_mode=3`。
- ORI_CMP_SPARSE：`ori_mask_mode=0/3/4` × `cmp_mask_mode=0/3`。
- HCA：`ori_mask_mode=0/3/4`，`cmp_mask_mode=3`。

mask=0 时提供相应的 `ori_topk_length`/`cmp_topk_length`；mask 非 0 时遵守接口关于 topk length 的传入限制。

### 3.2 Prefill 布局备份

从 PF-S2、PF-H2、PF-C2、PF-O2、PF-OC2 中各选 `Q=512`、`BS1=8`、`N1=64`、`KV=16K/128K` 的独立 case，增加 L-TT、L-BP、L-TP 备份。TND case 中每个 batch 使用完全相同的 Q/KV/cmp KV 长度，例如 `Q=512`、`KV=16K`；`cu_seqlens_*` 等间隔递增，不使用不均匀 `seqused_q`。

## 4. Decode 用例

Decode 固定 `Q=1`，重点覆盖 `BS1={1,2,8,32}`、`N1={32,64,128}` 和全部 KV；主测全部使用 L-BB，每个参数组合独立执行 FD=OFF/ON。

| 编号 | 模式 | `BS1` | `N1` | `KV` | `cmp_ratio` | TopK |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DC-S | SWA | 1、2、8、32 | 32、64、128 | 每个 KV 单独成 case | 1 | 0 |
| DC-H | HCA | 1、2、8、32 | 32、64、128 | 每个 KV 单独成 case | 4、128 | 0 |
| DC-C | CSA | 1、2、8、32 | 32、64、128 | 每个 KV 单独成 case | 4、128 | `K` 全集合 |
| DC-O | ORI_SPARSE | 1、2、8、32 | 32、64、128 | 每个 KV 单独成 case | 1 | `K1` 全集合 |
| DC-OC | ORI_CMP_SPARSE | 1、2、8、32 | 32、64、128 | 每个 KV 单独成 case | 4、128 | `K1/K` 全集合 |

主性能矩阵优先执行 `BS1=1,N1=128`（FD 重点）、`BS1=8,N1=64` 和 `BS1=32,N1=128` 三组；每个 KV、ratio 和 TopK 组合是独立 case，仍须完整展开。对 DC-S、DC-C、DC-O、DC-OC 额外执行第 3.1 节的 mask 组合。

### 4.1 Decode PA 专项

对 DC-S、DC-H、DC-C、DC-O、DC-OC 各选 `BS1=1,N1=128` 和 `BS1=8,N1=64` 的重要独立 case，在 `KV={128,2K,16K,64K,128K}` 增加 L-BP/L-TP 备份：`block_size=128`；`ori_block_table` shape 为 `(BS1, ceil(KV/128))`，有 cmp 时 `cmp_block_table` shape 为 `(BS1, ceil(CKV/128))`；`seqused_ori_kv`、`seqused_cmp_kv` 在一个 case 内固定，且每个 batch 取相同长度。每个 shape 同时测试连续页和随机置换页。

每条结果至少记录 `op、quant_mode、case_id、layout_q、layout_kv、BS1、N1、Q、KV、CKV、K1、K、cmp_ratio、mask、FD 开关、tiling key、FD 是否触发、主算子中位耗时、P90、正确性状态`。FD 收益使用 `median_us(FD=OFF) / median_us(FD=ON)` 计算；FD 未触发或不支持的 TopK 单独标记，不能计入收益汇总。
