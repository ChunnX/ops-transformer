# SlidingTileAttention 自验证说明

## 验证范围

根据任务书验收范围，本次只验证以下 18 个固定 case，不补动态场景。所有输入均为连续 BNSD，
`N=8`、`D=64`、tile=`[6,8,8]`，输入值域为 `[-0.5,0.5]`。

| case | dtype | shape `[B,N,S,D]` | text | seqShape | window | 覆盖点 |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | FP16 | `[1,8,115328,64]` | 128 | 30x48x80 | source mix | FastVideo 正式有 text |
| 2 | BF16 | `[1,8,115328,64]` | 128 | 30x48x80 | source mix | BF16 有 text |
| 3 | FP16 | `[1,8,115328,64]` | 128 | 30x48x80 | wide mix | 边界 clamp |
| 4 | BF16 | `[1,8,115328,64]` | 128 | 30x48x80 | wide mix | BF16 clamp |
| 5 | FP16 | `[1,8,82944,64]` | 0 | 36x48x48 | source mix | no-text source mix |
| 6 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | source mix | BF16 no-text |
| 7 | FP16 | `[1,8,82944,64]` | 0 | 36x48x48 | wide mix | wide clamp |
| 8 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | wide mix | BF16 wide clamp |
| 9 | FP16 | `[1,8,82944,64]` | 0 | 36x48x48 | uniform 333 | 正式性能形状 |
| 10 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | uniform 333 | BF16 性能形状 |
| 11 | FP16 | `[1,8,115328,64]` | 128 | 30x48x80 | uniform 333 | text + uniform |
| 12 | BF16 | `[1,8,115328,64]` | 128 | 30x48x80 | uniform 333 | BF16 text + uniform |
| 13 | FP16 | `[1,8,69120,64]` | 0 | 18x48x80 | uniform 333 | Wan 画布 |
| 14 | BF16 | `[1,8,69120,64]` | 0 | 18x48x80 | uniform 333 | BF16 Wan 画布 |
| 15 | FP16 | `[1,8,69120,64]` | 0 | 18x48x80 | Wan mixed | odd window mix |
| 16 | BF16 | `[1,8,69120,64]` | 0 | 18x48x80 | Wan mixed | BF16 odd mix |
| 17 | FP16 | `[2,8,69120,64]` | 0 | 18x48x80 | uniform 333 | batch=2 |
| 18 | BF16 | `[1,8,82944,64]` | 0 | 36x48x48 | 单 `[3,3,3]` | L2 window 广播 |

## Reviewer 复跑

```bash
source /path/to/cann/set_env.sh
bash experimental/attention/sliding_tile_attention/examples/run_18cases.sh
```

可通过环境变量覆盖默认值：

```text
SLIDING_TILE_ATTENTION_SOC_VERSION=ascend910_93
SLIDING_TILE_ATTENTION_DEVICE_ID=0
SLIDING_TILE_ATTENTION_BUILD_THREADS=16
```

脚本构建当前 PR 源码、在 `/tmp` 下临时安装 `sta_review` vendor、编译 PR 内 C++ runner，并依次执行
18 个 case。退出时删除隔离安装目录，不修改活动 CANN 的 vendor 或 `vendors/config.ini`。成功标志为：

```text
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=18
```

## 判定方法

runner 在完整长序列上执行当前 ACLNN custom op，使用独立的确定性输入和解析期望值检查 window、text、
dtype 和输出写回；不读取 OpForge task/cases/golden 文件，不调用 framework/reference fallback。

迁移来源的冻结 Ascend C 源码曾在 Ascend910_9382（physical=6, card=3, chip=0, logical=npu:0）通过
OpForge `eval_20260719_205421` 的 18/18 correctness gate，active-window geomean 为 17.4551x，最小为
10.2001x，18 个 case 均为单 kernel、零 device gap、零 active regression。该记录不是当前源码仓 runner
的替代证据；PR 发布结论使用当前 head 的 fresh package 和 runner 日志。

## 明确限制

本次不声明固定表之外的动态 head、`D=128`、`textLength=256`、任意 batch/sequence/window 支持。范围外
metadata 会显式失败，不 fallback。

BF16 路径已满足一次外部 STA kernel launch，但并非设备内全程保持 BF16：Q/K 保持 BF16，V tile 因当前
CATLASS PV lowering 限制在同一 kernel 内 staging 为 half，dense PV 后通过 fp32 修正并直接写回 BF16。
该边界不增加独立 Cast kernel，也不使用 framework、CPU、vendor whole-op 或其他后端 fallback。
