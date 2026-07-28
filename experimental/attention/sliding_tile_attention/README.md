# SlidingTileAttention

## 功能说明

SlidingTileAttention 是 FastVideo 视频生成场景使用的三维滑动 tile attention。输入 `q/k/v` 为 rank-4
BNSD Tensor，输出与 `q` 的 shape、dtype 和 layout 一致：

```text
attentionOut = softmax((Q @ K^T) / sqrt(D), masked_by_sliding_tile_window) @ V
```

图像 token 由三维画布和固定 tile `[6,8,8]` 划分。每个 image query 关注本 head 的局部 tile window；
窗口中心在画布边界处做 clamp。有 text 时，image query 额外关注全部 text KV，text query 关注完整序列。

## 支持边界

| 项 | 支持范围 |
| --- | --- |
| 产品 | Atlas A2/A3 系列，对应 `ascend910b`、`ascend910_93` |
| layout | rank-4 BNSD；`q/k/v` 可为连续或非连续 Tensor |
| dtype | `float16`、`bfloat16`，`q/k/v/output` 必须一致 |
| batch | 非空计算要求 `B > 0`；shape 一致的任意空 Tensor（包括 `B=0`）由 ACLNN GetWorkspaceSize 快返 |
| heads | `1 <= N <= 128` |
| head dimension | `D > 0` 且 `D % 8 == 0` |
| tile | 固定 `[6,8,8]` |
| window | 长度为 3 的 `[t,h,w]` 广播，或长度为 `3*N` 的逐 head 三元组；每个三元组的三个值均须为正奇数 |
| sequence | 由 `seqShape` 对应画布 token 数与 `textLength` 共同确定 |

ACLNN L2 对 `q/k/v` 显式执行 `Contiguous`，因此非连续输入先 materialize 为连续 BNSD，再进入自定义
Ascend C 路径。输入输出 shape/dtype 不匹配、非法 window 或其他不支持 metadata 会返回参数错误；实现不调用
Python、CPU、PyTorch、FastVideo、Triton/CUDA、vendor whole-op 或其他后端作为 fallback。

## 接口与实现

公开接口为标准两段式 ACLNN：

- `aclnnSlidingTileAttentionGetWorkspaceSize`
- `aclnnSlidingTileAttention`

详细参数见 [docs/aclnnSlidingTileAttention.md](docs/aclnnSlidingTileAttention.md)。实现架构见
[docs/SlidingTileAttention_design.md](docs/SlidingTileAttention_design.md)。

L2 负责契约校验、输入连续化和 window 规范化。shape 一致的任意空 Tensor 在
`aclnnSlidingTileAttentionGetWorkspaceSize` 直接返回成功，不创建 L0 launcher，也不发射 device kernel。
非空输入按合法运行时 metadata 分派：

| 运行时条件 | 路径 | 说明 |
| --- | --- | --- |
| `N=8 && D=64` | 固定快路径 | 保护原 18-case 主形状的既有调度和性能 |
| 其他合法 `N/D` | 动态通用路径 | runtime heads、`1/sqrt(D)`、分块 D 和逐 head window |
| 非法 metadata | fail loudly | 返回参数错误，不 fallback |

两条计算路径均由本目录源码构建。设备端使用 solution-owned MIX kernel 完成 QK、online softmax、PV、
rescale、归一化和写回。BF16 路径不发射外部 `Cast(Q/K/V) + FAI + Cast(O)` 链；当前 CATLASS PV
lowering 所需的 V tile 在同一次 STA kernel 内 staging 为 half，随后沿标准 online-softmax/PV/rescale
数据流直接写回 BF16。该内部表示转换不是独立 Cast 算子，也不是 framework/vendor fallback。

## 编译

```bash
source /path/to/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910_93 \
    --ops=sliding_tile_attention --vendor_name=sta_review -j16
```

A2 包将 `--soc` 改为 `ascend910b`。

## 验证

Host 契约测试：

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
    experimental/attention/sliding_tile_attention/tests/case_guard_test.cpp \
    -Iexperimental/attention/sliding_tile_attention/op_host \
    -o /tmp/sta_case_guard_test
/tmp/sta_case_guard_test
```

预期标志：

```text
ALL_SLIDING_TILE_ATTENTION_CASE_GUARD_TESTS_PASSED dynamic_heads=1..128 head_dim_multiple=8
```

22-case NPU reviewer 复跑入口：

```bash
source /path/to/cann/set_env.sh
bash experimental/attention/sliding_tile_attention/examples/run_22cases.sh
```

脚本会构建当前源码生成的自定义算子包，并安装到 `/tmp` 下的隔离 OPP 根中执行原 18 个固定快路径保护
case，以及 case 19、23、24、25 四个 N/D/window/batch/layout 泛化 case和独立空 Tensor API 检查。按本次
PR 自测范围，case 20、21、22、26 不进入 runner；这不改变实现的运行时支持边界，也不构成它们的性能结论。
设备输入使用非零、非均匀
Q/K 和确定性 V，host 期望值显式应用 runtime `1/sqrt(D)`，并采样所有合法的
`0,7,63,64,127,128,D-1` 列；退出时删除临时
目录，不修改活动 CANN 的 `vendors/config.ini` 或 vendor 目录。预期最终标志：

```text
API_PASS empty_tensor workspace=0
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=22
```

这 22 个 case 是 reviewer 可复跑的代表性证据，不是完整支持域。执行后的具体设备与日志证据边界见
[docs/self_validation.md](docs/self_validation.md)。
