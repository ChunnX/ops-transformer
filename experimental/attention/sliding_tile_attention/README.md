# SlidingTileAttention

## 功能说明

SlidingTileAttention 是 FastVideo 视频生成场景使用的三维滑动 tile attention。输入 `q/k/v` 为连续
BNSD Tensor，输出与 `q` 的 shape、dtype 和 layout 一致：

```text
attentionOut = softmax((Q @ K^T) / sqrt(D), masked_by_sliding_tile_window) @ V
```

图像 token 由三维画布和固定 tile `[6,8,8]` 划分。每个 image query 关注本 head 的局部 tile window；
窗口中心在画布边界处做 clamp。有 text 时，image query 额外关注全部 text KV，text query 关注完整序列。

## 交付边界

根据任务书验收范围，本次交付只覆盖自验证表中的 18 个固定 case，不扩展动态场景。固定边界如下：

| 项 | 支持范围 |
| --- | --- |
| 产品 | Atlas A2/A3 系列，对应 `ascend910b`、`ascend910_93` |
| layout | 连续 BNSD |
| dtype | `float16`、`bfloat16`，`q/k/v/output` 必须一致 |
| shape | `N=8`、`D=64`，以及固定 case 表声明的 `B/S` |
| tile | `[6,8,8]` |
| window | 单个 `[t,h,w]` 三元组广播，或 8 个 head 的 24 个整数 |
| text | 固定有 text 场景使用 `textLength=128` |

范围外 metadata 会返回参数错误，不调用 Python、CPU、PyTorch、FastVideo、Triton/CUDA、vendor whole-op
或其他后端作为 fallback。

## 接口与实现

公开接口为标准两段式 ACLNN：

- `aclnnSlidingTileAttentionGetWorkspaceSize`
- `aclnnSlidingTileAttention`

详细参数见 [docs/aclnnSlidingTileAttention.md](docs/aclnnSlidingTileAttention.md)。实现架构见
[docs/SlidingTileAttention_design.md](docs/SlidingTileAttention_design.md)。

L2 入口完成固定 case 校验和 window 规范化；L0 只登记一次 `SlidingTileAttentionFai` launcher；设备端以一个
MIX kernel 完成 QK、online softmax、PV、BF16 修正及输出写回。BF16 路径已移除旧的外部
`Cast(Q/K/V) + FAI + Cast(O)` 多 kernel 链，必要的数据转换由同一次 STA kernel launch 内部拥有。

## 编译

```bash
source /path/to/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910_93 \
    --ops=sliding_tile_attention --vendor_name=sta_review -j16
```

A2 包将 `--soc` 改为 `ascend910b`。

## 验证

Host 固定签名测试：

```bash
g++ -std=c++17 -Wall -Wextra -Werror \
    experimental/attention/sliding_tile_attention/tests/case_guard_test.cpp \
    -Iexperimental/attention/sliding_tile_attention/op_host \
    -o /tmp/sta_case_guard_test
/tmp/sta_case_guard_test
```

预期标志：

```text
ALL_SLIDING_TILE_ATTENTION_CASE_GUARD_TESTS_PASSED signatures=17 cases=18
```

NPU reviewer 复跑入口：

```bash
source /path/to/cann/set_env.sh
bash experimental/attention/sliding_tile_attention/examples/run_18cases.sh
```

脚本会构建当前源码生成的自定义算子包，并安装到 `/tmp` 下的隔离 OPP 根中执行 18 个固定 case；
退出时删除临时目录，不修改活动 CANN 的 `vendors/config.ini` 或 vendor 目录。预期最终标志：

```text
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=18
```

完整 case 表、验证口径和既有 OpForge 迁移证据见
[docs/self_validation.md](docs/self_validation.md)。
