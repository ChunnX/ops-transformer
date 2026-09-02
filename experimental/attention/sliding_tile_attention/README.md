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
| sequence | `seqShape=1` 支持任务书的 padded storage：`S-image_tokens <= 512` 且 `textLength <= S-image_tokens`；`seqShape=2/3` 为无 text 的固定画布长度 |

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

| 运行时条件 | tiling key / 路径 | 说明 |
| --- | --- | --- |
| `dtype=BF16 && D=64` | key 1 / BF16 D64 lean 变体 | 编译期裁剪不需要的通用分支，保留 runtime B/N/sequence/window 解码 |
| 已证明的 FP16 shape family | key 2/3/5/7/8/9 | N1 packed、D72 non-full grouped、D16 text grouped 和 D256 text packed |
| D256 padded-text 分片 | key 10/11 | solution-owned partial/merge，按 runtime capacity 分片并合并 |
| 其余合法 dtype/D | key 0 / 通用变体 | runtime heads、`1/sqrt(D)`、分块 D、逐 head window 和 D72 full-window 双概率 PV |
| 非法 metadata | 无 kernel / fail loudly | 返回参数错误，不 fallback |

所有 tiling key 均由本目录源码构建并打包。内核在变体内部继续按合法 runtime metadata 选择固定
384-row task decode、grouped/compact 或动态通用数据流；case id、输入值和 timing 不参与分派。设备端使用
solution-owned Ascend C kernel 完成 QK、online softmax、PV、rescale、归一化和写回。BF16 路径不发射外部
`Cast(Q/K/V) + FAI + Cast(O)` 链；当前 CATLASS PV
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

原始任务书 50-case NPU 功能复跑入口：

```bash
source /path/to/cann/set_env.sh
bash experimental/attention/sliding_tile_attention/examples/run_taskbook_cases.sh
```

该入口逐项执行任务书 `all_sta.json` 的 case 0..49；dtype、B/N/S/D、window、textLength、hasText 和
seqShape 与原始表一致。每个 case 都在当前 PR 构建的 ACLNN custom op 上执行，检查独立解析期望值、padded
storage query、同输入 bitwise 确定性和空 Tensor API。任务书中的历史性能数值不作为此功能复跑的结论。

额外 22-case NPU reviewer 复跑入口：

```bash
source /path/to/cann/set_env.sh
bash experimental/attention/sliding_tile_attention/examples/run_22cases.sh
```

两个脚本都会构建当前源码生成的自定义算子包，并安装到 `/tmp` 下的隔离 OPP 根中。reviewer 入口执行原
18 个固定快路径保护
case，以及 case 19、23、24、25 四个 N/D/window/batch/layout 泛化 case 和独立空 Tensor API 检查。
runner 固定执行上述 22 个 case；未列入该表的 workload 不由这份自验证日志给出性能结论。
设备输入使用非零、非均匀
Q/K 和确定性 V，host 期望值显式应用 runtime `1/sqrt(D)`，并采样所有合法的
`0,7,63,64,127,128,D-1` 列；退出时删除临时
目录，不修改活动 CANN 的 `vendors/config.ini` 或 vendor 目录。预期最终标志：

```text
API_PASS empty_tensor workspace=0
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED suite=taskbook cases=50
ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED suite=reviewer22 cases=22
```

任务书 50 例证明规定 case 全部执行；额外 22 例提供接口和泛化保护。两套 case 都不是完整支持域的穷举。
当前 PR 源码树摘要
`716922e9a9d522a2a3c768ea89935e2518630c869c90a163133de6f21a57984d` 已完成任务书
50/50 和 reviewer22 22/22 复跑；72 个设备 case 均为 bitwise deterministic，两个 suite 的空 Tensor
workspace0/API 检查均通过。原始日志、逐 case 精度/设备延迟 CSV、任务书字段映射、源码摘要和校验清单位于
[docs/evidence](docs/evidence)，完整判定边界见 [docs/self_validation.md](docs/self_validation.md)。CSV 延迟仅用于
本地复现和异常定位，不采用任务书历史性能数值，也不构成性能达标或 speedup 声明。
