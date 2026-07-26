# SMLA/MQSMLA batch 一致性需求冲突与对齐清单

> 版本：2026-07-27 检视意见五、`pr_9103`处理版
> 用途：把“已裁决”“实现暂定”“必须找人确认”分开，禁止测试代码静默替需求做决定。
> 第一需求源：`docs/SMLA batch一致性.md`；STC Excel 是验收草案，不反向覆盖主需求。

## 1. 使用方式与责任人

状态只有四种：

| 状态 | 含义 | 是否可进入正式验收 |
| --- | --- | --- |
| `CONFIRMED` | 主需求、接口和实现证据一致 | 可以 |
| `IMPLEMENTED_PENDING_CONFIRMATION` | 为推进测试已采用可证明的暂定方案，但仍需需求/算子确认 | 可以做预检，不可关闭需求问题 |
| `BLOCKED_BY_SPEC` | 输入或公式自相矛盾，无法唯一生成用例 | 不可以 |
| `TEST_GAP` | 需求清楚，但对应 executor/case/环境尚未完成 | 不可以 |

待确认角色：

| 角色 | 需要确认的内容 |
| --- | --- |
| 需求/STC owner | 用例意图、baseline/derived 关系、字段勘误、验收口径 |
| 算子 tiling/kernel owner | level 3 开关、规约块公式、FA/FD metadata、分核和 workspace 语义 |
| pytest owner | Excel schema、模式构造、TokenOrigin、报告和状态分类 |
| Runtime/torch_npu owner | deterministic level 3 的版本、SoC、custom 包支持范围 |

## 2. 冲突总表

| ID | 冲突 | 证据 A | 证据 B | 当前处理 | 状态/待确认人 |
| --- | --- | --- | --- | --- | --- |
| C-01 | 模式 4 到底 prepend、尾部追加，还是共同前缀 | `docs/SMLA batch一致性.md:9` 明确 `300+100` 与 `300+200` 的前 300 token 相同 | `docs/会议纪要.md:65,73` 说在前面加 token；STC 004/009/013/017 说在后面追加 | 采用主需求“共同前缀”，位置 mask 同步调整逻辑 KV；另外两种写法保留在本清单，不删除 | `IMPLEMENTED_PENDING_CONFIRMATION`；需求/STC owner、算子 mask owner |
| C-02 | 规约 tile 的 `/32` 是 floor 还是 ceil | 需求写 `align_up(S2_b/32,128)`，原 analyzer 解释为 `ceil` | PR8737 head `31a3b6d3` 的 C++ 整数表达式先做 `actTotalS2Size / 32`，即 floor | 白盒报告同时输出 floor/ceil quotient、tile 和 conflict；结构展开暂按 PR head，并显式报告 zero edge | `BLOCKED_BY_SPEC`；tiling owner |
| C-03 | `S2_b` 是纯 `seqused_kv` 还是 mask/top-k 后可见负载 | 需求多处用 `seqused_kv[b]` 表述 | PR8737 head 使用 mask-visible `actOriS2Size + actCmpS2Size` | analyzer 使用实际 mask window 的 ori+cmp，并在报告列出可见长度 | `IMPLEMENTED_PENDING_CONFIRMATION`；tiling owner |
| C-04 | STC 014/015 的类别 | 014 写 5120 “不是 32 整数倍”；015 又写 `<128` | `5120/32=160`，整除且大于 128 | 不按文字自动生成；暂归 `gt_128_divisible` | `BLOCKED_BY_SPEC`；STC owner |
| C-05 | STC 能否直接作为 generator 输入 | 001-020 只给部分维度，021-023 仅写“复用”，多行数组长度错误 | 两算子原 Excel parser/golden 需要 dtype、N1/N2/D、mask、template、block 等完整字段 | 原 STC 只做 traceability；新增独立小 shape 离线预检表，不篡改原表 | `BLOCKED_BY_SPEC`；STC owner、pytest owner |
| C-06 | metadata 当前布局版本 | 旧资料按 FA `36x8` | PR8737 当前布局为 FA `36x9`、FD `72x8`、tail 124 | parser 严格按版本解析；禁止把两种布局猜成一种 | `CONFIRMED`（代码版本内）；tiling owner需确认发布包版本 |
| C-07 | level 0 能否代替 level 3 | level 0 可调用普通 metadata/op | batch consistency 要求 level 3 固定规约；本机 runtime 拒绝 level 3 | 返回 `UNSUPPORTED`，禁止降级后报 PASS | `CONFIRMED`；Runtime owner负责环境 |
| C-08 | SMLA是否已有level3专用固定规约路径 | `pr_9103`对共享`flash_decode.h`和SMLA staging offset有改动 | SMLA host/kernel未见`GetDeterministicLevel()==3`或`BATCH_CONSISTENCY` tiling-key；MQSMLA则两者都有 | SMLA允许做level3黑盒关系测试，但PASS scope不得声称命中专用路径 | `BLOCKED_BY_SPEC`；SMLA tiling/kernel owner |
| C-09 | MQSMLA入口应设置level 1还是3 | 算子仓旧`tests/pytest/batch/mixed_quant_sparse_flash_mla_process.py`仍设置1 | `pr_9103` host只在`GetDeterministicLevel()==3`时开启batch consistency，本仓pytest_v2也请求3 | 禁止用旧level1 batch入口验收；修复或删除旧入口歧义 | `TEST_GAP`；算子pytest owner |
| C-10 | 旧安装树能否代表`pr_9103` | `/home/Ascend/vendors/custom_transformer`可正常加载 | 旧树manifest为`81730e...7a5e`，新staging为`ea1113...379a4`且关键库hash不同 | 正式probe强制staging处于两个搜索路径首位并校验期望manifest；不覆盖系统安装 | `CONFIRMED`（本轮身份门禁）；构建owner仍需包内commit id |

## 3. 模式 4 与 mask 的完整数学

### 3.1 RightDownCausal/Band 的共同锚点

令：

- `K`：当前 batch 的有效 ori KV 长度；
- `Q`：当前调用的有效 query 长度；
- `i`：query 在当前调用中的 0-based 位置；
- `threshold = K - Q + i + 1`。

RightDownCausal 可见右边界由 `threshold` 决定。Band 在此基础上使用：

```text
start = max(0, threshold - win_left - 1)   # win_left=-1 时 start=0
end   = min(K, threshold + win_right)
```

因此，只有 `threshold`、`win_left`、`win_right` 和对应 KV 内容均不变，才可声称同一 token 的
mask 数学输入等价。当前 `ActualInputSemanticOracle` 不只检查公式，还逐 token 检查 Q、ori/cmp
窗口内实际 KV、sparse indices、top-k 和 sinks。

### 3.2 三种说法为何不同

| 构造 | 旧 token 新位置 | 新 KV | 新 threshold | 数学结论 | 需求结论 |
| --- | ---: | ---: | --- | --- | --- |
| 只在 Q 尾部追加 `delta` | `i` | `K` | `K-(Q+delta)+i+1 = old-delta` | 右边界左移，不等价 | STC“后面追加”若未同步 KV，不能验收 |
| 在 Q 首部 prepend `delta` | `i+delta` | `K` | `K-(Q+delta)+(i+delta)+1 = old` | threshold 可保持 | 数学可行，但比较 token 不再是两个输入的“共同前缀” |
| Q 尾部增长且 KV 同步增长 `delta` | `i` | `K+delta` | `(K+delta)-(Q+delta)+i+1 = old` | threshold 与共同前缀均保持 | 当前模式 4 暂定实现 |

当前实现为：

```text
new_ori_S2 = old_ori_S2 + new_Q_length - old_Q_length
```

有 cmp KV 时还执行：

```text
new_cmp_S2, new_cmp_residual = divmod(new_ori_S2, cmp_ratio)
```

并检查 PA block table 是否能寻址新增逻辑 KV。容量不足不是数值失败，而是
`INVALID_TRANSFORM`。对于 Band，只要 threshold 和 window 参数不变，start/end 都不变；若只改 Q
而不改 KV，则 start/end 会一起漂移，这正是后期必须找 mask owner 核对的风险。

### 3.3 需要需求/算子 owner 明确回答

1. 模式 4 的产品定义是否正式采用“共同前缀 + KV 同步延长”？
2. STC 004/009/013/017 的“后面追加”是否隐含 KV 同步追加，追加 KV 的值来自哪里？
3. 会议纪要的 prepend 是历史建议，还是必须保留的独立第五种验收模式？
4. cmp mask 的 effective length 是否始终由 `divmod(ori_S2, cmp_ratio)` 得出？
5. `ori_win_right > 0` 时，新增 KV 是真实未来 token 还是仅扩大容量，如何生成等价数据？

## 4. PR8737 规约公式冲突

2026-07-20 只读审议时记录到 PR8737 head `31a3b6d3513c90761d553a1e5004fc53f6825415`
的表达式：

```cpp
actTotalS2Size = actOriS2Size + actCmpS2Size;
reductionTileSize =
    (actTotalS2Size / 32 + s2BaseSize_ - 1) / s2BaseSize_ * s2BaseSize_;
```

C++ 整数除法使第一步为 floor。需求/原 analyzer 的预期是：

```text
align_up(ceil(actual_S2 / 32), 128)
```

二者不能用一句“向上取整”混过去：

| actual S2 | floor quotient | ceil quotient | PR floor tile | 需求 ceil tile | 影响 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 128 | 128 | 128 | 128 | 无差异 |
| 4097 | 128 | 129 | 128 | 256 | tile 和分核都可能改变 |
| 5120 | 160 | 160 | 256 | 256 | 无差异；STC 014 文字仍错误 |
| 5121 | 160 | 161 | 256 | 256 | quotient 不同，但本级对齐后 tile 暂同 |
| 1..31 | 0 | 1 | 0 | 128 | PR 表达式出现 zero tile edge，必须确认是否有前置 clamp |

代码报告字段：`pr8737_floor_quotient`、`requirement_ceil_quotient`、
`pr8737_floor_tile`、`requirement_ceil_tile`、`quotient_formula_conflict`、
`tile_formula_conflict`、`pr8737_zero_tile_guard`。在 owner 确认前，报告冲突而不篡改任一方定义。

## 5. STC 001-027 逐项台账

### 5.1 全表公共缺口

001-020 普遍没有完整的 `q_type/ori_kv_type/cmp_kv_type`、`N1/N2/D`、template、ori/cmp
mask、window、block size/table、明确 baseline id 和具体 batch/token ids。“ori+cmp s”“S2”“调度块
大小”的字段语义也没有 schema/version。021-023 没有参数行。故原表目前是验收设计草案，不是可直接
执行的 generator manifest。

### 5.2 每条用例

| STC | 原文/关键字段 | 问题与当前判定 | 需要补齐/修订 | 状态/待确认人 |
| --- | --- | --- | --- | --- |
| 001 | “调度块128整对比基准”；B=16,S1=2,ori+cmp=4096,S2=507904,cmp_ratio=128 | 数学类别可作为 `eq_128`，但 `ori+cmp` 与巨大 S2 的关系未定义 | 完整算子参数、每 batch effective ori/cmp、mask/top-k，明确 4096 是哪个 token 的 visible S2 | `BLOCKED_BY_SPEC`；STC+tiling owner |
| 002 | “挑选001中所有batch的第一个token”；B=16,S1=1 | 可表达为 token 子集，但不是 kernel 分核指令；没有声明 KV 是否逐 batch 复用 | baseline=001、token id=0、完整 tensor 复用规则 | `IMPLEMENTED_PENDING_CONFIRMATION`；STC owner |
| 003 | “挑选001中部分batch”；B=4,S1=2 | 模式 2 语义可行，但“部分”不可复现 | 明确 4 个 batch id 或 seed/version | `BLOCKED_BY_SPEC`；STC owner |
| 004 | “在003各batch后面追加token”；B=4,S1=5,seqused_q=[3,4,5,5] | 与主需求共同前缀/会议 prepend 冲突；未说明 KV 同步，RightDown/Band 可能改变 | 明确新增 Q/KV 数据、比较 token ids、mask；按第3章给出等价证明 | `BLOCKED_BY_SPEC`；需求+mask owner |
| 005 | “调度块大于128对比基准”；B=8,S1=4,ori+cmp=12288,S2=1556480 | `12288/32=384` 可归大块基线；其余参数缺失 | 同 001，明确 visible S2 与物理 S2 | `BLOCKED_BY_SPEC`；STC+tiling owner |
| 006 | “从005选取1个batch的1个token”；B=1,S1=1 | 可做 mode 2/3 子集，但 id 未指定 | baseline=005、batch id、token id | `BLOCKED_BY_SPEC`；STC owner |
| 007 | “选取所有batch的前x个token”；B=3,S1=4,seqused_q=[2,3,2] | “所有batch”若指005应为8，但字段只有3；x 也不是一个值 | 改成指定3个 batch，或把 B/数组修为8；列出 token ids | `BLOCKED_BY_SPEC`；STC owner |
| 008 | “从005挑选其中2个batch”；B=2,S1=4 | 模式2可行，但 batch ids 未指定 | baseline=005、两个 batch id | `BLOCKED_BY_SPEC`；STC owner |
| 009 | “在008各batch后追加token”；B=2,S1=8,seqused_q=[5,8] | 与004同一 mask/共同前缀冲突，且重复写“校验精度” | 按004修订 | `BLOCKED_BY_SPEC`；需求+mask owner |
| 010 | “S2_b/32小于128”；ori+cmp=1152,K=1024,cmp_ratio=4 | `1152/32=36` 类别成立；但 top-k/mask 后 visible S2 是否仍1152不明 | 拆出 ori/cmp effective length、top-k 后可见值 | `BLOCKED_BY_SPEC`；STC+tiling owner |
| 011 | “从010挑选数个batch”；B=2 | 模式2可行，但 ids 未指定 | baseline=010、batch ids | `BLOCKED_BY_SPEC`；STC owner |
| 012 | “010中每个batch第一个token”；B=16,S1=1 | 可表达，需明确 exact reuse 和 origin | baseline/token ids/完整字段 | `IMPLEMENTED_PENDING_CONFIRMATION`；STC owner |
| 013 | “在011后面拼接不同token”；B=2,S1=4,seqused_q=[3,4] | 与004相同；“不同token”还可能直接改变被比较 Q | 明确共同 token、追加数据和 KV/mask 联动 | `BLOCKED_BY_SPEC`；需求+mask owner |
| 014 | “5120大于128但是不是整数倍”；ori+cmp=5120,调度块=256 | `5120/32=160`，是整数且大于128；类别文字错误 | 改为 `gt_128_divisible`，或把 S2 改成真正有余数的数 | `BLOCKED_BY_SPEC`；STC owner |
| 015 | “从014挑选数个batch，S2_b/32小于128”；ori+cmp=5120 | 继承014却写 `<128`；实际160>128 | 修正文案并列 batch ids | `BLOCKED_BY_SPEC`；STC owner |
| 016 | “014中每个batch第一个token”；B=16,S1=1 | 子集意图可行，但继承014的分类冲突 | 先修014，再补 origin | `BLOCKED_BY_SPEC`；STC owner |
| 017 | “在015后面拼接不同token”；B=2,S1=4 | 与004/013相同，并继承014/015冲突 | 先修014/015，再给模式4等价证明 | `BLOCKED_BY_SPEC`；需求+STC owner |
| 018 | “打乱batch顺序”；B=8,S1=4,seqused_q=[1,2,3,4]，cu_seqlens_q列有8个大数 | `seqused_q` 数量应为8；所谓 cu 值既非 B+1、非单调、也像 KV 长度 | 把8个大数移到明确的 KV 字段；补8项 Q length或B+1 prefix sum | `BLOCKED_BY_SPEC`；STC owner |
| 019 | “打乱token顺序”；B=1,S1=16,cu_seqlens_q=[638976] | TND prefix sum 应为 `[0,16]`；BSND通常无需该值；638976显然不是 Q prefix | 修列归属，并明确 token permutation 是否允许位置 mask | `BLOCKED_BY_SPEC`；STC+mask owner |
| 020 | “打乱各batch间token随机组合”；字段同018 | 与018相同，且跨 batch token 重组会改变位置 mask/对应 KV，不能仅重排 Q | 给出每个 token 的 Q/KV/mask origin 映射和合法性约束 | `BLOCKED_BY_SPEC`；需求+STC owner |
| 021 | “复用上述用例，aclnn直调” | 没有 case 集、接口参数、输出约定 | 列出复用哪些 ID、ACLNN API/version、metadata注入方式 | `TEST_GAP`；STC+ACLNN owner |
| 022 | “复用上述用例，torch直调” | 没有 case 集；当前 pytest_v2 eager 属于此方向但不能替表中定义 | 列出 ID/profile、预期状态和报告路径 | `BLOCKED_BY_SPEC`；STC owner |
| 023 | “复用上述用例，acl_graph” | 没有 case 集，graph capture 下 level3/metadata 生命周期未定义 | 列出 ID、capture边界、metadata是否同次调用 | `TEST_GAP`；Graph+STC owner |
| 024 | “N=125随机组合”；B=8,seqused_q只有4项 | N=125 只在文字中，没有 N1/N2 结构字段；数组长度错误 | 明确 N 指N1还是N2，补N1/N2/G及8项长度 | `BLOCKED_BY_SPEC`；STC owner |
| 025 | “N=3随机组合”；B=8,seqused_q只有4项 | 同024 | 同024 | `BLOCKED_BY_SPEC`；STC owner |
| 026 | “TND随机组合”；B=8,S1=4,seqused_q 4项,cu=[0,3,6,9,15] | cu只描述4段且最后一段长6>S1=4；长度不是B+1 | 给出9项合法prefix sum、8项effective Q和T1 | `BLOCKED_BY_SPEC`；STC owner |
| 027 | “PA_BBND随机组合”；B=8,seqused_q只有4项 | 缺block size、block num、ori/cmp block table和容量约束 | 补齐PA物理/逻辑映射与8项Q长度 | `BLOCKED_BY_SPEC`；STC owner |

## 6. 对齐会议建议议程

建议不要逐句散聊，按以下顺序一次关闭：

1. 先由需求 owner 裁决 C-01（模式4）和模式3是否允许位置敏感 token 任意重排；
2. tiling owner 用一条实际 C++/UT 确认 C-02/C-03，包括 `S2<32`；
3. STC owner 修 014/015、018-020、024-027 的确定性错误；
4. pytest owner提供 versioned manifest schema，原 Excel保留历史、不原地猜改；
5. ACLNN/Graph owner补 021/023 的明确 case profile 和调用边界；
6. 所有确认项记录 owner、日期、commit/custom包版本后再从本表移入 `CONFIRMED`。
