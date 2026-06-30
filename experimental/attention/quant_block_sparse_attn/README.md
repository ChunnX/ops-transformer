# quant_block_sparse_attn

TODO: 待实现 - 当前目录完成 quant_block_sparse_attn 的代码结构、FIA fullquant GQA 本地化迁移骨架和 pytest 测试框架迁移。

关键约束：

- 不直接 include `/home/dyx/ops-transformer/attention/common`，已将复用头文件复制到 `op_kernel/arch35/common` 与 `op_kernel/arch35/vf`。
- `op_kernel/arch35/quant_block_sparse_attn_*.h` 已完成自包含 include 路径和主要类名改名，但 BSA 特有流水逻辑仍需按 `/home/dyx/BSA方案2.4.md` 实现。
- `tests/pytest` 保留参考测试框架结构，golden、paramset、执行包装中不确定的 BSA 逻辑均以 TODO 标注。

待实现重点：

- kernel 主循环接入 `sparseSeqLen` tail-drain 保护、`sparseIndices` 到 actual KV token offset 的映射和空 sparse 输出写回。
- Vec1 按行应用 `q_descale[b, s1, n1] * k_descale[0] * softmax_scale`，BMM1 不复用 FIA block 级 Q deq scalar。
- BMM2/Vec2 接入 `v_descale[0]`。
- `quant_block_sparse_attn_attenmask.h` 将 compress mask 的 S2 offset 切到 `runInfo.sparseS2TokenOffset`，并覆盖 LEFT_UP / RIGHT_DOWN / TND / left padding 组合。
- pytest golden 按 BSA 接口重写输入生成、稀疏 reference、LSE 和非法 sparseIndices 用例。
