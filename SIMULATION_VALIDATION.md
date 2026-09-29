# Shared-start 仿真准备记录

目标：BS128 / H7168 / GMM1输出4096 / topk6 / EP8 / 每卡3个MoE+1个共享专家 / 28AIC+56AIV。

- Baseline source: `56b62be90`；optimized source: `6c8a5eddc`。
- Imported generator: `macech/mega-moe-sim@b15137e4`；未迁入其kernel实验改动。
- 本地使用CANN 9.2.0，Python3.13，依赖缓存隔离；编译并发8。
- 真实host编译使用主线固定ops-tensor提交 `eb28bc21b8c8ac95bd527f5035d1cbf216c2ebce`。
- 目标topk权重为FP32，旧bs72默认BF16仍可用；新负例验证dtype误配会被拒绝。
- 每专家收256个token，每卡768；host最大容量3072，每卡workspace101672960B、window12093440B（含异常dump前缀）。
- tiling为440B，context16400B，dedup=0，wave m groups=7，dispatch buffers=6，L1 A/B buffers=3/4。
- 两版没有添加同步、早退、额外UB/GM，只添加COND Scalar标记；独立空MoE入口边界只用于基线标记。
- 单次启动，通信窗口和workspace初始化为零；不复现实机预热后的cache/轮次状态。
- kernel bin尚未编译，debug npusim尚未运行；input-only目录不能直接运行。
- 完整数值精度未验证，CPU校验不替代设备正确性验证。

步骤与命令见 [操作说明](sim/README-shared-start-ab.md)。

## 本地完成结果

- 两版各16项CPU测试通过，19项COND标记静态配对通过。
- 两版host helper已实际构建（-j8），完整8卡input-only生成及校验通过。
- A/B比较：144个输入bin的SHA256一致，config/layout/memory_map/routing一致。
- 当前编译器生成参数中FP32目标签名唯一匹配：`MegaMoe_e18a9de48d39934db95dedf2ac6ff645_param.json`。
- 本分支输入：`sim_cases/shared-start-bs128-final-inputs`（`sim_cases/latest`指向它）。
- 本地只生成输入，不包含可启动kernel；服务器必须使用兼容CANN重新build并attach。

- 上述host构建和完整输入生成完成于许可证头及格式整理前；源码指纹因此变化，服务器需重新构建，不能复用旧构建清单。
