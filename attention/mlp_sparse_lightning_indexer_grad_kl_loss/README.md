# MLP Sparse Lightning Indexer Grad KL Loss：同步诊断版本

**不计算 KL Loss / 梯度，不能用于训练。** 输出由 torch 适配层初始化为零。

- `op_kernel/`：before 对齐的确定性纯同步流水，默认保留原 flag 分配。
- `experiments/cv_vv_minimal/`：独立最小协议；不被默认 kernel include。
- `op_host/`：形状检查、tiling、全核 mixed 调度；无外部 opbase 源码依赖。
- `torch_extension/`：遵循本仓库算子内适配目录规范，可直接由复现脚本加载。
- `examples/`：B=1、TND、Q=KV=1024、BF16、topK=2048、确定性测试。

编译和运行从仓库根目录执行，见 [一键入口](../../README.klloss.md)。
完整解释、流水图和复现局限见 [客户说明](../../docs/klloss_sync/README.md)。

默认同步源码迁移时保持原有执行逻辑：V0/C1、V1/C2、V2、SyncAll 与 flag 14 等均保留。
最小实验不含这些附加同步，配置通过生成目录内的头文件选择，不改默认源码。
默认 config 中保留的历史延迟/ID 隔离选项不属于客户验收矩阵；尤其 flag 15 会受默认
MIX/KFC wrapper 的隐式通知影响。需要清晰 A/B 对照时使用独立 minimal，而不是随意启用这些选项。

诊断构建路径是 `scripts/klloss/compile.sh`，已验证完整 Host/ACLNN/kernel 源码构建。
顶层框架 `build.sh` 的其他算子、联网依赖及其他 SoC 不在本次验收范围。
