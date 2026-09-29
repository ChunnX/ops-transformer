# 独立 CV/VV 最小协议

每个 AIV 执行 N 次 `Set<2,MTE3>(0) → Set<0,MTE2>(VV) → Wait(VV)`；
每个 AIC 仅执行 N 次 `Wait(0)`。`same` 的 VV=0，`split` 的 VV=15。
不做数值计算，不添加 SyncAll、ACK、延迟或设备打点。

`cann_mix_wrapper_shim.h` 仅供这个不用 Matmul/KFC/workspace 的编译单元使用，
去掉 CANN 自动 MIX wrapper 的 `clearWorkspace` 隐式 flag 15 通知。
**不能把这个 shim 复制到真实依赖 Matmul/KFC 的计算算子中。**

配置文件在独立构建目录生成；编译后反汇编核对 Cube 无 Set、AIV 每轮恰好两个 Set。
源码顺序不代表不同管线的完成顺序，也没有要求 Cube 的消费先于 VV 会合。

操作说明见 [仓库复现入口](../../../../README.klloss.md)；
实测和原因见 [客户说明](../../../../docs/klloss_sync/README.md)。
