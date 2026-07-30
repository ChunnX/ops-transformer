# QuantLightningIndexerV2 问题定位与解决报告

## 1. 背景

本次排查针对 Ascend 950 环境下 QuantLightningIndexerV2 的 MXFP8/MXFP4 redline 用例，重点覆盖：

- TND query + PA_BBND key；
- BSND query/key；
- sparse mode 0/3；
- return_value 开启与关闭；
- seed 0、1、2 三组随机数据。

代码分支：qliv2-pr8964-dump。

测试环境：

- 容器：782ee3866806；
- Conda 环境：shy39；
- CANN：9.1.0；
- SoC：Ascend 950；
- 安装目录：/home/s00892993/custom_path。

## 2. 初始现象

代表性失败用例为 quant_li_mxfp8_redline_tnd_pa_mask3。初始测试同时存在索引和返回值精度问题：

- PA_BBND 路径的索引集合明显偏离 golden；
- 修复 PA 地址后，索引比较通过，但返回值满足率仍约为 87%；
- 返回值错误呈稳定的行间交替分布：同一个 AIV 连续处理两行时，第一行容易失败，最后一行通常正确；
- 错误值可达到 1e8～1e9 数量级，与正确 BF16 score 不在同一数量级；
- 开启目标行 DumpTensor 后，被选中的目标行会恢复正确，而未被选中的同类行仍失败。

这些现象表明问题不只来自数值精度，还包含 PA 地址计算和异步流水同步两个独立缺陷。

## 3. 定位过程

### 3.1 PA_BBND 地址问题

通过小规模交叉测试比较 TND key 与 PA_BBND key：

- 相同数据在 TND key 路径通过；
- PA_BBND key 路径失败；
- eager 调用中动态 key stride 可能为空；
- tiling 在 stride 信息为空时将 keyStride0 保持为 0；
- PA 内核使用 physicalBlkId × keyStride0 定位物理块，因此所有物理块实际退化为读取 block 0。

同时检查 key dequant scale 的连续布局，确认 PA 场景的 fallback stride 需要包含 N2 维。

### 3.2 返回值错误

PA stride 修复后进行了逐行统计和 CPU/NPU dump 对齐：

- query scale 与 block table 完全一致；
- 目标行 CPU/NPU effective score 完全一致；
- TopK score key 与按索引计算的期望值完全一致；
- 改变 QLIV2_DEBUG_TARGET_S1_IDX 后，恢复正确的行会跟随 debug 目标移动；
- 非 LD 路径中 valueOutLocal_ 与 mrgValueLocal_ 指向同一个 mrgValueBuf_；
- 上一行通过 MTE3 将 valueOutLocal_ 搬到 GM，下一行通过 MTE2 将新的 TopK 输入写入同一块 UB；
- 原代码只有 MTE3→V 事件，无法阻止 MTE2 在上一行 MTE3 完成前覆盖共享 UB。

因此，返回值异常的根因是相邻行之间缺少 MTE3→MTE2 同步，而不是 score 计算、TopK 排序或 BF16 转换算法错误。

## 4. 修复方案

### 4.1 补齐 PA 连续布局 stride

文件：

attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp

修改：

- PA_BBND 且动态 stride 缺失时，按 blockSize × n2Size × physicalHeadDim 计算 keyStride0；
- MXFP4 场景考虑两个 FP4 元素打包为一个字节；
- MXFP8/MXFP4 key dequant scale fallback stride 增加 N2 维；
- 非 MX、非 HiFloat8 的 PA token scale 使用 blockSize × n2Size。

### 4.2 增加相邻行 MTE3→MTE2 同步

文件：

attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_vector.h

修改：

- 非 LD 路径处理同一个 AIV 的后续行前，设置并等待 MTE3_MTE2_EVENT；
- 确保上一行返回值完成 GM 搬出后，下一行 MTE2 才能复用 mrgValueBuf_。

### 4.3 默认关闭内核 dump

文件：

attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2.cpp

修改：

- 移除源码中强制定义的 ASCENDC_DUMP=1；
- 移除源码中强制定义的 QLIV2_DEBUG_DUMP；
- 普通构建默认不产生 DumpTensor 输出；
- 如需再次诊断，可在专用 debug 构建中显式定义这两个宏。

调试目标宏仍保留在 common 头文件中，当前目标为 B=0、N2=0、S1=0、QK S2 begin=0，仅在显式启用 QLIV2_DEBUG_DUMP 时生效。

### 4.4 扩充 redline 回归

文件：

- attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_paramset.py；
- attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_single.py。

修改：

- 启用 5 组 MXFP8/MXFP4 redline 参数；
- single 测试增加固定随机种子 0、1、2；
- 同时设置 Python random、NumPy 和 PyTorch 随机种子；
- 测试实例总数为 5 × 3 = 15。

本分支还包含 torch_extension/cann_ops_transformer/ops/__init__.py 的测试环境隔离改动，用于避免加载当前算子测试不依赖的扩展模块。

## 5. 构建与安装

执行命令：

~~~bash
bash build.sh --pkg --soc=ascend950 \
  --ops=quant_lightning_indexer_v2,quant_lightning_indexer_v2_metadata

./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --install-path=/home/s00892993/custom_path

source /home/s00892993/custom_path/vendors/custom_transformer/bin/set_env.bash
~~~

最终无 dump 算子包 SHA256：

9c0bc2008f0ac4f5c52da7a9ed3e3d2d73991d197f99f36f29902caf84d682fa

## 6. 回归结果

测试命令：

~~~bash
cd /home/s00892993/ops-transformer-qliv2mxfp4/attention/quant_lightning_indexer_v2/tests/pytest
bash test_run.sh single | tee redline_debug.log
~~~

最终结果：

- 15 passed；
- 0 failed；
- 26 warnings；
- pytest 退出码 0；
- 总耗时 617.30 秒；
- DumpTensor 输出 0 行；
- 日志大小约 100 KB。

26 条 warning 来自包含 -inf 数据时的 NumPy subtract RuntimeWarning，以及 pandas DataFrame 拼接 FutureWarning，不影响精度判定与测试结果。

## 7. 结论

本次问题由两个独立缺陷叠加导致：

1. PA_BBND 在缺少动态 stride 时错误使用 0 stride，导致物理块寻址错误；
2. 非 LD 返回值缓冲区与 TopK 输入缓冲区复用时缺少 MTE3→MTE2 同步，导致上一行返回值被下一行覆盖。

补齐 PA fallback stride 和行间流水同步后，索引与返回值比较均通过；关闭内核 dump 后，5 组 redline 参数在 3 个固定随机种子下共 15 个实例全部通过。
