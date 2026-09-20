# Arch22 迁移验证记录

日期：2026-09-20。仅 A2 910B3 上板，CANN 9.2.0_0902。

## 实现边界

- A2/A3 共用 arch22 FullLoad/Perf；A5 原 arch35 计算文件不修改。
- V2 无APE，固定coff=1和cacheMode=2，不迁移训练输出及线性cache。
- state按V2语义保存本次输入末尾min(seqUsed, block_size)个token，包括已压缩token；block0有效。
- 保留有效Split-K上限，BSH容量及state stride检查；TH不在Host读取设备长度值。
- 图模式及batch一致性不作为本次A2验收内容。A3/A5未上板，不将注册成功表述为实测通过。

## 已完成验证

| 验证 | 结果 |
| --- | --- |
| 独立普通A2包编译安装 | 通过 |
| A2 Host UT | 18/18，通过模板编号、空输入、参数负例及Split-K分片完整性检查 |
| 原生ACLNN样例 | 输出、全量state及未更新区域通过 |
| V2现有25项 + 新增114项pytest联合回归 | 139/139通过 |
| 新增连续state对照 | 4/4通过 |
| 普通包统一回归（现有25项 + 新增118项） | 143/143通过，45.83秒；JUnit failures/errors/skipped均为0 |
| 原生ACLNN非连续state内存检测 | FullLoad、Perf及NCL/flat-storage三项对照ERROR=0 |

Host UT命令：

```bash
bash build.sh --ophost_test --experimental --soc=ascend910b --ops=compressor_v2
```

精度回归命令（在 `tests/pytest` 下，使用普通包而非sanitizer包）：

```bash
python3 -m pytest -v test_compressor_single.py test_compressor_arch22.py
```

输出使用现有V2 Golden和判定标准，完整FP32 state另外检查rtol=1e-4/atol=1e-5；guard和未更新区域精确比较。

## 未解决问题

**MEMCHECK-001：PyTorch非连续state的4条illegal-write ERROR。**

sanitizer,ccec_g包实际拦截了FullLoad和Perf。尽管精度、guard和独立原生对照通过，PyTorch路径仍有4条ERROR；不能把ERROR等同于寄存器未复位Warning而忽略。
已证实FullLoad报错地址处于实际backing分配范围；最终根因尚未确定。整块backing拷回CPU的尝试没有消除ERROR。
按用户要求，此项延期到其他迁移步骤完成后处理；延期不代表安全验收通过。正式交付前需修复并重跑memcheck和普通包回归。

普通包完成两轮四场景V1/V2 kernel耗时对比：decode两轮均更快，prefill D128增加约0.6%至4.9%，D512变化约-1.7%至+1.8%。V2写回state语义不同，不视为相同工作量基准，不外推全部形状。原始CSV、日志和结果以本地迁移进度及性能报告为准。

代码整理及最终交付复核尚未完成，未完成项不计作通过。
