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
| 最终无插桩普通包统一回归（现有25项 + 新增118项） | 143/143通过，37.30秒；JUnit failures/errors/skipped均为0 |
| 原生ACLNN非连续state内存检测（未启用lite_exception的历史对照） | FullLoad、Perf及NCL/flat-storage三项对照ERROR=0；启用lite_exception后复现下述已知问题 |
| 补充sanitizer矩阵（外部验证脚本） | 16/16精度及state检查通过；原始日志仍有8条ERROR，不能按pytest通过判定安全全通过 |
| 单独重跑上述12项连续state安全对照 | 12/12通过，63.95秒，完整memcheck日志ERROR=0 |

Host UT命令：

```bash
bash build.sh --ophost_test --experimental --soc=ascend910b --ops=compressor_v2
```

精度回归命令（在 `tests/pytest` 下，使用普通包而非sanitizer包）：

```bash
python3 -m pytest -v test_compressor_single.py test_compressor_arch22.py
```

输出使用现有V2 Golden和判定标准，完整FP32 state另外检查rtol=1e-4/atol=1e-5；guard和未更新区域精确比较。

## 已定位工具问题

**MEMCHECK-001：CANN DFX非连续state边界描述导致的误报。**

sanitizer,ccec_g包实际拦截FullLoad和Perf。PyTorch默认启用lite_exception，原生ACLNN使用同样配置后亦复现；此前原生0 ERROR不能证明相同DFX配置下无报错。

实际读取到state输入/输出DFX边界均为逻辑Numel字节数。FullLoad逻辑大小2097152字节，合法stride跨度2098176字节，storage大小2099200字节；两次512字节写入均处于合法跨度内。aclTensor的shape/stride/storage描述正确。

在独立诊断进程中，仅修正两个DFX边界字段即可消除FullLoad和Perf报错；将FullLoad合法跨度缩短512字节，精确恢复score尾部一条ERROR。这些诊断修改不进入正式代码，也不用于宣称原始工具链安全验收通过。

本结论仅适用于已复现的state stride0 padding、有效地址位于真实storage内、且报错跨越dense logical end的相同模式。其他ERROR仍需单独定位。不得关闭lite_exception、跳过非连续用例或放宽精度判定来规避问题。正式工具链修复应由对应CANN/opbase版本修正DFX边界计算，当前未交付该修复。

普通包完成两轮四场景V1/V2 kernel耗时对比：decode两轮均更快，prefill D128增加约0.6%至4.9%，D512变化约-1.7%至+1.8%。V2写回state语义不同，不视为相同工作量基准，不外推全部形状。原始CSV、日志和结果以本地迁移进度及性能报告为准。

143项普通回归中，按实际state.is_contiguous()口径为123项非连续、20项连续；不能把单batch的padding请求误计为非连续。143项不是143项memcheck检测。

补充16项包括12项连续state（FP16/BF16、BSH/TH、短序列/长序列/ratio128）和4项非连续state（FP16/BF16、FullLoad/Perf）。原始检测有4条512字节和4条2048字节state尾部写入ERROR，与已定位模式一致。外部脚本和完整日志归档，不通过修改测试阈值或关闭DFX规避。

调测printf、桥接诊断分支及DFX hook不属于交付实现。功能实现基线为fb2b6776a；最终无插桩普通包SHA256为d411ff8bbef01cda9d94cd26ab3689c97ef42f445cdda9242e02d11a0aa600f2。最终回归和补充安全检测以本地归档报告为准。
