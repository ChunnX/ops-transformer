# grouped_matmul_swiglu_quant_v2 MxA8W4算子tiling部分测试验证及实现完善

## 任务说明

该任务要求你完成对tiling部分的测试验证，并进行代码修复和完善，你需要遵循测试驱动开发（TDD）的准则完成该任务。该任务核心目标是完成grouped_matmul_swiglu_quant_v2算子伪量化Mx8W4数据流的tiling拦截校验和切分逻辑的实现，对于该tiling部分代码的实现你可以使用算子的UT测试框架进行初步验证。下面是详细的工作安排：

- 读取agent工作流元文档理解项目情况。
- 根据[STC算子用例设计文档](./raw/development/stc_gmmsq.md)设计伪量化MxA8W4场景的UT用例，UT的代码在gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_host/test_grouped_matmul_swiglu_quant_v2_tiling.cpp中，请根据该代码逻辑支持新数据流的UT验证流程，并补充相关测试用例进行验证。注意该文档中的打桩机制并非真实有效，你需要按照当前的代码框架验证tiling部分的UT，不需要关注kernel部分的UT。
- tiling编码方案主要参考grouped_matmul_finalize_routing算子的伪量化MxA8W4数据流，这两个算子比较类似，代码可以进行参考。关键代码在gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h，而grouped_matmul_swiglu_quant_v2总体的编码方案在gmm/grouped_matmul_swiglu_quant_v2/.agent/raw/development/srs_coding_design.md中，如果有需要可以查看。
- 你需要先补充UT用例，对grouped_matmul_swiglu_quant_v2算子的tiling部分验证全部通过，才算任务完成。
- 最后根据STC补充ST用例，主要用于用户后续自行通过ttk测试框架进行集成测试，ST用例设计可以参考[ST_design_summary.md](raw/development/st_design/ST_design_summary.md)文档。关于生成ttk用例的方法请参考/home/sunduiyang/Code/gen_cases项目来生成用例，阅读其中的文档或项目来了解用法。

> 注意：为了完成这个复杂的任务，请进行编排规划，通过subagent进行具体任务的执行。

下面是编译运行UT的方法：

``` shell
# 编译运行
bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul_swiglu_quant_v2 -j12 --noexec
BUILD_PATH=$(pwd)/build ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut

# 开启日志输出，获取详细报错信息（日志很长，建议重定向到文件后再读取分析）
ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=0 BUILD_PATH=$(pwd)/build ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut
```

## 任务完成状态

### Tiling部分 - 🟡 进行中
- [x] 补充MxA8W4 CSV测试用例（8个用例：正常、边界、非法场景）
- [x] 完善tiling校验逻辑：CheckGroupNum()（E<=1024）、CheckScaleShapes()（wscale 4D/xscale 3D校验）
- [x] 编译验证tiling代码（ophost编译通过）
- [ ] 编译运行ut程序进行测试验证，直到所有用例通过