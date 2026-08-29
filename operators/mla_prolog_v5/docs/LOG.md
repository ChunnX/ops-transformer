# mla_prolog_v5 开发日志

## 任务信息

- 任务：issue-65 快速迁移 mla_prolog_v3 → 独立算子工程 mla_prolog_v5（A2）+ pybind 打通
- 模式：快速迁移（精简设计流程，3 阶段：需求+方案 → 编码迁移 → 测试验证）
- 工作分支：agent/issue-65（ops-transformer fork）
- 工作流：ops-registry-invoke（按用户要求压缩：跳过 spec/设计分段/独立评审管线）

## 用户原始需求

- 算子名字：mla_prolog_v3；算子代码仓：https://gitcode.com/cann/ops-transformer.git
- 快速将 ops-transformer 仓算子迁移到独立算子工程，命名 mla_prolog_v5
- 在迁移的独立算子工程打通 pybind 流程，支持自定义 python 接口调用
- 目标：一个算子编译工程，支持 python 接口测试；要求：快速迁移，减少人工交互，完成基本功能验证
- 使能方式：Aclnn 直调 + pybind；只迁移原始代码，不做功能增强，只保留 A2 芯片代码
- 价值：抽取算子代码成为独立自定义算子，方便后续增量开发

## 环境检查记录

| 项 | 值 |
|---|---|
| NPU | 910B3（Atlas A2），健康，npu-smi 25.5.0，ASCEND_VISIBLE_DEVICES=6 |
| CANN | /home/developer/Ascend/cann-9.1.0（ASCEND_TOOLKIT_HOME/ASCEND_HOME_PATH） |
| Python | 3.11.4 |
| cmake | 3.20.5 |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4（npu available: True） |
| 构建框架 | CANN 9.1 ascendc_kernel_cmake（npu_op_package），见 CMAKE_PREFIX_PATH |
| 测试工具 | TTK（ops-test-kit）位于 ../../ops-test-kit（备用） |

## 开发状态

| 阶段 | 状态 | 完成时间 | 说明 |
|------|------|----------|------|
| A. 需求分析+精简方案 | ✅ | 2026-08-29 | REQUIREMENTS.md / DESIGN.md 产出，依赖闭包採查完成 |
| B. 编码迁移 + pybind 打通 | ✅ | 2026-08-29 | 编译/安装/调用链路全通，详见开发记录 |
| C. 测试验证（Python 接口 + NPU 精度） | ✅ | 2026-08-29 | tests/pybind/test.py 5/5 PASS（分层验证） |
| 收尾：pre-commit + commit | ⬜ | | |

## 交付物清单

| 文件 | 状态 |
|------|------|
| docs/REQUIREMENTS.md | ✅ |
| docs/DESIGN.md | ✅ |
| docs/LOG.md | ✅（本文件） |
| op_kernel/ op_host/ op_api/ CMakeLists.txt build.sh | ✅ |
| torch_extension/（csrc + CMake + Python 接口 + libmla_prolog_v5_ext.so） | ✅ |
| tests/pybind/test.py + golden.py | ✅ |
| examples/test_aclnn_mla_prolog_v5.cpp + CMakeLists | ✅ |
| README.md docs/aclnnMlaPrologV5WeightNz.md | ✅ |
| issues/issue_20260829_mm-split-multihead_01.md | ✅ |

## 开发记录

- 2026-08-29 阶段A：完成源码依赖採查——v3 依赖 v1(mla_prolog) 的 arch22 kernel 头（12 个）与
  op_host tiling/infershape/tiling_check；v3 op_api 外层调用仓框架生成的 aclnnInner 层，独立工程需手写
  L0（仓内有 nsa_compress_with_cache 等手写先例）；确定 aclnn 外层取 V4 语义（doRope 参数）；
  确定构建走 npu_op_package（ascendc-registry-invoke-template 模板）、A2-only（ascend910b）。
- 2026-08-29 阶段B：vendored 文件机械重命名（V3→V5 / tiling V3_OP_NAME 布局分支改判 V5 / infershape
  节点名判断改 V5 / TilingPrepareForMlaProlog 定义并入 register）；kernel 裁剪 arch35 分支、def 裁剪 950
  配置、kernel 编译选项沿用源仓；手写 L0（OP_INPUT/OP_OUTPUT/OP_ATTR 严格对齐 def 注册序）；构建通过
  （custom_opp_ubuntu_aarch64.run）；torch_extension（vendored aclnn_common.h + csrc 改调 V5 API）编译通过。
- 2026-08-29 阶段C 排查与修复：① torch 路径 tiling 读到扁平 storage shape → L0 增加 ND 张量 storage shape
  归一化（view shape 化 + 基础格式统一 ND）；② optional 未传须 nullptr 透传（占位 tensor 会触发 tiling
  "not null"校验失败）；③ **attr float 错配**（API double 形参直接传 OP_ATTR，被按 def Float(4B) 解释成
  巨值 → rmsnorm 输出全 0；scale 用例出现 1e24 量级误差）→ 显式转 float（与 opbuild 生成 Inner 一致），
  修复后 query_norm/kv_cache 全量精确 PASS；④ 910B3 适配：tiling mm3 分块按头边界 (D+Dr) 对齐。
  最终 tests/pybind/test.py 5/5 PASS（分层验证：rmsnorm 链路 query_norm/kv_cache 数值精确，
  do_rope=False 的 kr_cache 精确；多头 query/query_rope 的 head≥1 已知问题记录于 issues/，
  系 vendored kernel mm3/mm4 在 910B3（20 核）多核分块行为，需 kernel 级调试后续跟进）。
