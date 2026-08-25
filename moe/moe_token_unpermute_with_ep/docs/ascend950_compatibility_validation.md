# MoeTokenUnpermuteWithEp Ascend 950 适配验证记录

## 适配范围

- 基线提交：`a2b38e196f2e0699494beb201df88cfdc7d2b267`（`origin/master`）。
- 验证分支：`codex/moe-token-unpermute-with-ep-ascend950`。
- 目标产品：Ascend 950PR/Ascend 950DT。
- 保持既有算子原型、dtype、属性默认值及 Ascend 910B、Ascend 910_93、Kirin 实现不变，仅增加 Ascend 950 注册、构建入口和测试资产。

## 实现说明

- Ascend 950 OpDef 配置通过 `opFile.value` 路由到 `moe_token_unpermute_with_ep_apt`。
- ACLNN 继续使用仓内既有公开接口；实机 plog 已验证 OpDef 路由能直接加载 `_apt` kernel，因此无需增加 `IsRegbase()` 分支或另造接口。
- Ascend 950 kernel 入口复用既有 LocalTensor 计算实现，保留原有 tiling key 和 FP16、BF16、FP32 组合，不复制或改写旧平台 kernel。
- 增加 Ascend 950 binary 配置、simplified key 配置和 CI 构建清单。
- ACLNN 将空的 `rangeOptional` 编码为默认 `[0, 0]`；tiling 在该状态下按接口约定忽略 `numTopk`，并回退到全输入 `MoeTokenUnpermute` 语义。
- Golden 仅使用 PyTorch 基础算子实现 gather、过滤、乘法和归并，不使用 NumPy，也不调用被测算子。

## 验证结果

| 验证项 | 命令或证据 | 结果 |
|---|---|---|
| Golden 语法与独立自检 | `/home/z50058944/venv/bin/python -m py_compile tests/assets/golden.py` 及固定输入自检 | 通过 |
| TTK CSV 校验 | `validate_ttk_csv.py --check-tensor-list`、`ttk aclnn --validate` | 16 条全部有效 |
| Host UT | `bash build.sh --ophost_test --ops=moe_token_unpermute_with_ep --soc=ascend950 --incremental` | 29/29 通过 |
| Ascend 950 定向打包 | `bash build.sh --pkg --soc=ascend950 --ops=moe_token_unpermute_with_ep` | 通过，生成 3 个 kernel object 及对应 JSON/config；run 包 SHA256 为 `bc986b709d6823ea80186a6cba8d5a9e20c39181431a8b7dc936f903314b7c18` |
| 隔离安装与加载确认 | 将本分支 run 包安装到独立 vendor 目录；plog 检查 kernel 路径 | 加载 `moe_token_unpermute_with_ep_apt/MoeTokenUnpermuteWithEp_07a82a6f9adc4f870c026d93f15d95bd.o` |
| TTK 单例冒烟 | Ascend950DT_9592，FP16 tokens + BF16 probs，TopK=2，range=[2,6] | 1/1 通过 |
| TTK ACLNN 全量 | Ascend950DT_9592，provided Torch golden，单设备串行执行 | 16/16 通过，0 失败，0 跳过 |

隔离 TTK 安装目录与最终 run 包中的 3 个 kernel object SHA256 完全一致：

- `MoeTokenUnpermuteWithEp_07a82a6f9adc4f870c026d93f15d95bd.o`：`a804b2263a70b7d0c98d2eb8bd96da6b8caff32aef536afbbcb123322be37a8f`。
- `MoeTokenUnpermuteWithEp_083278be2edfe5ef065870166a17808c.o`：`33abefef135c48bd54be1c30991847f8110e09145d59f6ea64c7e655c52a0978`。
- `MoeTokenUnpermuteWithEp_cca216df705a94b2276b63de9cffcad7.o`：`c891476ad7d6c6dbd060ec5ec9804db6c3c1b71e540df2de8395aadfaeaba860`。

全量 TTK 覆盖以下组合：

- permutedTokens：FP16、BF16、FP32；probs：FP16、BF16、FP32 及缺省。
- numTopk：1、2、8、512；hidden size：1、7、17、63、256、257、513、4096。
- 完整 range、非零 range start、range 外索引置零、probs 有/无。
- `rangeOptional` 为空时的有 probs 与无 probs 回退语义，并验证该状态下忽略 `numTopk`。
- 最小 shape 和 TopK=512 边界；Host UT 同时覆盖非法 TopK、probs 维度不匹配及 tokens rank 异常等约束负例。

## 说明

仓内现有 Kernel UT CMake 仅注册 `ascend910B1`，因此 `--opkernel_test --soc=ascend950` 只完成测试目标构建，没有产生可计数的 Ascend 950 Kernel UT 用例。本次 Ascend 950 kernel 的实际执行与精度门禁以上述隔离安装后的 16 条实机 TTK 结果为准。
