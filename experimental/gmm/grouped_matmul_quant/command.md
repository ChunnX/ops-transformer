# GroupedMatmulQuant 精度测试命令

本文命令用于在 Ascend 910B 服务器上从源码构建并安装算子，然后执行 PyTorch 真机精度测试。

从仓库根目录一键执行完整流程：

```bash
bash experimental/gmm/grouped_matmul_quant/run_precision_test.sh
```

脚本会根据自身路径定位仓库，日志默认保存在
`build_out/grouped_matmul_quant_precision_logs`。可通过环境变量适配服务器，例如：

```bash
CANN_ENV_SCRIPT=/opt/Ascend/ascend-toolkit/set_env.sh \
PYTHON_BIN=/path/to/python3 \
OPP_INSTALL_ROOT=/opt/Ascend/opp \
bash experimental/gmm/grouped_matmul_quant/run_precision_test.sh
```

未显式设置 `OPP_INSTALL_ROOT` 时，脚本优先使用 CANN 环境中的 `ASCEND_OPP_PATH`，若该变量为空则使用
`${ASCEND_HOME_PATH}/opp`。安装目录必须对当前用户可写。

以下内容是完整流程的分步命令。

以下命令默认：

- 仓库位于 `~/Projects/ops-transformer`；如果实际路径不同，请修改第一条 `cd` 命令。
- CANN 安装在默认路径；如果实际路径不同，请修改 `set_env.sh` 路径。
- 使用独立 vendor 名 `gmm_quant`，避免覆盖其他 experimental 算子包。
- 所有步骤使用同一个 Python 环境执行。

## 1. 进入仓库并加载 CANN 环境

```bash
cd ~/Projects/ops-transformer

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -e
set -o pipefail

echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
which python3
python3 --version
npu-smi info
```

确认 PyTorch、torch_npu 和 NPU 状态：

```bash
python3 -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('NPU available:', torch.npu.is_available()); print('NPU count:', torch.npu.device_count())"
```

`NPU available` 应为 `True`，并且 `NPU count` 应大于 0。

## 2. 构建 experimental 自定义算子包

在仓库根目录执行：

```bash
cd ~/Projects/ops-transformer

bash build.sh \
  --pkg \
  --experimental \
  --ops=grouped_matmul_quant \
  --soc=ascend910b \
  --vendor_name=gmm_quant
```

确认安装包已经生成：

```bash
find build_out -maxdepth 1 -type f -name 'cann-ops-transformer-gmm_quant_linux*.run' -print
```

正常情况下，`build_out` 中会出现一个名称类似下面的文件：

```text
cann-ops-transformer-gmm_quant_linux-<architecture>.run
```

## 3. 安装并激活自定义算子包

```bash
cd ~/Projects/ops-transformer

PACKAGE_PATH=$(find build_out -maxdepth 1 -type f -name 'cann-ops-transformer-gmm_quant_linux*.run' -print -quit)
test -n "${PACKAGE_PATH}"
test -f "${PACKAGE_PATH}"
echo "Installing ${PACKAGE_PATH}"

chmod +x "${PACKAGE_PATH}"
"${PACKAGE_PATH}" --install-path="${ASCEND_HOME_PATH}/opp"

source "${ASCEND_HOME_PATH}/opp/vendors/gmm_quant_transformer/bin/set_env.bash"

echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
```

检查安装目录：

```bash
test -d "${ASCEND_HOME_PATH}/opp/vendors/gmm_quant_transformer"
find "${ASCEND_HOME_PATH}/opp/vendors/gmm_quant_transformer" -maxdepth 3 -type f
```

后续每次打开新的终端，都需要重新执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "${ASCEND_HOME_PATH}/opp/vendors/gmm_quant_transformer/bin/set_env.bash"
```

## 4. 安装 Python 测试依赖

```bash
python3 -m pip install pytest numpy
```

检查依赖版本：

```bash
python3 -c "import pytest, numpy; print('pytest:', pytest.__version__); print('numpy:', numpy.__version__)"
```

## 5. 重新构建并安装 PyTorch 注册扩展

这一步负责注册：

```text
torch.ops.npu_ops_transformer_ext.grouped_matmul_quant
```

执行：

```bash
cd ~/Projects/ops-transformer/experimental/npu_ops_transformer_ext

python3 setup.py clean

# 只构建本次精度测试需要的算子，避免受其他 experimental 算子影响。
export NPU_OPS_TRANSFORMER_EXT_OPS=grouped_matmul_quant
set -o pipefail

python3 -m pip install \
  --verbose \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  -e . \
  2>&1 | tee grouped_matmul_quant_extension_build.log
```

必须确认上面的安装命令成功结束，不能只检查 pytest 是否已安装。构建日志中应包含：

```text
Selected torch operators: grouped_matmul_quant
Found operator: grouped_matmul_quant
```

## 6. 检查扩展和算子注册

```bash
python3 -c "import torch, torch_npu, npu_ops_transformer_ext; op = torch.ops.npu_ops_transformer_ext.grouped_matmul_quant; print('package:', npu_ops_transformer_ext.__file__); print('extension:', npu_ops_transformer_ext._C.__file__); print('NPU available:', torch.npu.is_available()); print('operator:', op); print('PrivateUse1 kernel:', torch._C._dispatch_has_kernel_for_dispatch_key('npu_ops_transformer_ext::grouped_matmul_quant', 'PrivateUse1'))"
```

最后两项应打印：

```text
operator: npu_ops_transformer_ext.grouped_matmul_quant
PrivateUse1 kernel: True
```

如果这里提示 `GroupedMatmulQuant` 未注册，不要继续运行 pytest。先单独执行下面的命令以显示真实的扩展加载错误：

```bash
python3 -c "import npu_ops_transformer_ext"
```

## 7. 执行完整精度测试

```bash
cd ~/Projects/ops-transformer/experimental/gmm/grouped_matmul_quant/tests

set -o pipefail
python3 -m pytest -sv test_grouped_matmul_quant.py \
  2>&1 | tee grouped_matmul_quant_precision.log
```

完整测试共 8 个用例：

- 4 个 CPU INT4 打包、Golden 和容差判定测试。
- 4 个 Ascend 910B 真机精度测试。

正常结果应为：

```text
8 passed
```

## 8. 精度输出和通过标准

每个 NPU 用例会输出：

```text
case=<case name> op=<operator path> dtype=<dtype> matched_ratio=<ratio> \
max_abs_error=<error> max_rel_error=<error> atol=<atol> rtol=<rtol> cap=<cap>
```

通过标准：

| 输出类型 | `atol` | `rtol` | 最低通过比例 | 最大绝对误差上限 |
| :--- | :---: | :---: | :---: | :---: |
| FP16 | `2^-9` | `2^-9` | `0.99` | `0.1` |
| BF16 | `2^-6` | `2^-6` | `0.99` | `1.0` |

四个真机用例覆盖：

- FP16 和 BF16。
- INT4 边界值 `-8` 和 `7`。
- 非零 offset。
- 单专家和多专家。
- 零 token 专家和不均衡专家。
- 不同 `scale_group_size`。
- M、N、K 尾块。
- Split-K 和非 Split-K。

## 9. 分阶段运行测试

只检查 CPU 端 INT4 打包和 Golden：

```bash
cd ~/Projects/ops-transformer/experimental/gmm/grouped_matmul_quant/tests
python3 -m pytest -sv test_grouped_matmul_quant.py -k 'not precision'
```

只执行 Ascend NPU 精度用例：

```bash
cd ~/Projects/ops-transformer/experimental/gmm/grouped_matmul_quant/tests
python3 -m pytest -sv test_grouped_matmul_quant.py -k precision
```

只执行一个指定用例，例如 FP16 Split-K：

```bash
python3 -m pytest -sv \
  'test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_single_tail_split_k]'
```

## 10. PR 测试记录建议

保存并记录以下信息：

```bash
python3 --version
python3 -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)"
npu-smi info
```

PR 测试说明中建议包含：

- 服务器和 Ascend 910B 型号。
- CANN、PyTorch、torch_npu 和 Python 版本。
- 算子包及 PyTorch 扩展的构建命令。
- `8 passed` 测试汇总。
- 四个 NPU 用例的 `matched_ratio`、`max_abs_error` 和 `max_rel_error`。
- `grouped_matmul_quant_extension_build.log` 和 `grouped_matmul_quant_precision.log` 的摘要。
