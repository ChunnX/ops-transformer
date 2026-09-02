# GroupedMatmulQuant 精度测试命令

本文命令用于在 Ascend 910B 服务器上从源码构建并安装算子，然后执行 PyTorch 真机精度测试。

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

python3 -m pip install \
  --verbose \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  -e . \
  2>&1 | tee grouped_matmul_quant_extension_build.log
```

必须确认上面的安装命令成功结束，不能只检查 pytest 是否已安装。

## 6. 检查扩展和算子注册

```bash
python3 -c "import torch, torch_npu, npu_ops_transformer_ext; print('package:', npu_ops_transformer_ext.__file__); print('extension:', npu_ops_transformer_ext._C.__file__); print('NPU available:', torch.npu.is_available()); print('operator:', torch.ops.npu_ops_transformer_ext.grouped_matmul_quant)"
```

最后一项应打印：

```text
operator: npu_ops_transformer_ext.grouped_matmul_quant
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

完整测试共 6 个用例：

- 2 个 CPU INT4 打包和 Golden 参考实现测试。
- 4 个 Ascend 910B 真机精度测试。

正常结果应为：

```text
6 passed
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
- `6 passed` 测试汇总。
- 四个 NPU 用例的 `matched_ratio`、`max_abs_error` 和 `max_rel_error`。
- `grouped_matmul_quant_extension_build.log` 和 `grouped_matmul_quant_precision.log` 的摘要。



error:
(base) j50061995@slave19:~/Projects/ops-transformer/experimental/npu_ops_transformer_ext$ python3 -m pip install \
  --verbose \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  -e . \
  2>&1 | tee grouped_matmul_quant_extension_build.log
Using pip 26.1.2 from /home/j50061995/miniconda3/lib/python3.14/site-packages/pip (python 3.14)
Obtaining file:///home/j50061995/Projects/ops-transformer/experimental/npu_ops_transformer_ext
  Checking if build backend supports build_editable: started
  Running command Checking if build backend supports build_editable
  Checking if build backend supports build_editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Running command Preparing editable metadata (pyproject.toml)
  running dist_info
  creating /tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info
  writing /tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/PKG-INFO
  writing dependency_links to /tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/dependency_links.txt
  writing requirements to /tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/requires.txt
  writing top-level names to /tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/top_level.txt
  writing manifest file '/tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/SOURCES.txt'
  reading manifest file '/tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/SOURCES.txt'
  writing manifest file '/tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext.egg-info/SOURCES.txt'
  creating '/tmp/pip-modern-metadata-2csmroy4/npu_ops_transformer_ext-0.0.1.dist-info'
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Building wheels for collected packages: npu_ops_transformer_ext
  Building editable for npu_ops_transformer_ext (pyproject.toml): started
  Running command Building editable for npu_ops_transformer_ext (pyproject.toml)
  running editable_wheel
  creating /tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info
  writing /tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/PKG-INFO
  writing dependency_links to /tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/dependency_links.txt
  writing requirements to /tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/requires.txt
  writing top-level names to /tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/top_level.txt
  writing manifest file '/tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/SOURCES.txt'
  reading manifest file '/tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/SOURCES.txt'
  writing manifest file '/tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext.egg-info/SOURCES.txt'
  creating '/tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext-0.0.1.dist-info'
  creating /tmp/pip-ephem-wheel-cache-j6g9u28h/wheels/9c/65/0d/8333df6c7bb9e37fb79554516ec89d2087a972a51dca4ce8bc/tmp1nqu0h7e/.tmp-v22p0ya7/npu_ops_transformer_ext-0.0.1.dist-info/WHEEL
  running build_py
  running build_ext
  -- The C compiler identification is GNU 11.4.0
  -- The CXX compiler identification is GNU 11.4.0
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Check for working C compiler: /usr/bin/cc - skipped
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: /usr/bin/c++ - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Build torch ops: ON
  -- /home/j50061995/Ascend/cann-9.0.0
  -- ASCEND_HOME = /home/j50061995/Ascend/cann-9.0.0
  -- BISHENG     = /home/j50061995/Ascend/cann-9.0.0/bin/bisheng
  -- Found Python3: /home/j50061995/miniconda3/bin/python3.1 (found version "3.14.6") found components: Interpreter Development Development.Module Development.Embed
  -- Python3_EXECUTABLE   = /home/j50061995/miniconda3/bin/python3.1
  -- Python3_INCLUDE_DIRS = /home/j50061995/miniconda3/include/python3.14
  -- Python3_LIBRARIES    = /home/j50061995/miniconda3/lib/libpython3.14.so
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
      import torch; print(torch.utils.cmake_prefix_path)
      ^^^^^^^^^^^^
  ModuleNotFoundError: No module named 'torch'
  CMake Error at CMakeLists.txt:85 (find_package):
    By not providing "FindTorch.cmake" in CMAKE_MODULE_PATH this project has
    asked CMake to find a package configuration file provided by "Torch", but
    CMake did not find one.

    Could not find a package configuration file provided by "Torch" with any of
    the following names:

      TorchConfig.cmake
      torch-config.cmake

    Add the installation prefix of "Torch" to CMAKE_PREFIX_PATH or set
    "Torch_DIR" to a directory containing one of the above files.  If "Torch"
    provides a separate development package or SDK, be sure it has been
    installed.


  -- Configuring incomplete, errors occurred!
  See also "/tmp/tmp2xk83lok.build-temp/npu_ops_transformer_ext._C/CMakeFiles/CMakeOutput.log".
  Traceback (most recent call last):
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
      main()
      ~~~~^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
      json_out["return_val"] = hook(**hook_input["kwargs"])
                               ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 303, in build_editable
      return hook(wheel_directory, config_settings, metadata_directory)
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/build_meta.py", line 474, in build_editable
      return self._build_with_temp_dir(
             ~~~~~~~~~~~~~~~~~~~~~~~~~^
          cmd, ".whl", wheel_directory, config_settings
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      )
      ^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/build_meta.py", line 410, in _build_with_temp_dir
      self.run_setup()
      ~~~~~~~~~~~~~~^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/build_meta.py", line 520, in run_setup
      super().run_setup(setup_script=setup_script)
      ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/build_meta.py", line 317, in run_setup
      exec(code, locals())
      ~~~~^^^^^^^^^^^^^^^^
    File "<string>", line 118, in <module>
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/__init__.py", line 117, in setup
      return distutils.core.setup(**attrs)  # type: ignore[return-value]
             ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/_distutils/core.py", line 186, in setup
      return run_commands(dist)
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
      dist.run_commands()
      ~~~~~~~~~~~~~~~~~^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/_distutils/dist.py", line 1000, in run_commands
      self.run_command(cmd)
      ~~~~~~~~~~~~~~~~^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/dist.py", line 1107, in run_command
      super().run_command(command)
      ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/_distutils/dist.py", line 1019, in run_command
      cmd_obj.run()
      ~~~~~~~~~~~^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/command/editable_wheel.py", line 140, in run
      self._create_wheel_file(bdist_wheel)
      ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/command/editable_wheel.py", line 350, in _create_wheel_file
      files, mapping = self._run_build_commands(dist_name, unpacked, lib, tmp)
                       ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/command/editable_wheel.py", line 273, in _run_build_commands
      self._run_build_subcommands()
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/command/editable_wheel.py", line 300, in _run_build_subcommands
      self.run_command(name)
      ~~~~~~~~~~~~~~~~^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/_distutils/cmd.py", line 341, in run_command
      self.distribution.run_command(command)
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/dist.py", line 1107, in run_command
      super().run_command(command)
      ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
    File "/home/j50061995/miniconda3/lib/python3.14/site-packages/setuptools/_distutils/dist.py", line 1019, in run_command
      cmd_obj.run()
      ~~~~~~~~~~~^^
    File "<string>", line 86, in run
    File "<string>", line 114, in build_cmake
    File "/home/j50061995/miniconda3/lib/python3.14/subprocess.py", line 420, in check_call
      raise CalledProcessError(retcode, cmd)
  subprocess.CalledProcessError: Command '['cmake', '/home/j50061995/Projects/ops-transformer/experimental/npu_ops_transformer_ext', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/j50061995/Projects/ops-transformer/experimental/npu_ops_transformer_ext/npu_ops_transformer_ext', '-DPYTHON_EXECUTABLE=/home/j50061995/miniconda3/bin/python3', '-DBUILD_TORCH_OPS=ON', '-DPYTHON_EXTENSION_USE_ABI3=ON', '-DPYTHON_INCLUDE_DIR=/home/j50061995/miniconda3/include/python3.14', '-DPYTHON_LIBRARIES=/home/j50061995/miniconda3/lib', '-DPy_LIMITED_API_VERSION=0x03080000']' returned non-zero exit status 1.
  An error occurred when building editable wheel for npu_ops_transformer_ext.
  See debugging tips in: https://setuptools.pypa.io/en/latest/userguide/development_mode.html#debugging-tips
  error: subprocess-exited-with-error
  
  × Building editable for npu_ops_transformer_ext (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> No available output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  full command: /home/j50061995/miniconda3/bin/python3 /home/j50061995/miniconda3/lib/python3.14/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py build_editable /tmp/tmp5zzvebgd
  cwd: /home/j50061995/Projects/ops-transformer/experimental/npu_ops_transformer_ext
  Building editable for npu_ops_transformer_ext (pyproject.toml): finished with status 'error'
  ERROR: Failed building editable for npu_ops_transformer_ext
Failed to build npu_ops_transformer_ext
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> npu_ops_transformer_ext
(base) j50061995@slave19:~/Projects/ops-transformer/experimental/npu_ops_transformer_ext$ python3 -c "import torch, torch_npu, npu_ops_transformer_ext; print('package:', npu_ops_transformer_ext.__file__); print('extension:', npu_ops_transformer_ext._C.__file__); print('NPU available:', torch.npu.is_available()); print('operator:', torch.ops.npu_ops_transformer_ext.grouped_matmul_quant)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch, torch_npu, npu_ops_transformer_ext; print('package:', npu_ops_transformer_ext.__file__); print('extension:', npu_ops_transformer_ext._C.__file__); print('NPU available:', torch.npu.is_available()); print('operator:', torch.ops.npu_ops_transformer_ext.grouped_matmul_quant)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch'