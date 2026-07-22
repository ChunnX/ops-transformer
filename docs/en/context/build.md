# build Parameter Description

## Introduction

build.sh is the build script for this project, located in the project root directory by default. Its function is to automatically compile, link, and configure source code, ultimately generating executable files, library files, or other target files that can be installed or run directly. Specifically, the script configures different parameters to implement multiple functions, including building various target libraries (such as libophost_transformer.so), compiling operator packages, executing unit tests, and so on.

## Usage Method

1. **Configure Environment Variables**

   Complete environment variable configuration by referring to [Environment Preparation](../context/quick_install.md).

   ```bash
   # Default path installation, using root user as example
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **Build Command Format**

   Using the compile operator package command as an example, the format is as follows, where `--vendor_name` and `--ops` are optional in this scenario.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   For full parameter meanings, refer to the parameter description section below. Select appropriate parameters according to actual situations.

## Parameter Description

build.sh supports multiple functions. View all function parameters through the following command.

```bash
bash build.sh --help
```

| Parameter Name | Optional/Required | Parameter Description |
|---|---|---|
| -j${n} | Optional | Specify compilation thread count. ${n} is the specific thread count, default value is 8 (such as -j8). If thread count exceeds CPU core count, it will automatically adjust to CPU core count. |
| -v | Optional | View CMake compilation configuration information. |
| -O${n} | Optional | Specify compilation optimization level, supports O0/O1/O2/O3 (such as -O3). ${n} is the optimization level identifier. |
| -u | Optional | Enable unit test (UT) compilation mode, compile all UT targets. |
| --help, -h | Optional | Print script usage help information. |
| --ops | Optional | Specify operators to be compiled, such as apply_rotary_pos_emb,rope_quant_kvcache. Multiple operators separated by comma ",", cannot be used simultaneously with --ophost, --opapi, --opgraph. |
| --soc | Optional | Specify NPU model, multiple soc separated by comma ",". |
| --jit | Optional | In static graph scenarios, when compiling the `cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run` full package, operator binary files do not need to be compiled (graph runtime will compile online). Configure this option to improve compilation speed. |
| --static | Optional | After configuration, generate static library files, including libcann_transformer_static.a and aclnn interface header files. Combined with --pkg parameter, generate static library compressed package. |
| --vendor_name | Optional | Specify custom operator package name, default value is custom. |
| --debug | Optional | Enable debug mode. |
| --cov | Optional | Reserved parameter, developers do not need to focus on this currently. |
| --noexec | Optional | Only compile unit test binary files, do not automatically execute compiled UT executable files. |
| --opkernel | Optional | Compile binary kernel. |
| --pkg | Optional | Generate installation package, cannot be used simultaneously with -u (UT mode) or --ophost, --opapi, --opgraph. |
| --make_clean | Optional | Execute basic cleanup operation (clean compilation products), script exits after execution. |
| --ophost | Optional | Compile libophost_transformer.so library, cannot be used simultaneously with --pkg, --ops. |
| --opapi | Optional | Compile libopapi_transformer.so library, cannot be used simultaneously with --pkg, --ops. |
| --opgraph | Optional | Compile libopgraph_transformer.so library, cannot be used simultaneously with --pkg, --ops. |
| --ophost_test | Optional | Compile ophost-related unit tests, equivalent to -u --ophost combination. |
| --opapi_test | Optional | Compile opapi-related unit tests, equivalent to -u --opapi combination. |
| --opgraph_test | Optional | Reserved parameter, developers do not need to focus on this currently. |
| --opkernel_test | Optional | Compile opkernel-related unit tests, equivalent to -u --opkernel combination. |
| --run_example | Optional | Compile specified operator and mode samples and execute compiled executable files. |
| --genop | Optional | Create AI Core custom operator initial directory. |
| --experimental | Optional | Compile user operators in the experimental directory. |
| --oom | Optional | Enable kernel-side oom memory detection function. |
| --cann_3rd_lib_path | Optional | Directory where third-party libraries are stored in offline compilation scenarios. |
