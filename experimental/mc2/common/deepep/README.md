# DeepEP 迁移算子的共享运行时

源码基线：DeepEP `7c4b276`。本目录提供 SHMEM 生命周期、事件、SDK 校验和直调编译支持；算子实现分别保存在 `experimental/mc2` 下对应目录。Python 命名空间为 `ops_transformer_mc2`，不覆盖 `deep_ep`。

## 环境与依赖

要求 Linux、Ascend950、CANN **9.2.0**、Python >= 3.11、CMake >= 3.20，以及配套的 PyTorch、torch_npu、pybind11。低时延通信使用每服务器 8 个连续 rank、EP16/32/64/128。编译和设备验证须在此环境进行。

参考仓库 `cmake/third_party` 的依赖管理方式，将原 DeepEP 的 Git 子模块转为 **固定提交的 FetchContent 依赖**，放入被 Git 忽略的 `third_party/mc2_deepep`。不复制子模块内容，也不依赖原 DeepEP 工作目录。SHMEM 固定为 `7f9d961d378225e69058080532cf4f5be16cc83d`；已有源码可以离线复用，版本不符会报错，不自动切换已有 checkout。

从 ops-transformer 根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S experimental/mc2/common/deepep/dependencies \
      -B experimental/mc2/common/deepep/build/dependencies
# 离线源码：给上述命令增加 -DMC2_SHMEM_SOURCE=/path/to/pinned/shmem

# 首次编译 SHMEM。上游脚本会重建其 build/install 目录，请使用独立依赖缓存。
(cd third_party/mc2_deepep/shmem && bash -e scripts/build.sh -soc_type Ascend950)
export SHMEM_ROOT="$PWD/third_party/mc2_deepep/shmem/install/shmem"
python3 experimental/mc2/common/deepep/scripts/record_shmem_sdk.py \
    --sdk "$SHMEM_ROOT" --revision 7f9d961d378225e69058080532cf4f5be16cc83d
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

也可直接设置 `SHMEM_ROOT` 指向相同提交构建的已有 SDK，无需获取源码。首次使用以相同命令记录 SDK；已有 `deepep-sdk.json` 时改用 `--verify`。记录操作校验公共头文件并保存头文件/动态库哈希；库的源码来源由 SDK 提供方保证。每次配置都会重新校验，不接受变化的 SDK。原子模块中的次级依赖仍由 SHMEM 自身构建脚本处理。

## 构建方式

与 `quant_all_reduce` 的独立直调工程相同，这些算子通过目录内 `build.sh` 构建，使用 CANN bisheng 编译 AscendC 内核、系统 C++ 编译器编译 Host/Python 绑定；不生成 OPP `.run` 包。仓库全量 experimental 遍历会跳过这些独立入口，不额外加载 SHMEM/PyTorch。

```bash
bash experimental/mc2/low_latency_dispatch/build.sh
export PYTHONPATH="$PWD/experimental/mc2/low_latency_dispatch/output/python:${PYTHONPATH:-}"
python3 -c 'from ops_transformer_mc2 import ElasticBuffer; print(ElasticBuffer)'
```

可以用 `MC2_BUILD_DIR`、`MC2_INSTALL_DIR` 指定构建/安装路径，用 `PYTHON` 指定 Python。`build.sh` 后的 `-D...` 透传给 CMake。只把一个安装目录放在 `PYTHONPATH` 首位；依赖此运行时的后续算子会安装同一包及所需前置算子。重新配置不会自动安装或修改系统 Python 包。

其他入口：[LowLatencyCombine](../../low_latency_combine/README.md) 构建 Dispatch + Combine；[MegaMoE](../../mega_moe/README.md) 另外安装融合计算和诊断工具。MegaMoE 需要依赖准备命令增加 `-DMC2_FETCH_CATLASS=ON`，CATLASS 固定为 `2b4bd7cbf7e67ebab23d7d5a99d472c83f90d9a1`，与仓库默认 CATLASS 缓存分离。

## 所有权与同步

使用独立于 torchrun rendezvous 的端口设置 `DEEPEP_SHMEM_ENDPOINT=tcp://<IPv4>:<port>`；所有 rank 必须一致。NPU 上下文由调用方创建，SHMEM workspace 由 `ElasticBuffer` 持有；所有 rank 按同一顺序调用、显式 `destroy()`，之后再销毁进程组。保留上游环境变量名便于复用运行配置。库加载时比较实际 SHMEM 库与构建哈希，发现差异会告警。

## 来源与验证范围

本次迁移与适配的版权署名为 `Copyright (c) 2026, edconeone`，遵循仓库根目录的 CANN 2.0 许可证；引用源码已有版权与许可证声明予以保留。DeepSeek 接口声明的 MIT 许可证和迁移基线携带的 CANN 许可证原文保存在 [LICENSES/mc2](../../../../LICENSES/mc2)，引用记录见仓库根目录的 `Third_Party_Open_Source_Software_Notice`。原始声明中的 `third-party/*.LICENSE` 路径对应这些保留的许可证原文。

CPU 测试用于验证 Host 参数、生命周期和布局约束，不能代替 NPU 编译、跨机精度和性能验收。
