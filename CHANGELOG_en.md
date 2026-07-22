# CHANGELOG

> This document records important changes for each version. Versions are listed in reverse chronological order.

## 8.5.0-beta.1

Release date: 2025-12-30

The first Beta version 8.5.0-beta.1 of the ops-transformer operator is now available. This version introduces multiple new features, bug fixes, and performance improvements. The version is still in the testing stage.
Community feedback is welcome to further improve the stability and functional completeness of ops-transformer.
For usage instructions, refer to the [official documentation](https://gitcode.com/cann/ops-transformer/blob/master/README_en.md).

### 🔗 Version Address

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
The version directory description is as follows:
├── aarch64                  # CPU type is ARM
│   ├── ops                  # ops operator package directory, used for archiving operator sub-packages
│   ├── ...
├── x86_64                   # CPU type is X86
│   ├── ops                  # ops operator package directory, used for archiving operator sub-packages
│   ├── ...
```

### 📌 Version Compatibility

**ops-transformer sub-package and related component compatibility with CANN versions**

| **CANN Sub-package Version** | **Compatible CANN Version** |
|:----------------------------------|---------------------|
| cann-ops-transformer 8.5.0-beta.1 | CANN 8.5.0-beta.1   |
| cann-ops-math 8.5.0-beta.1        | CANN 8.5.0-beta.1   |
| cann-ops-nn 8.5.0-beta.1          | CANN 8.5.0-beta.1   |
| cann-ops-cv 8.5.0-beta.1          | CANN 8.5.0-beta.1   |
| cann-hccl 8.5.0-beta.1            | CANN 8.5.0-beta.1   |
| cann-hixl 8.5.0-beta.1            | CANN 8.5.0-beta.1   |

### 🚀 Key Features

- [Engineering Capability] Transformer-class ONNX operator plugin support. ([#539](https://gitcode.com/cann/ops-transformer/pull/539))
- [Operator Implementation] Some inference operators add support for KirinX90. ([#609](https://gitcode.com/cann/ops-transformer/pull/609))
- [Documentation Optimization] Add QUICK_START, offline compilation mode, and improve AI Core/graph mode development guides. ([#612](https://gitcode.com/cann/ops-transformer/pull/612), [#629](https://gitcode.com/cann/ops-transformer/pull/629), [#342](https://gitcode.com/cann/ops-transformer/pull/342))
- [Documentation Optimization] Optimize the new operator contribution process in the contribution guide. ([#384](https://gitcode.com/cann/ops-transformer/pull/384))

### 🐛 Bug Fixes

- MC2 communication domain supports unified torch.dist.group issue. ([Issue47](https://gitcode.com/cann/ops-transformer/issues/47))
- add_example sample operator execution invocation issue fix. ([Issue174](https://gitcode.com/cann/ops-transformer/issues/174))
- Fix the install_deps.sh script not supporting openEuler system issue. ([Issue255](https://gitcode.com/cann/ops-transformer/issues/255))
