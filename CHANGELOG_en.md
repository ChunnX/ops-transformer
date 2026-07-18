# CHANGELOG #

> This document describes the important changes in each version. The versions are sorted in descending order of time.

## 8.5.0-beta.1 ##

Release Date: 2025 - 12 - 30

The first beta version 8.5.0-beta.1 of the ops-transformer operator is now available. This version introduces a number of new features, bug fixes, and performance improvements, and is still in the testing phase. We welcome community feedback to further improve the stability and functionality of ops-transformer. For details, see[Official Document](https://gitcode.com/cann/ops-transformer/blob/master/README.md)	.

### Version address ###

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)	

```
版本目录说明如下：
├── aarch64                  #The CPU type is ARM.
│   ├── ops                  #Ops operator package directory, which is used to archive operator subpackages.
│   ├── ...
├── x86_64                   #The CPU type is x86.
│   ├── ops                  #Ops operator package directory, which is used to archive operator subpackages.
│   ├── ...
```

### Version mapping ###

**Mapping between the ops-transformer subpackage and related components and CANN versions**

| **CANN Subpackage Version**       | **Matching CANN Version** |
| --------------------------------- | ------------------------- |
| cann-ops-transformer 8.5.0-beta.1 | CANN 8.5.0-beta.1         |
| cann-ops-math 8.5.0-beta.1        | CANN 8.5.0-beta.1         |
| cann-ops-nn 8.5.0-beta.1          | CANN 8.5.0-beta.1         |
| cann-ops-cv 8.5.0-beta.1          | CANN 8.5.0-beta.1         |
| cann-hccl 8.5.0-beta.1            | CANN 8.5.0-beta.1         |
| cann-hixl 8.5.0-beta.1            | CANN 8.5.0-beta.1         |

### .Key Features ###

 *  \[Engineering capability\] The ONNX operator plug-in of the transformer class is supported.[\# 539](https://gitcode.com/cann/ops-transformer/pull/539)	)
 *  \[Operator implementation\] Some inference operators support KirinX90.[\# 609](https://gitcode.com/cann/ops-transformer/pull/609)	)
 *  \[Documentation Optimization\] The Quick_START and offline compilation modes are added. The development guide in the aicore/graph mode is optimized.[\# 612](https://gitcode.com/cann/ops-transformer/pull/612)	,[\# 629](https://gitcode.com/cann/ops-transformer/pull/629)	,[\# 342](https://gitcode.com/cann/ops-transformer/pull/342)	)
 *  \[Document Optimization\] Optimized the new operator contribution process in the contribution guide. ([\# 384](https://gitcode.com/cann/ops-transformer/pull/384)	)

### "Resolve the problem. ###

 *  The mc2 communication domain supports the unified torch.dist.group problem.[Issue47](https://gitcode.com/cann/ops-transformer/issues/47)	)
 *  The problem that the add_example sample operator is invoked is solved. ([Issue174](https://gitcode.com/cann/ops-transformer/issues/174)	)
 *  The install_deps.sh script does not support the openEuler system.[Issue255](https://gitcode.com/cann/ops-transformer/issues/255)	)

