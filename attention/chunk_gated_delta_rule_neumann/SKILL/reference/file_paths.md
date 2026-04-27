# 关键文件路径模板

## 环境变量

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/latest      # 或 ~/Ascend/latest
ASCEND_OPP_PATH=${ASCEND_HOME_PATH}/opp
ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer
```

## CANN 标准路径

```
${ASCEND_HOME_PATH}/
├── bin/
│   └── setenv.bash                      # 环境设置脚本（必须 source）
│
├── ${ARCH}-linux/                       # ARCH = aarch64 或 x86_64
│   ├── include/
│   │   ├── acl/
│   │   │   └── acl.h                    # ACL 主头文件
│   │   │   └── acl_rt.h                 # Runtime API
│   │   │   └── acl_op.h                 # Operator API
│   │   │
│   │   └── aclnnop/
│   │       └── aclnn_util.h             # aclnn 工具函数
│   │       └── aclnn_common.h           # 公共定义
│   │
│   └── lib64/
│       ├── libascendcl.so               # ACL 运行库
│       ├── libnnopbase.so               # nnopbase 库
│       └── libc_sec.so                  # 安全库
│
└── opp/
    ├── builtin/                         # 内置算子
    │   └── op_impl/ai_core/tbe/
    │       └── kernel/                  # 内置 kernel
    │
    └── vendors/
        └── custom_transformer/          # 自定义算子安装位置
            ├── version.info             # 版本信息
            │
            ├── op_api/
            │   ├── include/
            │   │   ├── aclnnop/
            │   │   │   └── aclnn_${算子名}.h  # 算子 API 头文件
            │   │   └── ${算子名}.h            # 算子定义头文件
            │   │
            │   └── lib/
            │       ├── libcust_opapi.so       # 自定义算子 API 库
            │       └── libcust_opapi_ext.so   # 扩展库
            │
            └── op_impl/
                └── ai_core/
                    └── tbe/
                        ├── kernel/
                        │   └── ${soc}/              # SoC 版本目录
                        │       └── ${算子名}/
                        │           ├── *.o          # kernel object 文件
                        │           └── binary_info_config.json
                        │
                        └── op_tiling/
                            ├── src/
                            │   └── ${算子名}/
                            │       └── *.cpp        # tiling 实现源码
                            │
                            └── lib/
                                └── linux/
                                    └── aarch64/
                                        └── libcust_opmaster_rt2.0.so  # tiling 库
```

## 编译时路径变量

```bash
ARCH_INFO=$(uname -m)                    # aarch64 或 x86_64

# Include 路径
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
EAGER_INCLUDE_OPP_ACLNNOP_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"

# 自定义算子 Include 路径
CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include"

# Library 路径
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64"
CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib"

# Tiling Library 路径（运行时必须）
TILING_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
```

## 关键文件检查命令

```bash
# 检查 ACL 头文件
ls ${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/acl/acl.h

# 检查自定义算子 API 头文件
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include/aclnnop/aclnn_${算子名}.h

# 检查自定义算子库
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib/libcust_opapi.so

# 检查 kernel 文件
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${soc}/${算子名}/

# 检查 tiling 库
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
```

## 缺失文件处理

| 缺失文件 | 原因 | 解决方案 |
|---------|------|---------|
| aclnn_${算子名}.h | 算子未安装 | 重新运行 `.run` 安装包 |
| libcust_opapi.so | 算子未安装 | 重新运行 `.run` 安装包 |
| kernel/*.o | SoC 版本不匹配 | 使用正确 `--soc` 重新编译 |
| libcust_opmaster_rt2.0.so | 算子未安装 | 重新运行 `.run` 安装包 |