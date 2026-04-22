#!/usr/bin/env python3
"""Insert directory structure section and renumber sections in analysis reports."""

import re
import os

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

# Map: report filename -> (operator_dir_name, directory_tree_ascii)
# operator_dir_name is relative to moe/
REPORTS = {
    "moe_finalize_routing_v2_full_analysis.md": (
        "moe_finalize_routing_v2",
        """```
moe_finalize_routing_v2/
├── CMakeLists.txt                                      # 顶层构建配置
├── README.md                                           # 算子说明文档
├── op_host/                                            # Host侧代码
│   ├── moe_finalize_routing_v2_def.cpp                 # 算子定义（输入/输出/属性声明）
│   ├── moe_finalize_routing_v2_tiling.h                # Tiling数据结构定义
│   ├── moe_finalize_routing_v2_tiling.cpp              # Tiling策略实现(membase)
│   ├── moe_finalize_routing_v2_tiling_arch35.cpp       # arch35 Tiling策略
│   ├── moe_finalize_routing_v2_tiling_arch35.h         # arch35 Tiling数据结构
│   ├── moe_finalize_routing_v2_tiling_membase.cpp      # membase Tiling策略
│   ├── moe_finalize_routing_v2_infershape.cpp          # Shape推导
│   ├── config/                                         # 各SoC的二进制配置
│   │   ├── ascend910b/
│   │   ├── ascend910_93/
│   │   ├── ascend950/
│   │   ├── kirinx90/
│   │   └── kirin9030/
│   └── op_api/                                         # ACLNN API
│       ├── aclnn_moe_finalize_routing_v2.h             # API头文件
│       └── aclnn_moe_finalize_routing_v2.cpp           # API实现
├── op_kernel/                                          # Kernel侧代码（AscendC）
│   ├── moe_finalize_routing_v2.cpp                     # Kernel入口，TilingKey分发
│   ├── moe_finalize_routing_v2_apt.cpp                 # APT Kernel入口
│   ├── moe_finalize_routing_v2_bf16.h                  # BF16乒乓并行模板
│   ├── moe_finalize_routing_v2_bf16_all_bias.h         # BF16全载Bias
│   ├── moe_finalize_routing_v2_bf16_cuth.h             # BF16切H
│   ├── moe_finalize_routing_v2_bf16_cuth_k2.h          # BF16切H且K=2
│   ├── moe_finalize_routing_v2_bf16_cuth_k4.h          # BF16切H且K=4
│   ├── moe_finalize_routing_v2_bf16_cutk.h             # BF16大K切分
│   ├── moe_finalize_routing_v2_common.h                # 公共常量和工具
│   ├── moe_finalize_routing_v2_fp.h                    # FP32/FP16基础模板
│   ├── moe_finalize_routing_v2_fp_all_bias.h           # FP全载Bias
│   ├── moe_finalize_routing_v2_fp_cuth.h               # FP切H
│   ├── moe_finalize_routing_v2_fp_cuth_k2.h            # FP切H且K=2
│   ├── moe_finalize_routing_v2_fp_cuth_k4.h            # FP切H且K=4
│   ├── moe_finalize_routing_v2_fp_cutk.h               # FP大K切分
│   ├── moe_finalize_routing_v2_fp_db.h                 # FP Double Buffer
│   ├── moe_finalize_routing_v2_fp_db_all_bias.h        # FP Double Buffer全载Bias
│   ├── moe_finalize_routing_v2_fp_mix.h                # 混合精度模板
│   ├── moe_finalize_routing_v2_fp_mix_cuth.h           # 混合精度切H
│   ├── moe_finalize_routing_v2_fp_mix_cutk.h           # 混合精度切K
│   ├── moe_finalize_routing_v2_without_bias_fp.h       # 无Bias FP模板
│   ├── moe_finalize_routing_v2_without_bias_fp_cutk.h  # 无Bias FP切K
│   └── arch35/                                         # arch35专用Kernel
├── op_graph/                                           # 图模式
│   └── moe_finalize_routing_v2_proto.h                 # 算子原型注册
├── framework/                                          # 框架适配
│   └── npu_moe_finalize_routing_v2_onnx_plugin.cpp     # ONNX插件
├── docs/                                               # API文档
│   └── aclnnMoeFinalizeRoutingV2.md
├── examples/                                           # 调用示例
│   └── test_aclnn_moe_finalize_routing_v2.cpp
└── tests/ut/                                           # 测试代码
```"""
    ),
    "moe_finalize_routing_v2_grad_full_analysis.md": (
        "moe_finalize_routing_v2_grad",
        """```
moe_finalize_routing_v2_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_finalize_routing_v2_grad_def.cpp             # 算子定义
│   ├── moe_finalize_routing_v2_grad_tiling.h            # Tiling数据结构
│   ├── moe_finalize_routing_v2_grad_tiling.cpp          # Tiling策略实现
│   ├── moe_finalize_routing_v2_grad_tiling_membase.cpp  # membase Tiling
│   ├── moe_finalize_routing_v2_grad_infershape.cpp      # Shape推导
│   ├── moe_finalize_routing_v2_grad_tiling_arch35.cpp   # arch35 Tiling主入口
│   ├── *_split_h_tiling_arch35.cpp                      # arch35切H Tiling
│   ├── *_not_split_h_tiling_arch35.cpp                  # arch35不切H Tiling
│   ├── *_without_scale_tiling_arch35.cpp                # arch35无scale Tiling
│   ├── config/                                          # SoC配置(910b/910_93/950)
│   └── op_api/                                          # ACLNN API
├── op_kernel/
│   ├── moe_finalize_routing_v2_grad.cpp                 # Kernel入口
│   ├── moe_finalize_routing_v2_grad_apt.cpp             # APT入口
│   ├── moe_finalize_routing_v2_grad_base.h              # 基础模板
│   ├── *_with_scale_cut_h.h                             # 带scale切H模板
│   ├── *_with_scale_not_cut_h.h                         # 带scale不切H模板
│   ├── *_without_scale_cut_h.h                          # 无scale切H模板
│   ├── *_without_scale_not_cut_h.h                      # 无scale不切H模板
│   └── arch35/                                          # arch35专用(regbase)
├── op_graph/
│   └── moe_finalize_routing_v2_grad_proto.h
├── docs/
│   └── aclnnMoeFinalizeRoutingV2Grad.md
├── examples/
│   └── test_aclnn_moe_finalize_routing_v2_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_gating_top_k_full_analysis.md": (
        "moe_gating_top_k",
        """```
moe_gating_top_k/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_def.cpp               # 算子定义
│   ├── moe_gating_top_k_tiling.h              # Tiling数据结构
│   ├── moe_gating_top_k_tiling.cpp            # Tiling策略实现
│   ├── moe_gating_top_k_tiling_arch35.cpp     # arch35 Tiling
│   ├── moe_gating_top_k_tiling_base.cpp       # Tiling基类
│   ├── moe_gating_top_k_infershape.cpp        # Shape推导
│   └── config/                                # SoC配置(910b/910_93/950)
├── op_kernel/
│   ├── moe_gating_top_k.cpp                   # Kernel入口
│   ├── moe_gating_top_k_apt.cpp               # APT入口
│   ├── moe_gating_top_k_e_k_fullload.h        # E和K全载模板
│   ├── moe_gating_top_k_generalized.h         # 通用分组模板(DeepSeekV3)
│   ├── moe_gating_top_k_without_group.h       # 无分组模板
│   ├── common.h                               # 公共常量
│   └── arch35/                                # arch35专用(regbase)
├── op_graph/
│   └── moe_gating_top_k_proto.h
├── docs/
│   └── aclnnMoeGatingTopK.md
├── examples/
│   ├── test_aclnn_moe_gating_top_k.cpp
│   └── test_geir_moe_gating_top_k.cpp
└── tests/ut/
```"""
    ),
    "moe_gating_top_k_softmax_full_analysis.md": (
        "moe_gating_top_k_softmax",
        """```
moe_gating_top_k_softmax/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_softmax_def.cpp               # 算子定义
│   ├── moe_gating_top_k_softmax_tiling.h              # Tiling数据结构
│   ├── moe_gating_top_k_softmax_tiling.cpp            # Tiling策略
│   ├── moe_gating_top_k_softmax_tiling_base.cpp       # Tiling基类
│   ├── moe_gating_top_k_softmax_tiling_base.h         # Tiling基类头文件
│   ├── moe_gating_top_k_softmax_tiling_arch35.cpp     # arch35 Tiling
│   ├── moe_gating_top_k_softmax_tiling_k_full_load.cpp # K全载Tiling
│   ├── moe_gating_top_k_softmax_tiling_e_k_full_load.cpp # E和K全载Tiling
│   ├── moe_gating_top_k_softmax_tiling_perf.cpp       # 性能优化Tiling
│   ├── moe_gating_top_k_softmax_infershape.cpp        # Shape推导
│   └── config/                                        # SoC配置(910b/910_93/950)
├── op_kernel/
│   ├── moe_gating_top_k_softmax.cpp                   # Kernel入口
│   ├── moe_gating_top_k_softmax_apt.cpp               # APT入口
│   ├── moe_gating_top_k_softmax_k_fullload.h          # K全载模板
│   ├── moe_gating_top_k_softmax_e_k_fullload.h        # E和K全载模板
│   ├── moe_gating_top_k_softmax_perf.h                # 性能优化模板
│   └── arch35/                                        # arch35专用(regbase)
├── op_graph/
│   └── moe_gating_top_k_softmax_proto.h
├── framework/
│   └── npu_moe_gating_top_k_softmax_onnx_plugin.cpp   # ONNX插件
├── docs/
│   └── aclnnMoeGatingTopKSoftmax.md
├── examples/
│   └── test_aclnn_moe_gating_top_k_softmax.cpp
└── tests/ut/
```"""
    ),
    "moe_gating_top_k_softmax_v2_full_analysis.md": (
        "moe_gating_top_k_softmax_v2",
        """```
moe_gating_top_k_softmax_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_softmax_v2_def.cpp                    # 算子定义
│   ├── moe_gating_top_k_softmax_v2_tiling.h                   # Tiling数据结构
│   ├── moe_gating_top_k_softmax_v2_tiling.cpp                 # Tiling策略
│   ├── moe_gating_top_k_softmax_v2_tiling_base.cpp            # Tiling基类
│   ├── moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp      # K全载Tiling
│   ├── moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp     # E和K全载Tiling
│   ├── moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp        # K重归一化Tiling
│   ├── moe_gating_top_k_softmax_v2_tiling_perf.cpp            # 性能优化Tiling
│   ├── moe_gating_top_k_softmax_v2_regbase_tiling_perf.cpp    # regbase性能Tiling
│   ├── moe_gating_top_k_softmax_v2_infershape.cpp             # Shape推导
│   └── config/                                                # SoC配置(910b/950)
├── op_kernel/
│   ├── moe_gating_top_k_softmax_v2.cpp                        # Kernel入口
│   ├── moe_gating_top_k_softmax_v2_apt.cpp                    # APT入口
│   ├── moe_gating_top_k_softmax_v2_k_fullload.h               # K全载模板
│   ├── moe_gating_top_k_softmax_v2_ek_fullload.h              # E和K全载模板
│   ├── moe_gating_top_k_softmax_v2_k_renorm.h                 # K重归一化模板
│   ├── moe_gating_top_k_softmax_v2_perf.h                     # 性能优化模板
│   └── arch35/                                                # arch35专用(regbase)
├── op_graph/
│   └── moe_gating_top_k_softmax_v2_proto.h
├── docs/
│   └── aclnnMoeGatingTopKSoftmaxV2.md
├── examples/
│   └── test_aclnn_moe_gating_top_k_softmax_v2.cpp
└── tests/ut/
```"""
    ),
    "moe_fused_topk_full_analysis.md": (
        "moe_fused_topk",
        """```
moe_fused_topk/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_fused_topk_def.cpp               # 算子定义
│   ├── moe_fused_topk_tiling.h              # Tiling数据结构
│   ├── moe_fused_topk_tiling.cpp            # Tiling策略
│   ├── config/                              # SoC配置(910b/910_93)
│   └── op_api/                              # ACLNN API
│       ├── aclnn_moe_fused_topk.h
│       ├── aclnn_moe_fused_topk.cpp
│       ├── moe_fused_topk.h
│       └── moe_fused_topk.cpp
├── op_kernel/
│   ├── moe_fused_topk.cpp                   # Kernel入口
│   └── moe_fused_topk.h                     # Kernel模板(分组TopK+映射表)
├── docs/
│   └── aclnnMoeFusedTopk.md
├── examples/
│   └── test_aclnn_moe_fused_topk.cpp
└── tests/ut/
```"""
    ),
    "moe_init_routing_full_analysis.md": (
        "moe_init_routing",
        """```
moe_init_routing/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_def.cpp             # 算子定义
│   ├── moe_init_routing_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_tiling.cpp          # Tiling策略实现
│   ├── moe_init_routing_infershape.cpp      # Shape推导
│   └── config/                              # SoC配置(910b/910_93/950)
├── op_kernel/
│   ├── moe_init_routing.cpp                 # Kernel入口，TilingKey分发
│   ├── moe_init_routing_apt.cpp             # APT Kernel入口
│   ├── moe_common.h                         # 公共常量和工具
│   ├── moe_init_routing_fullload.h          # 全载模板
│   ├── moe_sort_base.h                      # 排序基类
│   ├── moe_sort_one_core.h                  # 单核排序
│   ├── moe_sort_multi_core.h                # 多核归并排序
│   ├── moe_mrgsort.h                        # Merge Sort
│   ├── moe_mrgsort_out.h                    # Merge Sort输出
│   ├── moe_src_to_dst_op.h                  # 源到目标映射(SIMD)
│   ├── moe_gather_out.h                     # Gather输出模板
│   ├── moe_gather_out_small_activate_row.h  # 小激活行Gather
│   └── arch35/                              # arch35专用(SIMT+regbase)
├── op_graph/
│   └── moe_init_routing_proto.h
├── framework/
│   └── npu_moe_init_routing_onnx_plugin.cpp # ONNX插件
├── docs/
│   └── aclnnMoeInitRouting.md
├── examples/
│   └── test_aclnn_moe_init_routing.cpp
└── tests/ut/
```"""
    ),
    "moe_init_routing_v2_full_analysis.md": (
        "moe_init_routing_v2",
        """```
moe_init_routing_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_v2_def.cpp             # 算子定义
│   ├── moe_init_routing_v2_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_v2_tiling.cpp          # Tiling策略实现
│   ├── moe_init_routing_v2_tiling_util.h       # Tiling工具函数
│   ├── moe_init_routing_v2_infershape.cpp      # Shape推导
│   └── op_api/                                 # ACLNN API
│       ├── aclnn_moe_init_routing_v2.h
│       └── aclnn_moe_init_routing_v2.cpp
├── op_kernel/
│   ├── moe_init_routing_v2.cpp                 # Kernel入口
│   ├── moe_v2_common.h                        # 公共常量
│   ├── moe_v2_init_routing_fullload.h         # 全载模板
│   ├── moe_v2_sort_base.h                     # 排序基类
│   ├── moe_v2_sort_one_core.h                 # 单核排序
│   ├── moe_v2_sort_multi_core.h               # 多核排序
│   ├── moe_v2_mrgsort.h                       # Merge Sort
│   ├── moe_v2_mrgsort_out.h                   # Merge Sort输出
│   ├── moe_v2_src_to_dst_op.h                 # 源到目标映射(SIMD)
│   ├── moe_v2_src_to_dst_op_simt.h            # 源到目标映射(SIMT)
│   ├── moe_v2_src_to_dst_with_capacity.h      # 带容量限制映射
│   ├── moe_v2_gather_out.h                    # Gather输出
│   ├── moe_v2_expert_token_out.h              # Expert Token输出
│   └── arch35/                                # arch35专用(SIMT+regbase)
├── op_graph/
│   └── moe_init_routing_v2_proto.h
├── docs/
│   ├── aclnnMoeInitRoutingV2.md
│   └── MoeInitRoutingV2算子设计介绍.md
├── examples/
│   └── test_aclnn_moe_init_routing_v2.cpp
└── tests/ut/
```"""
    ),
    "moe_init_routing_v3_full_analysis.md": (
        "moe_init_routing_v3",
        """```
moe_init_routing_v3/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_v3_def.cpp             # 算子定义
│   ├── moe_init_routing_v3_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_v3_tiling.cpp          # Tiling策略
│   ├── moe_init_routing_v3_tiling_arch35.cpp   # arch35 Tiling
│   ├── moe_init_routing_v3_tiling_base.cpp     # Tiling基类
│   ├── moe_init_routing_v3_infershape.cpp      # Shape推导
│   ├── config/                                 # SoC配置(910b/910_93/950)
│   └── op_api/                                 # ACLNN API
│       ├── aclnn_moe_init_routing_v3.h
│       ├── aclnn_moe_init_routing_v3.cpp
│       ├── moe_init_routing_v3.h
│       └── moe_init_routing_v3.cpp
├── op_kernel/
│   ├── moe_init_routing_v3.cpp                 # Kernel入口
│   ├── moe_init_routing_v3_apt.cpp             # APT入口
│   ├── moe_v3_common.h                        # 公共常量
│   ├── moe_v3_full_load.h                     # 全载主入口
│   ├── moe_v3_full_load_base.h                # 全载基类
│   ├── moe_v3_full_load_unquantized.h         # 非量化全载
│   ├── moe_v3_full_load_static_quant.h        # 静态量化全载
│   ├── moe_v3_full_load_dynamic_quant.h       # 动态量化全载
│   ├── moe_v3_gather_out.h                    # Gather输出
│   ├── moe_v3_gather_out_droppad.h            # DropPad Gather
│   ├── moe_v3_gather_static_quant.h           # 静态量化Gather
│   ├── moe_v3_gather_dynamic_quant.h          # 动态量化Gather
│   ├── moe_v3_gather_droppad_static_quant.h   # DropPad静态量化Gather
│   ├── moe_v3_sort_multi_core.h               # 多核排序
│   ├── moe_v3_mrgsort.h                       # Merge Sort
│   ├── moe_v3_mrgsort_out.h                   # Merge Sort输出
│   ├── moe_v3_mrgsort_performance.h           # 性能优化Merge Sort
│   ├── moe_v3_mrgsort_out_performance.h       # 性能优化Sort输出
│   ├── moe_v3_gather_sort_multi_core.h        # Gather排序多核
│   ├── moe_v3_row_idx_gather.h                # 行索引Gather
│   ├── moe_v3_expert_tokens_count.h           # Expert Token计数
│   └── arch35/                                # arch35专用
├── op_graph/
│   └── moe_init_routing_v3_proto.h
├── docs/
│   └── aclnnMoeInitRoutingV3.md
├── examples/
│   ├── test_aclnn_moe_init_routing_v3.cpp
│   └── test_geir_moe_init_routing_v3.cpp
└── tests/ut/
```"""
    ),
    "moe_init_routing_quant_full_analysis.md": (
        "moe_init_routing_quant",
        """```
moe_init_routing_quant/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_quant_def.cpp             # 算子定义
│   ├── moe_init_routing_quant_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_quant_tiling.cpp          # Tiling策略
│   ├── moe_init_routing_quant_infershape.cpp      # Shape推导
│   └── config/                                    # SoC配置(910b/910_93)
├── op_kernel/
│   ├── moe_init_routing_quant.cpp                 # Kernel入口
│   ├── moe_quant_common.h                        # 量化公共定义
│   ├── moe_quant_sort_base.h                     # 排序基类
│   ├── moe_quant_sort_one_core.h                 # 单核排序
│   ├── moe_quant_sort_multi_core.h               # 多核排序
│   ├── moe_quant_mrgsort.h                       # Merge Sort
│   ├── moe_quant_mrgsort_out.h                   # Merge Sort输出
│   ├── moe_quant_src_to_dst_op.h                 # 源到目标映射
│   ├── moe_quant_gather_out.h                    # Gather输出
│   └── moe_quant_gather_out_small_activate_row.h # 小激活行Gather
├── op_graph/
│   └── moe_init_routing_quant_proto.h
├── docs/
│   └── aclnnMoeInitRoutingQuant.md
├── examples/
│   └── test_aclnn_moe_init_routing_quant.cpp
└── tests/ut/
```"""
    ),
    "moe_init_routing_quant_v2_full_analysis.md": (
        "moe_init_routing_quant_v2",
        """```
moe_init_routing_quant_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_quant_v2_def.cpp             # 算子定义
│   ├── moe_init_routing_quant_v2_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_quant_v2_tiling.cpp          # Tiling策略
│   ├── moe_init_routing_quant_v2_tiling_base.cpp     # Tiling基类
│   ├── moe_init_routing_quant_v2_tiling_base.h       # Tiling基类头文件
│   ├── moe_init_routing_quant_v2_infershape.cpp      # Shape推导
│   └── config/                                       # SoC配置(910b/910_93/950)
├── op_kernel/
│   ├── moe_init_routing_quant_v2.cpp                 # Kernel入口
│   └── (复用moe_init_routing_v2的kernel模块)
├── inner/                                            # 内部共享模块
│   └── (与moe_init_routing_v2共享的排序/Gather模块)
├── op_graph/
│   └── moe_init_routing_quant_v2_proto.h
├── docs/
│   └── aclnnMoeInitRoutingQuantV2.md
├── examples/
│   └── test_aclnn_moe_init_routing_quant_v2.cpp
└── tests/ut/
```"""
    ),
    "moe_init_routing_v2_grad_full_analysis.md": (
        "moe_init_routing_v2_grad",
        """```
moe_init_routing_v2_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_v2_grad_def.cpp                # 算子定义
│   ├── moe_init_routing_v2_grad_tiling.h               # Tiling数据结构
│   ├── moe_init_routing_v2_grad_tiling.cpp             # Tiling策略
│   ├── moe_init_routing_v2_grad_tiling_arch35.cpp      # arch35 Tiling主入口
│   ├── moe_init_routing_v2_grad_tiling_base.cpp        # Tiling基类
│   ├── *_full_load_tiling_arch35.cpp                   # arch35全载Tiling
│   ├── *_split_h_tiling_arch35.cpp                     # arch35切H Tiling
│   ├── moe_init_routing_v2_grad_infershape.cpp         # Shape推导
│   ├── config/                                         # SoC配置(910b/910_93/950)
│   └── op_api/                                         # ACLNN API
│       ├── aclnn_moe_init_routing_v2_grad.h
│       └── aclnn_moe_init_routing_v2_grad.cpp
├── op_kernel/
│   ├── moe_init_routing_v2_grad.cpp                    # Kernel入口
│   ├── moe_init_routing_v2_grad_apt.cpp                # APT入口
│   ├── moe_init_routing_v2_grad_base.h                 # 基础模板
│   ├── *_with_activate.h                               # Activate模式模板
│   ├── *_with_dropless.h                               # Dropless模式模板
│   ├── *_with_pos_drop_and_pad_zero.h                  # PositionPad0模式模板
│   └── arch35/                                         # arch35专用(regbase)
├── op_graph/
│   └── moe_init_routing_v2_grad_proto.h
├── docs/
│   └── aclnnMoeInitRoutingV2Grad.md
├── examples/
│   └── test_aclnn_moe_init_routing_v2_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_token_permute_full_analysis.md": (
        "moe_token_permute",
        """```
moe_token_permute/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_def.cpp             # 算子定义
│   ├── moe_token_permute_tiling.h            # Tiling数据结构
│   ├── moe_token_permute_tiling.cpp          # Tiling策略
│   ├── moe_token_permute_infershape.cpp      # Shape推导
│   ├── config/                               # SoC配置(910b/910_93)
│   └── op_api/                               # ACLNN API
│       ├── aclnn_moe_token_permute.h
│       └── aclnn_moe_token_permute.cpp
├── op_kernel/
│   ├── moe_token_permute.cpp                 # Kernel入口
│   ├── moe_common.h                          # 公共常量
│   ├── moe_sort_base.h                       # 排序基类
│   ├── moe_sort_one_core.h                   # 单核Bitonic Sort
│   ├── moe_sort_multi_core.h                 # 多核Merge Sort
│   ├── moe_mrgsort.h                         # Merge Sort
│   ├── moe_mrgsort_out.h                     # Merge Sort输出
│   ├── moe_index_copy.h                      # 索引拷贝(搬运token)
│   └── moe_index_copy_spilt_d.h              # 切D维度索引拷贝
├── docs/
├── examples/
│   └── test_aclnn_moe_token_permute.cpp
└── tests/ut/
```"""
    ),
    "moe_token_permute_grad_full_analysis.md": (
        "moe_token_permute_grad",
        """```
moe_token_permute_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_grad_def.cpp             # 算子定义
│   ├── moe_token_permute_grad_tiling.h            # Tiling数据结构
│   ├── moe_token_permute_grad_tiling.cpp          # Tiling策略
│   ├── moe_token_permute_grad_infershape.cpp      # Shape推导
│   ├── config/                                    # SoC配置(910b/910_93)
│   └── op_api/                                    # ACLNN API
│       ├── aclnn_moe_token_permute_grad.h
│       └── aclnn_moe_token_permute_grad.cpp
├── op_kernel/
│   ├── moe_token_permute_grad.cpp                 # Kernel入口
│   └── moe_token_permute_grad.h                   # 反向Unpermute模板
├── docs/
├── examples/
│   └── test_aclnn_moe_token_permute_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_token_permute_with_ep_full_analysis.md": (
        "moe_token_permute_with_ep",
        """```
moe_token_permute_with_ep/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_ep_def.cpp           # 算子定义
│   ├── moe_token_permute_with_ep_tiling.h          # Tiling数据结构
│   ├── moe_token_permute_with_ep_tiling.cpp        # Tiling策略
│   ├── moe_token_permute_with_ep_infershape.cpp    # Shape推导
│   ├── moe_token_permute_tiling_base.h             # 共享Tiling基类
│   └── config/                                     # SoC配置(910b/910_93)
├── op_kernel/
│   ├── moe_token_permute_with_ep.cpp               # Kernel入口
│   ├── moe_common.h                                # 公共常量
│   ├── moe_sort_base.h                             # 排序基类
│   ├── moe_sort_one_core.h                         # 单核排序
│   ├── moe_sort_multi_core.h                       # 多核排序
│   ├── moe_mrgsort.h                               # Merge Sort
│   ├── moe_mrgsort_out.h                           # Merge Sort输出
│   ├── moe_index_copy.h                            # 索引拷贝
│   └── moe_index_copy_spilt_d.h                    # 切D维度索引拷贝
├── docs/
├── examples/
│   └── test_aclnn_moe_token_permute_with_ep.cpp
└── tests/ut/
```"""
    ),
    "moe_token_permute_with_ep_grad_full_analysis.md": (
        "moe_token_permute_with_ep_grad",
        """```
moe_token_permute_with_ep_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_ep_grad_def.cpp           # 算子定义
│   ├── moe_token_permute_with_ep_grad_tiling.h          # Tiling数据结构
│   ├── moe_token_permute_with_ep_grad_tiling.cpp        # Tiling策略
│   ├── moe_token_permute_with_ep_grad_infershape.cpp    # Shape推导
│   └── config/                                          # SoC配置(910b/910_93)
├── op_kernel/
│   ├── moe_token_permute_with_ep_grad.cpp               # Kernel入口
│   ├── moe_token_permute_with_ep_grad.h                 # EP反向模板
│   ├── moe_token_unpermute_grad_base.h                  # Unpermute基类
│   ├── moe_token_unpermute_grad_prob_none.h             # 无prob反向
│   └── moe_token_unpermute_grad_prob_not_none.h         # 有prob反向
├── docs/
├── examples/
│   └── test_aclnn_moe_token_permute_with_ep_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_token_permute_with_routing_map_full_analysis.md": (
        "moe_token_permute_with_routing_map",
        """```
moe_token_permute_with_routing_map/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_routing_map_def.cpp           # 算子定义
│   ├── moe_token_permute_with_routing_map_tiling.h          # Tiling数据结构
│   ├── moe_token_permute_with_routing_map_tiling.cpp        # Tiling策略
│   ├── config/                                              # SoC配置(910b/910_93)
│   └── op_api/                                              # ACLNN API
│       ├── aclnn_moe_token_permute_with_routing_map.h
│       ├── aclnn_moe_token_permute_with_routing_map.cpp
│       ├── moe_token_permute_with_routing_map.h
│       └── moe_token_permute_with_routing_map.cpp
├── op_kernel/
│   ├── moe_token_permute_with_routing_map.cpp               # Kernel入口
│   ├── moe_common.h                                         # 公共常量
│   ├── masked_select_v3.h                                   # MaskedSelect V3
│   ├── moe_sort_base.h / moe_sort_one_core.h / moe_sort_multi_core.h  # 排序模块
│   ├── moe_mrgsort.h / moe_mrgsort_out.h                   # Merge Sort
│   ├── moe_index_copy.h / moe_index_copy_split_d.h         # 索引拷贝
│   ├── moe_sort_multi_core_last_dim.h                       # 按最后一维排序
│   ├── moe_permute_prob.h                                   # Prob重排
│   └── arch35/
│       └── gather_v2_simd_two_dim.h                         # arch35 Gather
├── docs/
├── examples/
│   └── test_aclnn_moe_token_permute_with_routing_map.cpp
└── tests/ut/
```"""
    ),
    "moe_token_permute_with_routing_map_grad_full_analysis.md": (
        "moe_token_permute_with_routing_map_grad",
        """```
moe_token_permute_with_routing_map_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_routing_map_grad_def.cpp           # 算子定义
│   ├── moe_token_permute_with_routing_map_grad_tiling.h          # Tiling数据结构
│   ├── moe_token_permute_with_routing_map_grad_tiling.cpp        # Tiling策略
│   ├── moe_token_permute_with_routing_map_grad_infershape.cpp    # Shape推导
│   ├── moe_token_permute_grad_tiling.h                           # 复用基础版Tiling
│   ├── config/                                                   # SoC配置(910b/910_93)
│   └── op_api/                                                   # ACLNN API
│       ├── aclnn_moe_token_permute_with_routing_map_grad.h
│       ├── aclnn_moe_token_permute_with_routing_map_grad.cpp
│       ├── moe_token_permute_with_routing_map_grad.h
│       └── moe_token_permute_with_routing_map_grad.cpp
├── op_kernel/
│   ├── moe_token_permute_with_routing_map_grad.cpp               # Kernel入口
│   ├── moe_token_permute_with_routing_map_grad.h                 # RoutingMap反向模板
│   └── moe_token_unpermute.h                                     # 复用Unpermute
├── docs/
├── examples/
│   └── test_aclnn_moe_token_permute_with_routing_map_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_token_unpermute_full_analysis.md": (
        "moe_token_unpermute",
        """```
moe_token_unpermute/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_def.cpp             # 算子定义
│   ├── moe_token_unpermute_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_infershape.cpp      # Shape推导
│   ├── config/                                 # SoC配置(910b/910_93/310p/kirin)
│   └── op_api/                                 # ACLNN API
│       ├── aclnn_moe_token_unpermute.h
│       └── aclnn_moe_token_unpermute.cpp
├── op_kernel/
│   ├── moe_token_unpermute.cpp                 # Kernel入口
│   ├── moe_token_unpermute.h                   # 主计算模板
│   └── moe_token_unpermute_common.h            # 公共定义
├── docs/
├── examples/
│   └── test_aclnn_moe_token_unpermute.cpp
└── tests/ut/
```"""
    ),
    "moe_token_unpermute_grad_full_analysis.md": (
        "moe_token_unpermute_grad",
        """```
moe_token_unpermute_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_grad_def.cpp             # 算子定义
│   ├── moe_token_unpermute_grad_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_grad_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_grad_infershape.cpp      # Shape推导
│   ├── config/                                      # SoC配置(910b/910_93)
│   └── op_api/                                      # ACLNN API
│       ├── aclnn_moe_token_unpermute_grad.h
│       └── aclnn_moe_token_unpermute_grad.cpp
├── op_kernel/
│   ├── moe_token_unpermute_grad.cpp                 # Kernel入口
│   ├── moe_token_unpermute_grad_base.h              # 基础模板
│   ├── moe_token_unpermute_grad_prob_none.h         # 无prob反向
│   └── moe_token_unpermute_grad_prob_not_none.h     # 有prob反向
├── docs/
├── examples/
│   └── test_aclnn_moe_token_unpermute_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_token_unpermute_with_ep_full_analysis.md": (
        "moe_token_unpermute_with_ep",
        """```
moe_token_unpermute_with_ep/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_with_ep_def.cpp             # 算子定义
│   ├── moe_token_unpermute_with_ep_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_with_ep_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_with_ep_infershape.cpp      # Shape推导
│   └── config/                                         # SoC配置(910b/910_93/kirin)
├── op_kernel/
│   ├── moe_token_unpermute_with_ep.cpp                 # Kernel入口
│   └── moe_token_unpermute_with_ep.h                   # EP版Unpermute模板
├── docs/
├── examples/
│   └── test_aclnn_moe_token_unpermute_with_ep.cpp
└── tests/ut/
```"""
    ),
    "moe_token_unpermute_with_ep_grad_full_analysis.md": (
        "moe_token_unpermute_with_ep_grad",
        """```
moe_token_unpermute_with_ep_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_with_ep_grad_def.cpp             # 算子定义
│   ├── moe_token_unpermute_with_ep_grad_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_with_ep_grad_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_with_ep_grad_infershape.cpp      # Shape推导
│   └── config/                                              # SoC配置(910b/910_93)
├── op_kernel/
│   ├── moe_token_unpermute_with_ep_grad.cpp                 # Kernel入口
│   ├── moe_token_unpermute_grad_base.h                      # Unpermute反向基类
│   ├── moe_token_unpermute_grad_prob_none.h                 # 无prob反向
│   └── moe_token_unpermute_grad_prob_not_none.h             # 有prob反向
├── docs/
├── examples/
│   └── test_aclnn_moe_token_unpermute_with_ep_grad.cpp
└── tests/ut/
```"""
    ),
    "moe_token_unpermute_with_routing_map_full_analysis.md": (
        "moe_token_unpermute_with_routing_map",
        """```
moe_token_unpermute_with_routing_map/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_with_routing_map_def.cpp             # 算子定义
│   ├── moe_token_unpermute_with_routing_map_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_with_routing_map_tiling.cpp          # Tiling策略
│   ├── config/                                                  # SoC配置(910b/910_93/kirin)
│   └── op_api/                                                  # ACLNN API
│       ├── aclnn_moe_token_unpermute_with_routing_map.h
│       ├── aclnn_moe_token_unpermute_with_routing_map.cpp
│       ├── moe_token_unpermute_with_routing_map.h
│       └── moe_token_unpermute_with_routing_map.cpp
├── op_kernel/
│   ├── moe_token_unpermute_with_routing_map.cpp                 # Kernel入口
│   ├── moe_token_unpermute_with_routing_map_not_pad.h           # 非Pad模式模板
│   ├── moe_token_unpermute_with_routing_map_pad.h               # Pad模式模板
│   ├── moe_sort_base_with_routing_map.h                         # 排序基类
│   └── masked_select.h                                          # MaskedSelect
├── docs/
├── examples/
│   └── test_aclnn_moe_token_unpermute_with_routing_map.cpp
└── tests/ut/
```"""
    ),
    "moe_token_unpermute_with_routing_map_grad_full_analysis.md": (
        "moe_token_unpermute_with_routing_map_grad",
        """```
moe_token_unpermute_with_routing_map_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_with_routing_map_grad_def.cpp             # 算子定义
│   ├── moe_token_unpermute_with_routing_map_grad_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_with_routing_map_grad_tiling.cpp          # Tiling策略
│   ├── config/                                                       # SoC配置(910b/910_93)
│   └── op_api/                                                       # ACLNN API
├── op_kernel/
│   ├── moe_token_unpermute_with_routing_map_grad.cpp                 # Kernel入口
│   ├── moe_token_unpermute_with_routing_map_grad_apt.cpp             # APT入口
│   ├── moe_token_unpermute_with_routing_map_grad_base.h              # 基础模板
│   ├── *_prob_none_drop_pad_false.h                                  # 无prob非Pad
│   ├── *_prob_none_drop_pad_true.h                                   # 无prob有Pad
│   ├── *_prob_not_none_drop_pad_false.h                              # 有prob非Pad
│   ├── *_prob_not_none_drop_pad_true.h                               # 有prob有Pad
│   └── arch35/                                                       # arch35专用(regbase)
├── docs/
├── examples/
│   └── test_aclnn_moe_token_unpermute_with_routing_map_grad.cpp
└── tests/ut/
```"""
    ),
    "3rd_moe_gather_v2_full_analysis.md": (
        "3rd/moe_gather_v2",
        """```
3rd/moe_gather_v2/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── op_api/                                  # ACLNN API（轻量级L0算子）
│       ├── moe_gather_v2.h                      # API头文件
│       └── moe_gather_v2.cpp                    # API实现（复用CANN内置GatherV2）
└── (无自定义kernel/tiling/infershape)
```"""
    ),
    "3rd_moe_inplace_index_add_full_analysis.md": (
        "3rd/moe_inplace_index_add",
        """```
3rd/moe_inplace_index_add/
├── CMakeLists.txt
├── op_host/
│   ├── moe_inplace_index_add_def.cpp                     # 算子定义
│   ├── arch35/                                           # arch35专用Tiling
│   │   ├── moe_inplace_index_add_tiling.h                # Tiling数据结构
│   │   ├── moe_inplace_index_add_tiling.cpp              # Tiling主入口
│   │   ├── moe_inplace_index_add_tiling_arch35.h/.cpp    # arch35 Tiling基类
│   │   ├── *_base_tiling.cpp                             # 基础Tiling
│   │   ├── *_simd_tiling.h/.cpp                          # SIMD Tiling
│   │   ├── *_simt_tiling.h/.cpp                          # SIMT Tiling
│   │   ├── *_simd_sort_tiling.h/.cpp                     # SIMD排序Tiling
│   │   ├── *_simt_sort_tiling.h/.cpp                     # SIMT排序Tiling
│   │   └── *_determinstic_tiling.h/.cpp                  # 确定性Tiling
│   └── op_api/                                           # ACLNN API
│       ├── moe_inplace_index_add.h
│       └── moe_inplace_index_add.cpp
├── op_kernel/
│   ├── moe_inplace_index_add.cpp                         # Kernel入口
│   └── arch35/                                           # arch35专用Kernel
│       ├── moe_inplace_index_add_common.h                # 公共定义
│       ├── moe_inplace_index_add_simd.h                  # SIMD模板
│       ├── moe_inplace_index_add_simt.h                  # SIMT模板
│       ├── moe_inplace_index_add_simd_sort.h             # SIMD排序模板
│       ├── moe_inplace_index_add_simt_sort.h             # SIMT排序模板
│       ├── moe_inplace_index_add_determinstic.h          # 确定性模板
│       ├── moe_inplace_index_add_determinstic_notquant.h # 非量化确定性
│       └── indices_sort_utils.h                          # 索引排序工具
└── tests/ut/
```"""
    ),
    "3rd_moe_inplace_index_add_with_sorted_full_analysis.md": (
        "3rd/moe_inplace_index_add_with_sorted",
        """```
3rd/moe_inplace_index_add_with_sorted/
├── CMakeLists.txt
├── op_host/
│   ├── moe_inplace_index_add_with_sorted_def.cpp         # 算子定义
│   ├── moe_inplace_index_add_with_sorted_tiling.h        # Tiling数据结构
│   ├── moe_inplace_index_add_with_sorted_tiling.cpp      # Tiling策略
│   └── op_api/                                           # ACLNN API
│       ├── moe_inplace_index_add_with_sorted.h
│       └── moe_inplace_index_add_with_sorted.cpp
├── op_kernel/
│   ├── moe_inplace_index_add_with_sorted.cpp             # Kernel入口
│   ├── moe_inplace_index_add_with_sorted_base.h          # 基础模板
│   ├── moe_inplace_index_add_with_sorted_avg.h           # Avg模式模板
│   └── moe_inplace_index_add_with_sorted_fix.h           # Fix模式模板
└── tests/ut/
```"""
    ),
    "3rd_moe_masked_scatter_full_analysis.md": (
        "3rd/moe_masked_scatter",
        """```
3rd/moe_masked_scatter/
├── CMakeLists.txt
├── op_host/
│   ├── moe_masked_scatter_def.cpp                        # 算子定义
│   ├── arch35/                                           # arch35专用Tiling
│   │   ├── moe_masked_scatter_tiling.h
│   │   └── moe_masked_scatter_tiling.cpp
│   └── op_api/                                           # ACLNN API
│       ├── moe_masked_scatter.h
│       └── moe_masked_scatter.cpp
├── op_kernel/
│   ├── moe_masked_scatter.cpp                            # Kernel入口
│   └── arch35/
│       └── moe_masked_scatter.h                          # arch35 Kernel模板
└── tests/ut/
```"""
    ),
}


def process_report(report_filename, op_dir, dir_tree):
    filepath = os.path.join(DOCS_DIR, report_filename)
    if not os.path.exists(filepath):
        print(f"  SKIP: {report_filename} not found")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already has directory structure section
    if "目录结构与文件职责" in content:
        print(f"  SKIP: {report_filename} already has directory structure")
        return False

    # Find all section headers
    section_pattern = re.compile(r'^## (\d+)\. (.+)$', re.MULTILINE)
    sections = list(section_pattern.finditer(content))

    if not sections:
        print(f"  SKIP: {report_filename} - no numbered sections found")
        return False

    # Find max section number
    max_num = max(int(m.group(1)) for m in sections)

    # Renumber from highest to lowest (to avoid collisions)
    for n in range(max_num, 1, -1):
        content = re.sub(
            f'^## {n}\\. ',
            f'## {n+1}. ',
            content,
            flags=re.MULTILINE
        )

    # Insert directory structure after "## 1." section
    # Find the end of section 1 (next section starts at ## 2. which is now renumbered to ## 3.)
    # Actually, after renumbering, old ## 2. is now ## 3.
    # We need to insert before the new ## 3. (which was old ## 2.)

    # Find the position after section 1 (before the next ## section)
    new_section2_match = re.search(r'^## 3\. ', content, re.MULTILINE)
    if not new_section2_match:
        # Try ## 2. in case there was only 1 section originally
        new_section2_match = re.search(r'^## 2\. ', content, re.MULTILINE)
        if not new_section2_match:
            print(f"  SKIP: {report_filename} - can't find insertion point")
            return False

    insert_pos = new_section2_match.start()

    # Build the new section
    new_section = f"\n## 2. 目录结构与文件职责\n\n{dir_tree}\n\n"

    content = content[:insert_pos] + new_section + content[insert_pos:]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  OK: {report_filename} - added directory structure, renumbered sections")
    return True


def main():
    success = 0
    failed = 0
    skipped = 0

    for report_filename, (op_dir, dir_tree) in REPORTS.items():
        print(f"Processing: {report_filename}")
        try:
            result = process_report(report_filename, op_dir, dir_tree)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n=== Summary ===")
    print(f"Success: {success}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
