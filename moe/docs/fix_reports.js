#!/usr/bin/env node
/**
 * Insert directory structure section and renumber sections in analysis reports.
 */
const fs = require('fs');
const path = require('path');

const DOCS_DIR = __dirname;

const REPORTS = {
  "moe_finalize_routing_v2_grad_full_analysis.md": {
    tree: `moe_finalize_routing_v2_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_finalize_routing_v2_grad_def.cpp             # 算子定义
│   ├── moe_finalize_routing_v2_grad_tiling.h            # Tiling数据结构
│   ├── moe_finalize_routing_v2_grad_tiling.cpp          # Tiling策略实现
│   ├── moe_finalize_routing_v2_grad_tiling_membase.cpp  # membase Tiling
│   ├── moe_finalize_routing_v2_grad_infershape.cpp      # Shape推导
│   ├── moe_finalize_routing_v2_grad_tiling_arch35.cpp   # arch35 Tiling主入口
│   ├── *_split_h / *_not_split_h / *_without_scale_tiling_arch35.cpp  # arch35子策略
│   ├── config/                                          # SoC配置(910b/910_93/950)
│   └── op_api/                                          # ACLNN API
├── op_kernel/
│   ├── moe_finalize_routing_v2_grad.cpp                 # Kernel入口
│   ├── moe_finalize_routing_v2_grad_base.h              # 基础模板
│   ├── *_with_scale_cut_h / *_with_scale_not_cut_h.h    # 带scale模板
│   ├── *_without_scale_cut_h / *_without_scale_not_cut_h.h  # 无scale模板
│   └── arch35/                                          # arch35专用(regbase)
├── op_graph/ └── moe_finalize_routing_v2_grad_proto.h
├── docs/ └── aclnnMoeFinalizeRoutingV2Grad.md
├── examples/ └── test_aclnn_moe_finalize_routing_v2_grad.cpp
└── tests/ut/`
  },
  "moe_gating_top_k_full_analysis.md": {
    tree: `moe_gating_top_k/
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
│   ├── moe_gating_top_k_e_k_fullload.h        # E和K全载模板
│   ├── moe_gating_top_k_generalized.h         # 通用分组模板(DeepSeekV3)
│   ├── moe_gating_top_k_without_group.h       # 无分组模板
│   ├── common.h                               # 公共常量
│   └── arch35/                                # arch35专用(regbase)
├── op_graph/ └── moe_gating_top_k_proto.h
├── docs/ └── aclnnMoeGatingTopK.md
├── examples/
│   ├── test_aclnn_moe_gating_top_k.cpp
│   └── test_geir_moe_gating_top_k.cpp
└── tests/ut/`
  },
  "moe_gating_top_k_softmax_full_analysis.md": {
    tree: `moe_gating_top_k_softmax/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_softmax_def.cpp               # 算子定义
│   ├── moe_gating_top_k_softmax_tiling.h              # Tiling数据结构
│   ├── moe_gating_top_k_softmax_tiling.cpp            # Tiling策略
│   ├── moe_gating_top_k_softmax_tiling_base.cpp/.h    # Tiling基类
│   ├── moe_gating_top_k_softmax_tiling_arch35.cpp     # arch35 Tiling
│   ├── *_k_full_load.cpp / *_e_k_full_load.cpp        # 全载Tiling变体
│   ├── moe_gating_top_k_softmax_infershape.cpp        # Shape推导
│   └── config/                                        # SoC配置(910b/910_93/950)
├── op_kernel/
│   ├── moe_gating_top_k_softmax.cpp                   # Kernel入口
│   ├── *_k_fullload.h / *_e_k_fullload.h              # 全载模板
│   └── arch35/                                        # arch35专用(regbase)
├── op_graph/ └── moe_gating_top_k_softmax_proto.h
├── framework/ └── npu_moe_gating_top_k_softmax_onnx_plugin.cpp
├── docs/ └── aclnnMoeGatingTopKSoftmax.md
├── examples/ └── test_aclnn_moe_gating_top_k_softmax.cpp
└── tests/ut/`
  },
  "moe_gating_top_k_softmax_v2_full_analysis.md": {
    tree: `moe_gating_top_k_softmax_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_softmax_v2_def.cpp                    # 算子定义
│   ├── moe_gating_top_k_softmax_v2_tiling.h                   # Tiling数据结构
│   ├── moe_gating_top_k_softmax_v2_tiling.cpp                 # Tiling策略
│   ├── *_k_fullload.cpp / *_ek_fullload.cpp / *_k_renorm.cpp  # 全载/重归一化Tiling
│   ├── moe_gating_top_k_softmax_v2_infershape.cpp             # Shape推导
│   └── config/                                                # SoC配置(910b/950)
├── op_kernel/
│   ├── moe_gating_top_k_softmax_v2.cpp                        # Kernel入口
│   ├── *_k_fullload.h / *_ek_fullload.h / *_k_renorm.h       # 全载/重归一化模板
│   └── arch35/                                                # arch35专用(regbase)
├── op_graph/ └── moe_gating_top_k_softmax_v2_proto.h
├── docs/ └── aclnnMoeGatingTopKSoftmaxV2.md
├── examples/ └── test_aclnn_moe_gating_top_k_softmax_v2.cpp
└── tests/ut/`
  },
  "moe_fused_topk_full_analysis.md": {
    tree: `moe_fused_topk/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_fused_topk_def.cpp               # 算子定义
│   ├── moe_fused_topk_tiling.h              # Tiling数据结构
│   ├── moe_fused_topk_tiling.cpp            # Tiling策略
│   ├── config/                              # SoC配置(910b/910_93)
│   └── op_api/                              # ACLNN API
│       ├── aclnn_moe_fused_topk.h/.cpp
│       └── moe_fused_topk.h/.cpp
├── op_kernel/
│   ├── moe_fused_topk.cpp                   # Kernel入口
│   └── moe_fused_topk.h                     # Kernel模板(分组TopK+映射表)
├── docs/ └── aclnnMoeFusedTopk.md
├── examples/ └── test_aclnn_moe_fused_topk.cpp
└── tests/ut/`
  },
  "moe_init_routing_full_analysis.md": {
    tree: `moe_init_routing/
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
│   ├── moe_common.h                         # 公共常量和工具
│   ├── moe_init_routing_fullload.h          # 全载模板
│   ├── moe_sort_base.h / moe_sort_one_core.h / moe_sort_multi_core.h  # 排序模块
│   ├── moe_mrgsort.h / moe_mrgsort_out.h   # Merge Sort
│   ├── moe_src_to_dst_op.h                 # 源到目标映射(SIMD)
│   ├── moe_gather_out.h                     # Gather输出模板
│   └── arch35/                              # arch35专用(SIMT+regbase)
├── op_graph/ └── moe_init_routing_proto.h
├── framework/ └── npu_moe_init_routing_onnx_plugin.cpp
├── docs/ └── aclnnMoeInitRouting.md
├── examples/ └── test_aclnn_moe_init_routing.cpp
└── tests/ut/`
  },
  "moe_init_routing_v2_full_analysis.md": {
    tree: `moe_init_routing_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_v2_def.cpp             # 算子定义
│   ├── moe_init_routing_v2_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_v2_tiling.cpp          # Tiling策略实现
│   ├── moe_init_routing_v2_tiling_util.h       # Tiling工具函数
│   ├── moe_init_routing_v2_infershape.cpp      # Shape推导
│   └── op_api/                                 # ACLNN API
├── op_kernel/
│   ├── moe_init_routing_v2.cpp                 # Kernel入口
│   ├── moe_v2_common.h                        # 公共常量
│   ├── moe_v2_init_routing_fullload.h         # 全载模板
│   ├── moe_v2_sort_*.h                        # 排序模块(单核/多核)
│   ├── moe_v2_mrgsort.h / moe_v2_mrgsort_out.h  # Merge Sort
│   ├── moe_v2_src_to_dst_op.h / *_simt.h / *_with_capacity.h  # 源到目标映射
│   ├── moe_v2_gather_out.h / moe_v2_expert_token_out.h  # 输出模块
│   └── arch35/                                # arch35专用(SIMT+regbase)
├── op_graph/ └── moe_init_routing_v2_proto.h
├── docs/
│   ├── aclnnMoeInitRoutingV2.md
│   └── MoeInitRoutingV2算子设计介绍.md
├── examples/ └── test_aclnn_moe_init_routing_v2.cpp
└── tests/ut/`
  },
  "moe_init_routing_v3_full_analysis.md": {
    tree: `moe_init_routing_v3/
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
├── op_kernel/
│   ├── moe_init_routing_v3.cpp                 # Kernel入口
│   ├── moe_v3_common.h                        # 公共常量
│   ├── moe_v3_full_load*.h                    # 全载模板(非量化/静态量化/动态量化)
│   ├── moe_v3_gather_*.h                      # Gather输出(含DropPad/量化变体)
│   ├── moe_v3_sort/mrgsort*.h                 # 排序模块
│   ├── moe_v3_expert_tokens_count.h           # Expert Token计数
│   └── arch35/                                # arch35专用
├── op_graph/ └── moe_init_routing_v3_proto.h
├── docs/ └── aclnnMoeInitRoutingV3.md
├── examples/
│   ├── test_aclnn_moe_init_routing_v3.cpp
│   └── test_geir_moe_init_routing_v3.cpp
└── tests/ut/`
  },
  "moe_init_routing_quant_full_analysis.md": {
    tree: `moe_init_routing_quant/
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
│   ├── moe_quant_sort_*.h                        # 排序模块
│   ├── moe_quant_mrgsort*.h                      # Merge Sort
│   ├── moe_quant_src_to_dst_op.h                 # 源到目标映射
│   ├── moe_quant_gather_out*.h                   # Gather输出
├── op_graph/ └── moe_init_routing_quant_proto.h
├── docs/ └── aclnnMoeInitRoutingQuant.md
├── examples/ └── test_aclnn_moe_init_routing_quant.cpp
└── tests/ut/`
  },
  "moe_init_routing_quant_v2_full_analysis.md": {
    tree: `moe_init_routing_quant_v2/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_quant_v2_def.cpp             # 算子定义
│   ├── moe_init_routing_quant_v2_tiling.h            # Tiling数据结构
│   ├── moe_init_routing_quant_v2_tiling.cpp          # Tiling策略
│   ├── moe_init_routing_quant_v2_tiling_base.cpp/.h  # Tiling基类
│   ├── moe_init_routing_quant_v2_infershape.cpp      # Shape推导
│   └── config/                                       # SoC配置(910b/910_93/950)
├── op_kernel/
│   └── moe_init_routing_quant_v2.cpp                 # Kernel入口(复用v2排序/Gather)
├── inner/                                            # 内部共享排序/Gather模块
├── op_graph/ └── moe_init_routing_quant_v2_proto.h
├── docs/ └── aclnnMoeInitRoutingQuantV2.md
├── examples/ └── test_aclnn_moe_init_routing_quant_v2.cpp
└── tests/ut/`
  },
  "moe_init_routing_v2_grad_full_analysis.md": {
    tree: `moe_init_routing_v2_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_init_routing_v2_grad_def.cpp                # 算子定义
│   ├── moe_init_routing_v2_grad_tiling.h               # Tiling数据结构
│   ├── moe_init_routing_v2_grad_tiling.cpp             # Tiling策略
│   ├── moe_init_routing_v2_grad_tiling_arch35.cpp      # arch35 Tiling主入口
│   ├── moe_init_routing_v2_grad_tiling_base.cpp        # Tiling基类
│   ├── *_full_load / *_split_h_tiling_arch35.cpp       # arch35子策略
│   ├── moe_init_routing_v2_grad_infershape.cpp         # Shape推导
│   ├── config/                                         # SoC配置(910b/910_93/950)
│   └── op_api/                                         # ACLNN API
├── op_kernel/
│   ├── moe_init_routing_v2_grad.cpp                    # Kernel入口
│   ├── moe_init_routing_v2_grad_base.h                 # 基础模板
│   ├── *_with_activate.h / *_with_dropless.h           # Activate/Dropless模式
│   ├── *_with_pos_drop_and_pad_zero.h                  # PositionPad0模式
│   └── arch35/                                         # arch35专用(regbase)
├── op_graph/ └── moe_init_routing_v2_grad_proto.h
├── docs/ └── aclnnMoeInitRoutingV2Grad.md
├── examples/ └── test_aclnn_moe_init_routing_v2_grad.cpp
└── tests/ut/`
  },
  "moe_token_permute_full_analysis.md": {
    tree: `moe_token_permute/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_def.cpp             # 算子定义
│   ├── moe_token_permute_tiling.h            # Tiling数据结构
│   ├── moe_token_permute_tiling.cpp          # Tiling策略
│   ├── moe_token_permute_infershape.cpp      # Shape推导
│   ├── config/                               # SoC配置(910b/910_93)
│   └── op_api/                               # ACLNN API
├── op_kernel/
│   ├── moe_token_permute.cpp                 # Kernel入口
│   ├── moe_common.h                          # 公共常量
│   ├── moe_sort_*.h                          # 排序模块(Bitonic/Merge Sort)
│   ├── moe_mrgsort.h / moe_mrgsort_out.h    # Merge Sort
│   ├── moe_index_copy.h / moe_index_copy_spilt_d.h  # 索引拷贝
├── docs/
├── examples/ └── test_aclnn_moe_token_permute.cpp
└── tests/ut/`
  },
  "moe_token_permute_grad_full_analysis.md": {
    tree: `moe_token_permute_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_grad_def.cpp             # 算子定义
│   ├── moe_token_permute_grad_tiling.h            # Tiling数据结构
│   ├── moe_token_permute_grad_tiling.cpp          # Tiling策略
│   ├── moe_token_permute_grad_infershape.cpp      # Shape推导
│   ├── config/                                    # SoC配置(910b/910_93)
│   └── op_api/                                    # ACLNN API
├── op_kernel/
│   ├── moe_token_permute_grad.cpp                 # Kernel入口
│   └── moe_token_permute_grad.h                   # 反向Unpermute模板
├── docs/
├── examples/ └── test_aclnn_moe_token_permute_grad.cpp
└── tests/ut/`
  },
  "moe_token_permute_with_ep_full_analysis.md": {
    tree: `moe_token_permute_with_ep/
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
│   ├── moe_common.h / moe_sort_*.h                 # 排序公共模块
│   ├── moe_mrgsort.h / moe_mrgsort_out.h           # Merge Sort
│   ├── moe_index_copy.h / moe_index_copy_spilt_d.h # 索引拷贝
├── docs/
├── examples/ └── test_aclnn_moe_token_permute_with_ep.cpp
└── tests/ut/`
  },
  "moe_token_permute_with_ep_grad_full_analysis.md": {
    tree: `moe_token_permute_with_ep_grad/
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
│   ├── moe_token_unpermute_grad_base.h                  # Unpermute反向基类
│   ├── moe_token_unpermute_grad_prob_none.h             # 无prob反向
│   └── moe_token_unpermute_grad_prob_not_none.h         # 有prob反向
├── docs/
├── examples/ └── test_aclnn_moe_token_permute_with_ep_grad.cpp
└── tests/ut/`
  },
  "moe_token_permute_with_routing_map_full_analysis.md": {
    tree: `moe_token_permute_with_routing_map/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_routing_map_def.cpp           # 算子定义
│   ├── moe_token_permute_with_routing_map_tiling.h          # Tiling数据结构
│   ├── moe_token_permute_with_routing_map_tiling.cpp        # Tiling策略
│   ├── config/                                              # SoC配置(910b/910_93)
│   └── op_api/                                              # ACLNN API
├── op_kernel/
│   ├── moe_token_permute_with_routing_map.cpp               # Kernel入口
│   ├── masked_select_v3.h                                   # MaskedSelect V3(提取有效位)
│   ├── moe_sort_*.h / moe_mrgsort*.h                        # 排序模块
│   ├── moe_index_copy.h                                     # 索引拷贝
│   ├── moe_sort_multi_core_last_dim.h                       # 按最后一维排序(dropAndPad)
│   ├── moe_permute_prob.h                                   # Prob重排
│   └── arch35/ └── gather_v2_simd_two_dim.h                 # arch35 Gather
├── docs/
├── examples/ └── test_aclnn_moe_token_permute_with_routing_map.cpp
└── tests/ut/`
  },
  "moe_token_permute_with_routing_map_grad_full_analysis.md": {
    tree: `moe_token_permute_with_routing_map_grad/
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
├── op_kernel/
│   ├── moe_token_permute_with_routing_map_grad.cpp               # Kernel入口
│   ├── moe_token_permute_with_routing_map_grad.h                 # RoutingMap反向模板
│   └── moe_token_unpermute.h                                     # 复用Unpermute
├── docs/
├── examples/ └── test_aclnn_moe_token_permute_with_routing_map_grad.cpp
└── tests/ut/`
  },
  "moe_token_unpermute_full_analysis.md": {
    tree: `moe_token_unpermute/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_def.cpp             # 算子定义
│   ├── moe_token_unpermute_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_infershape.cpp      # Shape推导
│   ├── config/                                 # SoC配置(910b/910_93/310p/kirin)
│   └── op_api/                                 # ACLNN API
├── op_kernel/
│   ├── moe_token_unpermute.cpp                 # Kernel入口
│   ├── moe_token_unpermute.h                   # 主计算模板
│   └── moe_token_unpermute_common.h            # 公共定义
├── docs/
├── examples/ └── test_aclnn_moe_token_unpermute.cpp
└── tests/ut/`
  },
  "moe_token_unpermute_grad_full_analysis.md": {
    tree: `moe_token_unpermute_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_grad_def.cpp             # 算子定义
│   ├── moe_token_unpermute_grad_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_grad_tiling.cpp          # Tiling策略
│   ├── moe_token_unpermute_grad_infershape.cpp      # Shape推导
│   ├── config/                                      # SoC配置(910b/910_93)
│   └── op_api/                                      # ACLNN API
├── op_kernel/
│   ├── moe_token_unpermute_grad.cpp                 # Kernel入口
│   ├── moe_token_unpermute_grad_base.h              # 基础模板
│   ├── moe_token_unpermute_grad_prob_none.h         # 无prob反向
│   └── moe_token_unpermute_grad_prob_not_none.h     # 有prob反向
├── docs/
├── examples/ └── test_aclnn_moe_token_unpermute_grad.cpp
└── tests/ut/`
  },
  "moe_token_unpermute_with_ep_full_analysis.md": {
    tree: `moe_token_unpermute_with_ep/
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
├── examples/ └── test_aclnn_moe_token_unpermute_with_ep.cpp
└── tests/ut/`
  },
  "moe_token_unpermute_with_ep_grad_full_analysis.md": {
    tree: `moe_token_unpermute_with_ep_grad/
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
├── examples/ └── test_aclnn_moe_token_unpermute_with_ep_grad.cpp
└── tests/ut/`
  },
  "moe_token_unpermute_with_routing_map_full_analysis.md": {
    tree: `moe_token_unpermute_with_routing_map/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_unpermute_with_routing_map_def.cpp             # 算子定义
│   ├── moe_token_unpermute_with_routing_map_tiling.h            # Tiling数据结构
│   ├── moe_token_unpermute_with_routing_map_tiling.cpp          # Tiling策略
│   ├── config/                                                  # SoC配置(910b/910_93/kirin)
│   └── op_api/                                                  # ACLNN API
├── op_kernel/
│   ├── moe_token_unpermute_with_routing_map.cpp                 # Kernel入口
│   ├── moe_token_unpermute_with_routing_map_not_pad.h           # 非Pad模式
│   ├── moe_token_unpermute_with_routing_map_pad.h               # Pad模式
│   ├── moe_sort_base_with_routing_map.h                         # 排序基类
│   └── masked_select.h                                          # MaskedSelect
├── docs/
├── examples/ └── test_aclnn_moe_token_unpermute_with_routing_map.cpp
└── tests/ut/`
  },
  "moe_token_unpermute_with_routing_map_grad_full_analysis.md": {
    tree: `moe_token_unpermute_with_routing_map_grad/
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
│   ├── *_prob_none_drop_pad_false.h                                  # 无prob非Pad
│   ├── *_prob_none_drop_pad_true.h                                   # 无prob有Pad
│   ├── *_prob_not_none_drop_pad_false.h                              # 有prob非Pad
│   ├── *_prob_not_none_drop_pad_true.h                               # 有prob有Pad
│   └── arch35/                                                       # arch35专用(regbase)
├── docs/
├── examples/ └── test_aclnn_moe_token_unpermute_with_routing_map_grad.cpp
└── tests/ut/`
  },
  "3rd_moe_gather_v2_full_analysis.md": {
    tree: `3rd/moe_gather_v2/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── op_api/                                  # ACLNN API（轻量级L0算子）
│       ├── moe_gather_v2.h                      # API头文件
│       └── moe_gather_v2.cpp                    # API实现（复用CANN内置GatherV2）
└── (无自定义kernel/tiling/infershape)`
  },
  "3rd_moe_inplace_index_add_full_analysis.md": {
    tree: `3rd/moe_inplace_index_add/
├── CMakeLists.txt
├── op_host/
│   ├── moe_inplace_index_add_def.cpp                     # 算子定义
│   ├── arch35/                                           # arch35专用Tiling
│   │   ├── *_tiling.h/.cpp                               # Tiling主入口
│   │   ├── *_base / *_simd / *_simd_sort_tiling.cpp/.h   # SIMD Tiling策略
│   │   ├── *_simt / *_simt_sort_tiling.cpp/.h            # SIMT Tiling策略
│   │   └── *_determinstic_tiling.cpp/.h                  # 确定性Tiling
│   └── op_api/                                           # ACLNN API
├── op_kernel/
│   ├── moe_inplace_index_add.cpp                         # Kernel入口
│   └── arch35/                                           # arch35专用Kernel
│       ├── *_simd.h / *_simt.h                           # SIMD/SIMT计算模板
│       ├── *_simd_sort.h / *_simt_sort.h                 # 排序变体
│       ├── *_determinstic.h / *_determinstic_notquant.h  # 确定性模板
│       └── indices_sort_utils.h                          # 索引排序工具
└── tests/ut/`
  },
  "3rd_moe_inplace_index_add_with_sorted_full_analysis.md": {
    tree: `3rd/moe_inplace_index_add_with_sorted/
├── CMakeLists.txt
├── op_host/
│   ├── moe_inplace_index_add_with_sorted_def.cpp         # 算子定义
│   ├── moe_inplace_index_add_with_sorted_tiling.h        # Tiling数据结构
│   ├── moe_inplace_index_add_with_sorted_tiling.cpp      # Tiling策略
│   └── op_api/                                           # ACLNN API
├── op_kernel/
│   ├── moe_inplace_index_add_with_sorted.cpp             # Kernel入口
│   ├── *_base.h                                          # 基础模板
│   ├── *_avg.h                                           # Avg模式模板
│   └── *_fix.h                                           # Fix模式模板
└── tests/ut/`
  },
  "3rd_moe_masked_scatter_full_analysis.md": {
    tree: `3rd/moe_masked_scatter/
├── CMakeLists.txt
├── op_host/
│   ├── moe_masked_scatter_def.cpp                        # 算子定义
│   ├── arch35/                                           # arch35专用Tiling
│   │   ├── moe_masked_scatter_tiling.h
│   │   └── moe_masked_scatter_tiling.cpp
│   └── op_api/                                           # ACLNN API
├── op_kernel/
│   ├── moe_masked_scatter.cpp                            # Kernel入口
│   └── arch35/ └── moe_masked_scatter.h                  # arch35 Kernel模板
└── tests/ut/`
  }
};

function processReport(filename, tree) {
  const filepath = path.join(DOCS_DIR, filename);
  if (!fs.existsSync(filepath)) {
    console.log(`  SKIP: ${filename} not found`);
    return false;
  }

  let content = fs.readFileSync(filepath, 'utf-8');

  if (content.includes('目录结构与文件职责')) {
    console.log(`  SKIP: ${filename} already has directory structure`);
    return false;
  }

  // Find all ## N. section headers
  const sectionRegex = /^## (\d+)\. /gm;
  const sections = [];
  let match;
  while ((match = sectionRegex.exec(content)) !== null) {
    sections.push({ num: parseInt(match[1]), index: match.index });
  }

  if (sections.length === 0) {
    console.log(`  SKIP: ${filename} - no numbered sections found`);
    return false;
  }

  const maxNum = Math.max(...sections.map(s => s.num));

  // Renumber from highest to lowest (avoid collisions)
  for (let n = maxNum; n >= 2; n--) {
    const regex = new RegExp(`^## ${n}\\. `, 'gm');
    content = content.replace(regex, `## ${n + 1}. `);
  }

  // Find insertion point (old ## 2. is now ## 3.)
  const insertMatch = content.match(/^## 3\. /m);
  if (!insertMatch) {
    // Try ## 2. in case there was only 1 section
    const match2 = content.match(/^## 2\. /m);
    if (!match2) {
      console.log(`  SKIP: ${filename} - can't find insertion point`);
      return false;
    }
  }

  const insertPos = content.indexOf(insertMatch ? '## 3. ' : '## 2. ');

  const newSection = `\n## 2. 目录结构与文件职责\n\n\`\`\`\n${tree}\n\`\`\`\n\n`;

  content = content.slice(0, insertPos) + newSection + content.slice(insertPos);

  fs.writeFileSync(filepath, content, 'utf-8');
  console.log(`  OK: ${filename} - added directory structure, renumbered sections`);
  return true;
}

// Main
let success = 0, skipped = 0, failed = 0;

for (const [filename, info] of Object.entries(REPORTS)) {
  console.log(`Processing: ${filename}`);
  try {
    if (processReport(filename, info.tree)) {
      success++;
    } else {
      skipped++;
    }
  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    failed++;
  }
}

console.log(`\n=== Summary ===`);
console.log(`Success: ${success}, Skipped: ${skipped}, Failed: ${failed}`);
