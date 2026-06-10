# stem_oam_prep_varlen_q 算子迁移说明

## 算子定位

**stem_oam_prep_varlen_q** 是一个 Attention 预处理算子，用于 Stem OAM (Output-Aware Metric) block-sparse attention 机制。

## 最终放置位置

```
ops-transformer/attention/stem_oam_prep_varlen_q/
```

## 迁移理由

### 1. 仓库定位分析

**ops-transformer** (Transformer专用算子仓库):
- attention: FlashAttention, SparseAttention, MLA等
- ffn: Feed-Forward Network相关
- moe: Mixture of Experts
- posembedding: Position Embedding
- gmm: Gated Multimodal Modules

**ops-nn** (基础神经网络算子仓库):
- activation: 激活函数
- conv: 卷积
- matmul: 矩阵乘法
- norm: 归一化
- pooling: 池化
- index: 索引操作

### 2. 算子归属分析

**stem_oam_prep_varlen_q 算子特征**:
- 功能: 为 OAM block-sparse attention 提供 Q 侧预处理
- 应用: Transformer 模型推理阶段的 attention 计算
- 类别: Attention 预处理算子（不是基础索引算子）

**同类算子参考** (ops-transformer/attention/):
- `block_sparse_attention` - Block稀疏attention
- `mla_preprocess` - MLA预处理
- `kv_quant_sparse_flash_attention` - KV量化稀疏attention
- `sparse_flash_attention` - 稀疏flash attention

### 3. 决策依据

✅ **ops-transformer/attention/** 的合理性:
1. 算子本质是 attention 预处理，属于 Transformer 专用
2. 与 `block_sparse_attention`、`mla_preprocess` 等算子同类
3. 符合 ops-transformer 的定位（Transformer专用算子）
4. 不符合 ops-nn 的定位（基础神经网络算子）

❌ **ops-nn/index/** 的不合理性:
1. index 目录主要处理变长序列索引、查找操作
2. stem_oam_prep_varlen_q 虽然涉及 varlen，但核心是 attention 预处理
3. 功能定位不匹配

## 完成的工作

### 1. 算子迁移
- ✅ 从 `/tmp/operator_design/stem_oam_prep_varlen_q/` 克隆
- ✅ 移动到 `ops-transformer/attention/stem_oam_prep_varlen_q/`
- ✅ 移除临时放置位置 `ops-nn/index/stem_oam_prep_varlen_q/`

### 2. 构建系统适配
- ✅ 创建根目录 `CMakeLists.txt`
- ✅ 创建 `op_host/CMakeLists.txt`
- ✅ 配置编译选项和依赖

### 3. 目录结构验证
```
stem_oam_prep_varlen_q/
├── CMakeLists.txt         ✅ 已更新
├── docs/                  ✅ 设计文档
├── op_api/                ✅ ACLNN接口
├── op_extension/          ✅ 扩展接口
├── op_host/               ✅ Host代码
│   ├── CMakeLists.txt     ✅ 已创建
│   ├── arch35/            ✅ 架构特定实现
│   ├── *_def.cpp          ✅ 定义文件
│   └── *_infershape.cpp   ✅ Shape推导
├── op_kernel/             ✅ Kernel代码
│   ├── arch35/            ✅ 架构特定实现
│   └── *_arch35.cpp       ✅ Kernel实现
├── scripts/               ✅ 构建脚本
└── test_stem_scale.py     ✅ 测试文件
```

## 下一步建议

### 1. 构建验证
```bash
cd ops-transformer
mkdir build && cd build
cmake .. -DASCEND_OP_NAME=stem_oam_prep_varlen_q
make -j
```

### 2. 功能测试
```bash
cd ops-transformer/attention/stem_oam_prep_varlen_q
python test_stem_scale.py  # 或相应的测试脚本
```

### 3. 集成测试
- 验证与其他 attention 算子的兼容性
- 测试在 Stem OAM 完整流程中的表现
- 性能基准测试

## 算子核心特性回顾

### 计算流程
```
Input (FP8/BF16/FP16) 
  → Varlen Slice (变长切分)
  → Zero Padding (补零对齐)
  → Cast FP32 (精度提升)
  → Block Reshape (块重排)
  → Copy Scales (复制缩放因子)
  → Weighted Group Sum (加权和)
  → Flatten (展平)
  → Cast BF16 (输出精度)
  → Output (BF16)
```

### 并行模式
- **Mode 0**: batch×H_q ≥ AICore，(batch, head)粒度并行
- **Mode 1**: batch×H_q < AICore，(batch, head, qb_range)粒度并行
- **Mode 2**: block粒度并行，负载均衡最优

### 关键参数
- stemBlockSize: 128 (块大小)
- stemStride: 16 (步长)
- R=8, S=16, D=128 (压缩参数)
- 压缩比: 128 tokens → 2048 dimensions (1/8)

### 架构支持
- 仅支持 arch35 (DAV_3510)
- FP8/BF16/FP16 输入
- BF16 输出

## 相关文档

- 设计文档: `docs/StemOamPrepVarlenQ需求分析和方案设计说明书.md`
- API文档: `op_api/aclnn_stem_oam_prep_varlen_q.h`
- 测试代码: `test_stem_scale.py`