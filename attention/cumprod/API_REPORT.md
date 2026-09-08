# Cumprod API 探索报告（API_REPORT.md）

> Stage 2 输出。来源：pypto 源码仓 `code_repo/pypto`（docs/zh/pypto_pro、examples、tests）、pypto 0.2.1 安装包（cann-9.2.0）、ops-transformer 既有 pypto 算子（dense_lightning_indexer_softmax_lse_v2、sparse_flash_mla_softmax_l1_norm）。所有 API 均经源码/文档核实。

## 1. 可行性判定

**可行**。pypto-pro（tile 级 SIMD DSL）具备实现 cumprod 所需的全部原语；ops-transformer 构建体系已内置 pypto kernel 支持（`require_pypto_pro` / `enable_pypto_kernel` / `pypto_codegen.py`，cmake/func.cmake:639-720）。

## 2. API 映射表

| 需求 | pypto-pro API | 依据 |
|---|---|---|
| 裸指针 + 运行时 shape 视图 | `pl.Ptr[pl.DT_UINT8]` 形参 + `pl.make_tensor(ptr, shape, strides, dtype=input_dtype)` | dense kernel:877-1016 |
| dtype 特化（fp16/bf16/fp32 多二进制） | `@pl.jit(datatype={"x": "input_dtype"})`，模块级 `input_dtype` 编译期常量；二进制按 simplifiedKey `d=` 分发 | dense def + binary_info_config.json（d=0/1 两变体实测） |
| Tiling 数据 | `@dataclass class CumprodTiling`（int 字段），kernel 末位形参；host 侧 `GetTilingData<CumprodTiling>()`（force-include 生成头） | TilingData.md；dense tiling.cpp:414 |
| TilingKey | `class CumprodTilingKey: is_last_axis = TilingKeyField(bits=1, values=[0,1])`，`@pl.jit(tiling_key=...)`；host 侧 `context->SetTilingKey()` 按同一位布局手工打包 | tiling_key.md；dense tiling.cpp:437-440 |
| GM→UB 搬运 | `pl.load(tile, tensor, [i, j, k])` + `pl.set_validshape(tile, [1, cnt])`（尾块） | TilingData.md 完整示例 |
| UB→GM 搬运 | `pl.store(tensor, tile, [i, j, k])` | 同上 |
| dtype 转换（fp16/bf16↔fp32） | `pl.cast(out_tile, src_tile, mode=pl.RoundMode.CAST_RINT)`（tile 级，目标 dtype 由 out tile 决定） | language/_api.py:447 |
| 寄存器加载/存储 | `vf.load_align(tile, offset)` / `vf.store_align(tile, reg, preg)`（offset 单位为元素，支持运行时值） | dense kernel:127/188 |
| 向量乘 | `vf.mul(src0, src1, preg)`（支持 FP16/BF16/FP32/INT32/UINT32；MERGING 不支持 → 用 ZEROING+select 组合保位） | vf_computation/basic_arithmetic/mul.md |
| 标量广播 | `vf.full(scalar, dtype=pl.DT_FP32)`（scalar 可为运行时值） | dense kernel:97 |
| 按 lane 掩码 | `vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=...)`、`vf.update_mask(cnt, dtype=...)`（前 cnt 个 lane，cnt 可运行时） | dense kernel:96/120 |
| lane 选择 | `vf.select(src_true, src_false, preg)`（preg 内取 true，否则 false） | _vf_api.py:697 |
| VST→VLD 同 pipe 依赖 | `vf.mem_bar(mode=pl.MemBarMode.VST_VLD)`（同一 UB tile 先写后读必须） | test_fa_with_mask.py:299 |
| UB 标量读取 | `pl.getval(tile, idx)`（idx 可运行时，返回 fp32） | dense kernel:364 |
| 多核划分 | `pl.get_block_idx()` / `pl.get_block_num()` + `pl.range(core, total, cores)` | TilingData.md add_dynrank 示例 |
| 跨 pipe 同步 | `auto_mutex=True`（tile group mutex 自动管理 MTE2/V/MTE3） | dense kernel:857 |
| 标量 min/max | `pl.min(a, b)` / `pl.max(a, b)`（运行时标量） | dense kernel:590-591 |
| 二进制头生成 | `generate_binary_headers(kernel)`（由 `pypto_codegen.py` 在 CMake configure 期调用） | pypto_compile.py:224 |

## 3. 关键约束清单

1. **VL（寄存器长度）**：单寄存器 256B → fp32 64 lanes / fp16·bf16 128 lanes（reg_tensor.md）。扫描 chunk 宽度取 **VEC=64（fp32 lanes）**。
2. **vf.mul MERGING 不支持**：Hillis-Steele 步进用 `mul(ZEROING) + select` 组合保持未激活 lane 原值。
3. **同 UB tile 先写后读必须 `vf.mem_bar(VST_VLD)`**（V pipe 内乱序）。
4. **cross-width cast 只在 tile 级做**（`pl.cast`）；寄存器级 `vf.astype` 跨位宽需 layout/interleave 拼接，不采用。
5. **TilingKey 字段名不得与 kernel 形参/模块级变量冲突**；datatype 变量名同理（jit.py:1230-1268）。
6. **tiling dataclass 字段顺序即 C struct 布局**，host 侧 TILING_DATA_DEF 需与之一致（全 int64）。
7. **dtype 二进制变体**由 op_host/config/ascend950/`<op>_binary.json` 的 op_list 枚举（每个 dtype 组合一条 → 一个 fatbin）。
8. **simplified_key.ini** 需配置 `[Cumprod] default=0`（影响 opc --simplified_key_mode）。
9. 平台：a5（DAV_3510），vector_core_num=56；ops 编译目标 `Ascend950PR_9599`（本机 npu-smi 显示 Ascend950PR_9579，custom_cv/custom_nn 包同目标已在用，兼容）。
10. 环境：pypto_pro 0.2.1 仅存在于 cann-9.2.0 安装包 site-packages；构建需 `PYTHONPATH` 注入（已在 dense 构建冒烟中验证全链路可行）。

## 4. 参考实现信息

- 同仓参考：`attention/dense_lightning_indexer_softmax_lse_v2`（pypto-pro + tiling + TilingKey + enable_pypto_kernel 全套，已本地构建成功）。
- 算法参考：canndev tik `cum_computer.py`（逐位 vmul 迭代扫描 + 单位元初始化 + 内外维压平）；PyTorch CUDA `scan_dim`（末轴块内 Sklansky 树形扫描 + 跨块 carry，非末轴线程绑内行串行累乘）。
- pypto 内置 `pypto.cumprod`（frontend 级）支持 FP16/BF16/FP32、1-4 维，佐证数值口径可行性。
