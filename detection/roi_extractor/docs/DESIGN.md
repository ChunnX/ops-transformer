## 算子名称

- **op_name**: roi_extractor
- **设计时间**: 2026-09-07
- **基于 SPEC**: `custom/roi_extractor/SPEC.md`
- **基于 EXPLORE_REPORT**: `custom/roi_extractor/EXPLORE_REPORT.md`
- **接入模式参照**: `attention/dense_lightning_indexer_softmax_lse_v2/`（op_kernel `@pl.jit` + Ptr/make_tensor + TilingData/TilingKey + op_host def/infershape/tiling + CMake）
- **目标平台**: Ascend950PR（编译目标 ascend950；JIT 验证 `TILE_FWK_DEVICE_ID=6`）

---

## Knowledge Bindings

> 完整、机器可读的 requirement 清单见 [`DESIGN_BINDINGS.json`](DESIGN_BINDINGS.json)。
> 本文只记录 R0–R8 的实际设计决策；每条活动 requirement 的 `planned_location` 必须准确指向
> 对应决策及最终 `test_roi_extractor.py` file/symbol，不复制 Binding 分组或 requirement 表格。

---

## §0 Module 划分（R0 输出）

> 依据 [Module划分](../../.opencode/skills/pypto-pro-op-design/references/module_partitioning.md)，先按数据依赖确定Module边界，再按Section边界继续拆分。

### 维度契约

- **kernel 接收与处理维度**：kernel 以 8 个裸指针 `pl.Ptr[pl.DT_UINT8]`（features_0..7）+ `rois` + `index` + `y` 接收原始输入，全部 shape/层数信息经 TilingData 传递（dense_lightning 同款 dynrank 形态）。kernel 内以 `pl.make_tensor` 重建类型化视图，**不做任何 host 侧 reshape/stride 归一化**：

| SPEC 输入 shape | kernel 接收 | kernel 内索引/stride 映射 | 输出形态 |
|-----------------|-------------|---------------------------|----------|
| features_i `[N,C,H_i,W_i]` 4D NCHW（i<L 有效；缺席层 host 填 H=W=1） | 原始裸指针 | `make_tensor(ptr_i, [N*C*H_i, W_i], [W_i, 1], dtype=input_dtype)`；行号 = `n*C*H_i + c*H_i + y`，strip 切片 load 按此寻址 | — |
| rois `[R,5]` | 原始裸指针 | `make_tensor(rois, [R, 5], [5, 1], dtype=input_dtype)`；逐行 load+cast fp32 后 `getval` | — |
| index `[R]` int32 | 原始裸指针 | `make_tensor(index, [R, 1], [1, 1], dtype=pl.DT_INT32)`；`getval(idx_view, n)`（仅 `HasIndex==1` 编译期分支内访问） | — |
| y `[R,C,pooled_height,pooled_width]` | 原始裸指针 | `make_tensor(y, [R*C, pH*pW], [pH*pW, 1], dtype=input_dtype)`；store 偏移 `[y_row*C + c0, px_off]`，内维 pH*pW 连续 | SPEC 要求形态（覆盖写，仅非 don't-care 行） |

- **kernel 内维度适配规则**：R/N/C/H_l/W_l/L/pooled/sample_num 全部为运行时 TilingData 字段；`pl.Ptr` 不做 rank/dtype 检查，wrapper 负责真实槽位校验（见 §5 TensorList ABI）。

### Wrapper 边界外操作

空

### 划分依据

数据依赖分析：SPEC 公式 (1)–(6) 为**单向串行链**——层级选择 → 坐标换算 → 采样数 → batch 守卫 → 逐样本双线性累加 → 平均写出。不存在需要"基于最终结果重新遍历完整数据范围"的第二遍扫描：每个 RoI 的全部状态（坐标、采样数、累加器）在寄存器/UB 内闭合，跨 RoI 无任何数据依赖（index 重复映射属 SPEC 声明的未定义输入，不设计仲裁）。每一步均为纯 Vector 语义（gather/逐元素乘加/平均），全部落在 Vec(UB) 数据通路，无 Cube/matmul，无跨执行域边界，因此**无 Section 边界可拆**。片上同时存活数据（strip 168KB + out_tile 16KB + 杂项，见 §3）不超过 UB 容量，无需按空间拆分。

**结论：单 Module**——数据依赖为单链、单一 Vector Section、片上空间足够，不需要拆分。

### Module 列表

| Module | 目的 | 输入依赖 | 输出 | Section |
|-------|------|---------|------|---------|
| Module 1 | 完整 RoiExtractor 语义：逐 (roi, 通道块) 的层级选择、坐标换算、strip 驻留 + 4 点 gather 双线性采样、fp32 累加平均、index 重映射覆盖写 y | features_0..7 / rois / index（primary） | y `[R,C,pH,pW]` | section_vector |

### 归约轴容量结论

- **归约轴是否可能超单 tile**：样本累加轴 sh*sw 运行时可变且无上界（sample_num ≤ 1024 为固定模式；自适应模式 `sw=ceil(bin_w)` 随 roi 尺寸放大，包络内可达数千）。该轴**不需要**跨 tile 归约状态：累加器是每 (roi, pixel) 一个 64-lane fp32 寄存器，样本按 (i,j) 顺序流入寄存器累加（`vf.add`，layernorm 样例 L89-95 同款跨 `pl.range` 寄存器累加器），与 tile 尺寸无关。
- **多Tile归约方案**：不涉及（寄存器累加 + 特征驻留 strip 按 (i,j) chunk 分批，无跨遍历统计量）。

### Module 级数据流

```
GM features_l ─[pl.load 切片]→ (fp16: staging ─[pl.cast]→) strip_f32(UB, 168KB)
                                        │
GM rois ─[pl.load 行]→ rois_raw ─[pl.cast]→ rois_f32 ─[getval]→ 标量坐标链(S/V)
                                        │
                    strip_f32 ─[vf.gather ×4/样本]→ RF 寄存器(64ch fp32)
                                        │              ─[vf.muls/vf.add fp32 累加]→ acc 寄存器
GM index ─[getval]→ y_row（HasIndex==1 编译期分支）
                                        │
                    acc ─[vf.muls 1/(sh*sw)]→ vf.scatter→ out_tile(UB, 16KB)
                                        │                    └(fp16: pl.cast RINT 原地)→
                    out_tile ─[pl.store]→ GM y[y_row, c0:c0+63, :, :]
```

---

## §1 API 映射（R1 输出）

> 按 `.opencode/references/performance-constraints.md`「强制 2：Vector 默认使用 VF」填写。每个 Vector 步骤冻结唯一实现。目标版本：pypto_pro 0.2.1 / Ascend950PR（EXPLORE_REPORT §7）。

### 数值安全边界总则（先于步骤表）

SPEC §7 cast 边界链冻结为：**坐标链/权重/累加全 fp32；层级链 fp32 sqrt → fp16(RINT) 比较；输出单次舍入**。实现载体：`pl.cast`（tile 级，无 lane 排布歧义，用于一切标量转换回环与 strip 装载转换）+ `vf.astype`（本设计不依赖其 layout 语义）+ `vf.sqrt(precision=True)`（寄存器级，见下表）。fp16 输入在**进入运算前**于 strip 装载处一次性展宽为 fp32（widen → compute → accumulate，precision.md KB「Widening a narrow-dtype reduction」顺序要求），热循环内不存在任何 fp16 数值运算，因此无窄 dtype 平方/长累加溢出问题；fp16 输出仅在 out_tile 处单次 RINT 舍入。

### Module 1 API 调用序列（含每步 vector_selection）

```
Module 1 — roi_extractor（单 section_vector）:

  ── 每工作项 (roi n, 通道块 cb) 序言（标量链 + 小 tile 转换回环）──
  S1  rois 行读入:  set_validshape(rois_raw[1,16], [1,5]) → pl.load(rois_raw, rois_view, [n,0])
      → pl.cast(rois_f32, rois_raw, CAST_NONE)（fp16→fp32 扩展；fp32 模式同路径直拷语义）
      → getval(rois_f32, 0..4) 得 x0r,y0r,x1r,y1r,bidx_r（fp32 标量）
  S2  batch 守卫:   coord_f32a[0,0]=bidx_r → pl.cast(coord_i32, coord_f32a, CAST_FLOOR)
      → b=getval → if b<0 or b>=N: continue（don't-care 行不写）
  S3  层级选择:     prod=(x1r-x0r)*(y1r-y0r)（fp32 标量算术；x0r..y1r 依次为 rois[n,1..4]）
      → if prod<0: continue（负积显式丢弃——vf.sqrt 要求 src 非负，前置判断替代 NaN 路径，
         与 SPEC §9 反转坐标 don't-care 语义一致）
      → coord_f32a[0,0]=prod → VF: load_align → vf.sqrt(reg,preg,precision=True) → store_align
         → getval 得 s_f32（IEEE 语义目标，边界用例取整数面积两类模式均精确）
      → pl.cast(coord_f16, coord_f32a, CAST_RINT)（fp32→fp16 round-to-nearest-even）
      → pl.cast(coord_f32b, coord_f16, CAST_NONE)（fp16→fp32 回读，值精确保持）
      → s16=getval(coord_f32b,0)，与 tiling.b_pow2[1..L-1]（host 预计算、fp16 可精确表示的
         fp32 值）做纯 fp32 标量比较：s16<=B[1]→l=0；L==1 且 s16>B[L-1]→丢弃；
         L>=2 且 s16>B[L-1]→l=L-1；否则扫描 l=1..L-2 找 B[l]<s16<=B[l+1]，无匹配→丢弃
  S4  坐标换算:     标量 fp32：sc=tiling.spatial_scale[l]；x0..y1 = rois 标量×sc；
         tiling.aligned==1 时四坐标各减 0.5；roi_w=max(x1-x0,1.0)；roi_h 同；
         pooled_w_f/pooled_h_f 由 tiling int 经 [1,8] i32→f32 cast 回环得 float 标量；
         bin_w=roi_w/pooled_w_f；bin_h=roi_h/pooled_h_f（fp32 除法）
  S5  采样数:       coord_f32a ← bin_w → pl.cast(coord_i32, coord_f32a, CAST_CEIL) → sw；
         bin_h 同理得 sh；tiling.sample_num>0 时 sw=sh=sample_num
         sw_f/sh_f 由 i32→f32 cast 回环得（供 inv 计算）；inv_sw=1.0/sw_f 等纯 fp32 标量除法
  S6  y_row:        HasIndex==1（编译期折叠）：y_row=getval(idx_view, n)；否则 y_row=n
  S7  层视图选择:   8 个层视图已在 kernel 顶部无条件构造；if l==0..7 链选 lvl_view
  S8  采样分数表:   每 ROI 一次 VF：vf.arange(0,DT_INT32)→vf.astype(DT_FP32)→vf.adds(0.5)
         →vf.muls(inv_sw)→vf.store_align(frac_j)；frac_i 同款（(i+0.5)*inv_sh）
         （astype INT32→FP32 为 astype.md 表4 原生路径；表长 128；sw>128 的自适应大 roi
           回退为逐样本 [1,8] cast 回环，见 S10 注）

  ── 像素循环（px 组 ≤ 64 对齐 out_tile）──
  S9  像素基量:     每 ph：ph_f（i32→f32 回环一次）；每 pw：pw_f 同；
         xbase = x0 + pw_f*bin_w；ybase = y0 + ph_f*bin_h（fp32 标量）
  S10 逐样本坐标:   for i0(i-chunk, SI 自适应) / j0(j-chunk, SJ 自适应):
           每 chunk 用首/末样本的 clamp 后坐标定窗口 [yc0,yc0+ys)×[xc0,xc0+xs)，
           ys≤STRIP_YS、xs≤STRIP_XS；窗口装载坐标 clamp 进 [0,H-1]×[0,W-1] 且尺寸≥1
         for i in chunk_i, j in chunk_j:
           x = xbase + getval(frac_j,j)*bin_w；yc = ybase + getval(frac_i,i)*bin_h
           （sw>128 时 frac 表不覆盖，改用 (j 标量 i32→f32 回环) 直接计算——同语义）
           越界判据 x<-1 或 x>W 或 yc<-1 或 yc>H → continue（贡献 0，分母不变）
           xc=min(max(x,0),W-1)；yc_c 同；coord_f32a[0,0]=xc,[0,1]=yc_c
           → pl.cast(coord_f32b, coord_f32a, CAST_FLOOR)（fp32→fp32 同型取整）
           → lx=xc-getval(coord_f32b,0)；ly 同
           → pl.cast(coord_i32, coord_f32b, CAST_TRUNC)（已整值，TRUNC 无损）
           → x_lo,y_lo；x_hi=min(x_lo+1,W-1)；y_hi 同
  S11 strip 装载:  for c in 0..c_eff-1:
            slice = strip[c*STRIP_YS:(c+1)*STRIP_YS, 0:STRIP_XS]（运行时切片，dense_lightning
              reduce_sum_vec_tile[0:1, ci*TKV_HALF:] 同款运行期下标先例）
            set_validshape(slice,[ys,xs])（先设后用：fp32 定义 load 装载窗，fp16 定义
              cast 转换区域——尾块铁则）
            fp32 模式: pl.load(slice, lvl_view, [b*C*H+c*H+yc0, xc0])
            fp16 模式: set_validshape(staging,[ys,xs]) → pl.load(staging, lvl_view, 同偏移)
                       （staging 同样先设后 load——tail_block_handling L74-80「set_validshape
                         应在 load 之前调用，用于约束 GM 搬入」；物理 [28,32]，有效窗与
                         slice 同 [ys,xs]，保证后续 cast 源区域覆盖一致）
                       → pl.cast(slice, staging, CAST_NONE)（fp16→fp32 于运算前展宽；源/目的
                         物理 shape 与行跨度不同合法——cast.md，转换区域 [ys,xs] 内元素按
                         逻辑坐标逐元素对应）
          装载完成后 set_validshape(strip 全 tile, 物理满幅) 再进入 gather（消除
            slice 窗口与 gather 有效范围语义的歧义，见 §7 验证点 VP3）
  S12 4 点 gather:  VF 调用（每样本）：lane_base = vf.arange(0,DT_UINT32) → vf.muls(STRIP_CH)
           → pos00 = (y_lo-yc0)*STRIP_XS + (x_lo-xc0)；pos01/pos10/pos11 以 x_hi/y_hi
             替换对应分量（x_hi>x_lo 时 +1 列；y_hi>y_lo 时 +STRIP_XS 行；相等时同址退化）
           → off_k = vf.adds(lane_base, pos_k) → vf.mins(off_k, STRIP_ELEMS-1)
             （越界保险，活动 lane 恒不触发）
           → v_k = vf.gather(strip, off_k, preg_ch)（fp32 源 + DT_UINT32 索引，
             gather.md dtype 表原生组合；preg_ch=update_mask(c_eff) 屏蔽尾通道）
  S13 双线性累加:   acc = vf.add(acc, vf.muls(v00, whh, preg_ch), preg_ch) ×4 角
           （权重 whh/whl/wlh/wll 为该样本的 fp32 标量，muls 标量广播；全 fp32 寄存器链）
  S14 平均+散写:    out = vf.muls(acc, inv_area, preg_ch)
           → idx = vf.arange(0,DT_UINT32)→vf.muls(PX_PHYS)→vf.adds(px-px0)
           → vf.scatter(out_tile, out, idx, preg_ch)（fp32 源 + u32 索引，逐 lane 唯一）
  S15 组尾写出:     set_validshape(out_tile,[c_eff,n_px])
           fp16 模式: set_validshape(out_f16 视图,[c_eff,n_px])
                      → pl.cast(out_f16, out_tile, CAST_RINT)（同地址原地 fp32→fp16，
                        cast.md 明示支持；两 tile 组共用 mutex_id=1）
            → pl.store(y_view, out_tile(f32)/out_f16(f16), [y_row*C+c0, px0])
```

> **Stage 5 as-built 修订（§1 S8 stale 行，2026-09-08；stage5-check 复核意见落实）**：
> 上述伪代码中 **S8 采样分数表步骤已随 Stage 5 Exp-B 删除**（PERFORMANCE_REPORT
> §6.1：`(j+0.5)*inv_sw`/`(i+0.5)*inv_sh` 纯标量算术替代，位级同构；frac_j/frac_i
> tile 组与 `if sw<=FRAC_N` 分支一并退役）。S10 行内 `getval(frac_j,j)` 引用同样
> 为 Stage 4 历史形态——as-built 采样分数直接由标量算术产生；原「sw>128 回退
> 标量回环」的覆盖上限不再存在（分数表删除后无 128 上限，宽窗几何走 §4 as-built
> 的快/回退双路径）。上文保留 Stage 4 原文作历史对照。

### vector_selection 汇总表

| 步骤 | implementation | api_sequence | decision_reason | kb_template_evidence | applicable_conditions |
|------|---------------|--------------|-----------------|---------------------|------------------------|
| S1/S2/S4/S5/S9/S10 标量链与取整回环 | tile_op（转换/搬运，非数值计算） | `pl.load`/`pl.cast`(FLOOR/CEIL/TRUNC/RINT/NONE)/`pl.getval`/下标写 | 数据搬运与类型转换不受强制 2 约束；标量域为 kernel 语言本体 | n/a（cast.md/cast 转换表、getval.md 为 API 依据非 KB 模板） | 全 dtype/shape |
| S3 sqrt | **vf** | `vf.sqrt(reg, preg, precision=True)` | default_vf（无 KB 模板要求 pl.sqrt） | n/a | src 非负（S3 前置 prod≥0 守卫保证）；fp32 |
| S3/S10 层级与坐标比较 | 标量域 | kernel 标量表达式 | 控制流/标量比较，非 Vector 数值步骤 | n/a | fp32 同型比较 |
| S8 分数表构造 | **vf** | `vf.arange→vf.astype→vf.adds→vf.muls→vf.store_align` | default_vf | n/a | sw/sh ≤ 128；超界回退标量回环 |
| S11 strip 装载 | tile_op（搬运+转换） | `pl.load`（切片/staging）/`pl.cast`(NONE) | 数据搬运；fp16→fp32 转换非数值计算 | n/a | load 目的 tile 内维 32B 对齐（fp32: strip 切片 24×4B=96B；fp16: staging 32×2B=64B） |
| S12 gather | **vf** | `vf.gather`(fp32+u32) | default_vf；KB vec-ub-strip-gather 核心路线 | n/a | 索引在 strip 物理范围（mins 保险） |
| S13 累加 | **vf** | `vf.muls`/`vf.add` | default_vf | n/a | fp32 全链 |
| S14 平均+散写 | **vf** | `vf.muls`/`vf.scatter`(fp32+u32) | default_vf；KB vec-scatter-owner-model §7（用 vf.scatter 不用 pl.scatter） | n/a | index 逐 lane 唯一 |
| S15 写回 | tile_op（转换/搬运） | `pl.cast`(RINT)/`pl.store` | 输出转换与搬运 | n/a | 内维 pH*pW 段连续；valid 收窄 |

> **Stage 5 as-built 修订（vector_selection 表 S8 行，2026-09-08；stage5-check
> 复核意见落实）**：表中「S8 分数表构造」行已随 Exp-B 删除退役——as-built **无
> 独立 VF 构表步骤**，采样分数 `(j+0.5)*inv_sw`/`(i+0.5)*inv_sh` 属标量域算术
> （S1/S2/S4/S5/S9/S10 标量链域，行内 int→float 直接提升）。该行保留为
> Stage 4 历史形态；as-built 在用 VF 步骤 = S3 sqrt / S12 gather / S13 累加 /
> S14 平均+散写（S11/S15 仍为 tile_op）。

### 超越函数数值安全边界

| API | 输入理论范围 | 目标 dtype 上限 | 是否溢出 | 防护措施 |
|-----|-------------|----------------|----------|----------|
| `vf.sqrt(precision=True)` | prod=(x1-x0)*(y1-y0)，SPEC 值域下 |max| ≤ 65504² ≈ 4.3e9 < fp32 max 3.4e38 | fp32 无溢出；**负数输入未定义** | **前置 `prod<0 → continue` 显式丢弃**（don't-care 行，SPEC §9 反转坐标语义）；精度取 precision=True（近似 IEEE；§10 recipe 边界行取整数面积，任何正确 sqrt 实现均精确，规避 fp16 舍入歧义） |
| `1.0/sw_f`（除法） | sw_f ≥ 1.0 | fp32 | 无（sw≥1 由 ceil/max(1) 保证） | ceil 结果 ≥1；SPEC §9 "sample_num=0 且 bin<1 → ceil 后 1" |
| `bin_w=roi_w/pooled_w_f` | roi_w ≥ 1.0，pooled_w_f ≥ 1.0 | fp32 | 无 | max(Δ,1) 保护 + pooled ∈ [1,1024] host 校验 |
| 权重 lx/ly | x_c−floor(x_c) ∈ [0,1] | fp32 | 无 | clamp 后 floor，差恒 ∈ [0,1] |

无 exp/log/tanh。fp16 输出单次 RINT 舍入，无中间 fp16 舍入点。

---

## §2 Tile 规划（R2 输出）

### 2.1 关键常量定义

```python
# ── 算子关键常量（coder 必须逐字使用） ──
CH_BLOCK  = 64    # 每工作项通道数 = fp32 VF 寄存器 lane 宽（两 dtype 模式统一，无 dtype 分支常量）
STRIP_YS  = 28    # strip 每通道最大行数（物理行距）          [Stage 4 形态，已被 Stage 5 取代]
STRIP_XS  = 24    # strip 最大列数（物理列距；24*4B=96B）      [Stage 4 形态，已被 Stage 5 取代]
STRIP_CH  = STRIP_YS * STRIP_XS        # 672，每通道 strip 元素数 [Stage 4 形态]
STRIP_ELEMS = CH_BLOCK * STRIP_CH      # 43008，strip 总元素（gather 索引上限充裕；Stage 5 沿用）
PX_PHYS   = 64    # out_tile 像素列物理宽度（pooled_h*pooled_w 的分组粒度）
FRAC_N    = 128   # 采样分数表长度（sw/sh 超界回退标量路径）  [Stage 4 形态，Stage 5 已删除分数表]
INV_SAFE  = 1     # gather 索引 mins 钳制上限偏移（STRIP_ELEMS-1）
```

> **Stage 5 as-built 修订（2026-09-08，最终实现 = PERFORMANCE_REPORT.md §8 Cand-2）**：
> 分数表删除（Exp-B）与 strip 宽视图化（Exp-C）后，最终常量为：
> ```python
> CH_BLOCK   = 64    # 不变
> STRIP_COLS = 48    # strip 物理列距（48*4B=192B，32B 倍数；快路径 x 窗口 xs_roi≤48）
> STRIP_ROWS = 896   # strip 物理行数（64ch×14 行 @ c_eff=64；与 Stage 4 [1792,24] 同址同总量 43008）
> STAGE_ROWS = 56    # staging 物理行数（fp16 装载中转；≥ 任意 ys_cap）
> STAGE_COLS = 48    # staging 物理列距（48*2B=96B / fp32 视图 192B，均 32B 对齐）
> PX_PHYS    = 64    # 不变
> XS_BUDGET  = 46    # 回退路径每像素 x-chunk 列预算（+2 边沿 ≤ 48）
> JB         = 4     # 快路径每 VF 累加调用合批样本数（sw≥4 时生效）
> FRAC_N     = 128   # 仅作历史对齐注释保留，代码中无分数表
> ```
> strip 每通道行数由编译期常量（28）改为**运行时 pitch**（`pitch = ys_c × 48`，
> `ys_c = chunk 行窗高`，容量约束 `c_eff × ys_c ≤ 896`，即 `ys_cap = min(56, 896//c_eff)`）。

> **注意**：strip 恒为 **fp32**（两 dtype 模式统一，fp16 特征在装载处 `pl.cast` 展宽），使 gather 恒为 `DT_FP32 源 + DT_UINT32 索引` 原生组合，彻底回避 fp16 源的 DT_UINT16 索引 ≤65535 窗口约束与 `vf.astype` layout 语义歧义（EXPLORE_REPORT §6.1 裁定项的落点）。通道块恒为 64：fp16 模式下无需 128-lane 分裂（KB vec-ub-strip-gather「b16 index window」与 precision.md「astype layout」两条警示的规避）。

### Tile 属性表

| 用途 | 变量名 | shape | dtype | 内存空间 | layout | 大小(B) | 备注 |
|------|--------|-------|-------|---------|--------|---------|------|
| 特征驻留 strip | `strip` | `[1792, 24]`（=64ch×28 行 × 24 列） | FP32（恒定） | UB(Vec) | `—` | 172032 | TileGroup 单槽；每通道子块 = 行 `[c*28,(c+1)*28)`；valid 逐切片设置，gather 前恢复物理满幅 |
| 输出暂存/累加落盘 | `out_tile` | `[64, 64]`（ch × px） | FP32 | UB(Vec) | `—` | 16384 | 行=通道（GM 同构），列=像素；逐像素 vf.scatter 写列 |
| 输出 fp16 视图 | `out_f16` | `[64, 64]` | FP16 | UB(Vec) | `—` | （同地址 16384 区域的前 8192B 逻辑视图） | 与 out_tile **同地址**（fp32→fp16 原地 cast，cast.md 明示支持）；共用 mutex_id=1 |
| fp16 装载中转 | `staging` | `[28, 32]` | input_dtype | UB(Vec) | `—` | 3584(fp32 视图上界) | fp16 模式：load→cast 进 strip 切片；fp32 模式不使用（直接 load 进 strip）。物理宽取 32：fp16 内维 32×2B=64B、fp32 视图 32×4B=128B 均 32B 对齐（原 [28,24] 在 fp16 下内维 24×2B=48B 不是 32 的倍数，违反 load.md「Tile shape 需要 32 字节对齐」，已改形）；有效窗 [ys,xs]≤[28,24] 与 strip 切片按逻辑坐标逐元素对应（cast.md：源、目的 Tile 的物理 shape 和行跨度可以不同） |
| rois 行原始 | `rois_raw` | `[1, 16]` | input_dtype | UB(Vec) | `—` | 64(fp32 视图上界) | 内维 16×2B=32B 满足对齐（fp16 下 [1,8] 仅 16B 不合法，故取 16） |
| rois 行 fp32 | `rois_f32` | `[1, 8]` | FP32 | UB(Vec) | `—` | 32 | cast 目标；getval 源 |
| 坐标回环 A | `coord_f32a` | `[1, 8]` | FP32 | UB(Vec) | `—` | 32 | setval 写入待转换标量 |
| 坐标回环 B | `coord_f32b` | `[1, 8]` | FP32 | UB(Vec) | `—` | 32 | 同型取整（FLOOR/CEIL）结果 |
| 坐标回环 I | `coord_i32` | `[1, 8]` | INT32 | UB(Vec) | `—` | 32 | 整型结果（TRUNC） |
| 层级链 fp16 中转 | `coord_f16` | `[1, 16]` | FP16 | UB(Vec) | `—` | 32 | s 的 fp32→fp16(RINT)→fp32 回读链 |
| 采样分数表 j | `frac_j` | `[1, 128]` | FP32 | UB(Vec) | `—` | 512 | (j+0.5)*inv_sw，VF 一次构建 |
| 采样分数表 i | `frac_i` | `[1, 128]` | FP32 | UB(Vec) | `—` | 512 | (i+0.5)*inv_sh |

> 所有 tile 内维字节数均为 32 的倍数（24×4=96、64×4=256、64×2=128、**32×2=64（staging fp16 内维；其 fp32 视图 32×4=128）**、16×2=32、8×4=32、128×4=512），满足 `load.md`/`store.md`「Tile shape 需要 32 字节对齐，不对齐报错；valid_shape 可以不对齐」约束（staging 物理宽由 24 改为 32 即本轮修订项：原 24×2B=48B 不对齐）。layout 列 `—`：纯 Vec ND 路径无布局要求（TileType.md 默认表）。

> **Stage 5 as-built 修订（tile 属性表差异，最终实现 = Cand-2）**：
>
> | 用途 | 变量名 | shape（as-built） | dtype | 变更说明 |
> |---|---|---|---|---|
> | 特征驻留 strip | `strip` | **[896, 48]** | FP32（恒定） | 与 Stage 4 [1792,24] **同址同总量**（0x00000，172032B，43008 元素）；每通道子块行域变为运行时 `[c*ys_c,(c+1)*ys_c)`，列域 `[0,xs_roi)`；gather lane 间距 = 运行时 `pitch=ys_c*48`（原编译期 672）；快路径整窗装载、回退路径逐像素 chunk 装载均走此视图 |
> | fp16 装载中转 | `staging` | **[56, 48]** | input_dtype | 原 [28,32]（3584B）→ [56,48]（fp32 视图上界 10752B，@0x2F000）；行数 ≥ 任意 `ys_cap=min(56,896//c_eff)`，快路径行组窗与回退 chunk 窗共用；48×2B=96B / 48×4B=192B 均 32B 对齐 |
> | 采样分数表 j/i | `frac_j`/`frac_i` | **已删除** | — | Exp-B：`(j+0.5)*inv_sw` 纯算术位级同构替代（PERFORMANCE_REPORT §6.1）；VA_FRAC_J/VA_FRAC_I（各 512B）与 mutex 9/10 释放 |
> | 坐标回环 B | `coord_f32b` | **已删除** | — | FLOOR+TRUNC 双跳 → 直接 FP32→INT32 `CAST_FLOOR` 单跳（cast.md L77；Stage 4 S2/S5 已在役证明该直转）；VA_COORD_B（32B）与 mutex 6 释放 |
> | 坐标回环 A/I | `coord_f32a`/`coord_i32` | [1,8]（不变） | FP32/INT32 | 角色扩展：快路径 x 侧 4 样本合批 FLOOR（一次 TCVT 转换 4 值）+ 行组/窗口边沿 2 值批 + y 侧每 (ph,i) 单值；S3 层级链 fp32 落点复用 rois_f32（coord_b 退役后） |
> | 其余 tile | — | 不变 | — | out/out_f16（同址 mutex 1）、rois_raw/rois_f32、coord_f16、acc 均维持 Stage 4 合同 |
>
> 对齐核验（as-built 全表）：48×4=192、64×4=256、64×2=128、48×2=96、16×2=32、8×4=32 —— 全部 32B 倍数。

### TileGroup槽位访问表

| TileGroup | depth | 每个Tile的mutex ID | 访问方式 | 槽位表达式 | 同步及有界/游标说明 |
|---|---:|---|---|---|---|
| `strip_g` | 1 | `[0]` | `current()` + 运行时切片 | 切片行 `[c*28:(c+1)*28)`，c∈[0,c_eff)⊂[0,64) | 单槽；MTE2(切片 load)→V(gather) 由 auto_mutex 经 id=0 管理；同地址连续 load 前有 cast/gather 读取，无未读 WAW |
| `out_g`(fp32) | 1 | `[1]` | `current()` | — | 单槽；V(scatter)→MTE3(store) 经 id=1 |
| `out_g16`(fp16) | 1 | `[1]`（与 fp32 组共用） | `current()` | — | 与 out_tile 同地址复用同一 mutex_id（api_mapping「复用同一片上地址的 Tile 使用相同 mutex_id」）；cast(V) 涉及两组、auto_mutex 带同 ID 同步 |
| `stage_g` | 1 | `[2]` | `current()` | — | 单槽；load(MTE2)→cast(V) 链式，每通道读后复写 |
| `roisraw_g`/`roisf32_g` | 1 | `[3]` / `[4]` | `current()` | — | 单槽小 tile |
| `coorda_g`/`coordb_g`/`coordi_g`/`coordf16_g` | 1 | `[5]`/`[6]`/`[7]`/`[8]` | `current()` | — | 标量回环；S/V 混合 pipe 由 auto_mutex 按组同步 |
| `fracj_g`/`fraci_g` | 1 | `[9]`/`[10]` | `current()` | — | VF 写(V)→getval 读(S) |

> mutex ID 共 11 个 ∈ [0,31]（tiling.md KB 硬上限 32）。全部 tile 声明于 kernel 顶部**一次**（KB vec-tensorlist-fixed-arity「declare tile groups ONCE」）；无 `group[i]` 运行时下标（切片下标为 tile 视图语法，非组槽位）。
>
> **Stage 5 as-built 修订**：fracj_g/fraci_g（mutex 9/10）与 coordb_g（mutex 6）随分数表删除/直转替换而退役；as-built 在用 mutex ID = **9 个**（0 strip / 1 out+out16 / 2 stage / 3 roisraw / 4 roisf32 / 5 coorda / 7 coordi / 8 coordf16 / 11 acc），仍在 [0,31) 硬上限内；strip_g 槽位说明同步改为「切片行 `[c*ys_c,(c+1)*ys_c)` 运行时行域」。

### tile_dims stride 注意事项

不涉及（未使用 `load_tile`/`store_tile` 的 tile_dims 形态；全部 `pl.load`/`pl.store` 绝对元素偏移直取）。

---

## §3 片上空间布局（R3 输出）

> 纯 Vector 算子仅涉及 UB(Vec) 一个空间；Mat/L1、L0A/L0B/L0C 均不使用。地址从 0x00000 起算，逐 tile 32B 对齐连续排列。

### 片上地址映射表（按内存空间分节）

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 起始字节 | 每槽字节数 | 槽位数 | 生命周期 | 结束字节（不含） | 备注 |
|---------|------|--------|-------|-------|--------|---------|-----------:|------:|----------|------------------|------|
| UB(Vec) | 特征驻留 strip | `strip` | `[1792,24]` | FP32 | `—` | 0x00000 | 172032 | 1 | 驻留（每 chunk 重载有效区） | 0x2A000 | 64 通道×28×24 |
| UB(Vec) | 输出暂存 | `out_tile` | `[64,64]` | FP32 | `—` | 0x2A000 | 16384 | 1 | 驻留（每 px 组重写） | 0x2E000 | 列=像素 |
| UB(Vec) | 输出 fp16 视图（同地址） | `out_f16` | `[64,64]` | FP16 | `—` | 0x2A000 | 逻辑 8192 | 1 | 与 out_tile 互斥复用（同地址顺序复写） | 0x2E000（区域共享） | 原地 cast；线性区段按元素位对齐 |
| UB(Vec) | fp16 装载中转 | `staging` | `[28,32]` | input_dtype | `—` | 0x2E000 | 3584 | 1 | 临时（单通道 load→cast 间） | 0x2EE00 | fp32 上界计容（fp16 实占 1792）；物理宽 32 保 fp16 内维 64B 对齐 |
| UB(Vec) | rois 行原始 | `rois_raw` | `[1,16]` | input_dtype | `—` | 0x2EE00 | 64 | 1 | 临时 | 0x2EE40 | |
| UB(Vec) | rois 行 fp32 | `rois_f32` | `[1,8]` | FP32 | `—` | 0x2EE40 | 32 | 1 | 临时 | 0x2EE60 | |
| UB(Vec) | 坐标回环 A | `coord_f32a` | `[1,8]` | FP32 | `—` | 0x2EE60 | 32 | 1 | 临时（单样本） | 0x2EE80 | |
| UB(Vec) | 坐标回环 B | `coord_f32b` | `[1,8]` | FP32 | `—` | 0x2EE80 | 32 | 1 | 临时 | 0x2EEA0 | |
| UB(Vec) | 坐标回环 I | `coord_i32` | `[1,8]` | INT32 | `—` | 0x2EEA0 | 32 | 1 | 临时 | 0x2EEC0 | |
| UB(Vec) | 层级链 fp16 中转 | `coord_f16` | `[1,16]` | FP16 | `—` | 0x2EEC0 | 32 | 1 | 临时 | 0x2EEE0 | |
| UB(Vec) | 分数表 j | `frac_j` | `[1,128]` | FP32 | `—` | 0x2EEE0 | 512 | 1 | 驻留（每 ROI 重写） | 0x2F0E0 | |
| UB(Vec) | 分数表 i | `frac_i` | `[1,128]` | FP32 | `—` | 0x2F0E0 | 512 | 1 | 驻留（每 ROI 重写） | 0x2F2E0 | |

> 半开区间 `[起始, 结束)`；唯一同地址对为 `out_tile`/`out_f16`（fp32→fp16 原地 cast 的文档化场景），生命周期严格串行（scatter 完成→cast→store），共用 mutex_id=1 由 auto_mutex 顺序化，无并发重叠。其余 tile 两两不相交。`out_f16` 行的"每槽字节数"按 fp32 物理区域 16384B 计容（cast 不缩物理区段，fp16 逻辑视图占用其前 8192B，区段分配仍按 16384B 保留）。

**各空间地址高水位**:
- UB(Vec): 0x2F2E0 = 193248 / 253952 = **76.1%**（余量 60704B ≈ 23.9%）
- L1(Mat) / L0A / L0B / L0C：不使用（纯 Vector 算子）

> **Stage 5 as-built 修订（地址映射，最终实现 = Cand-2）**：
>
> | UB(Vec) 用途 | 变量名 | shape | dtype | 起始字节 | 结束字节（不含） | 说明 |
> |---|---|---|---|---|---|---|
> | 特征驻留 strip | `strip` | [896,48] | FP32 | 0x00000 | 0x2A000 | 同址改形（172032B 不变） |
> | 输出暂存 + fp16 视图 | `out_tile`/`out_f16` | [64,64] | FP32/FP16 | 0x2A000 | 0x2E000 | 不变（mutex 1 共用） |
> | rois 行原始 | `rois_raw` | [1,16] | input_dtype | 0x2EE00 | 0x2EE40 | 不变 |
> | rois 行 fp32 | `rois_f32` | [1,8] | FP32 | 0x2EE40 | 0x2EE60 | 不变（S3 层级链落点复用） |
> | 坐标回环 A | `coord_a` | [1,8] | FP32 | 0x2EE60 | 0x2EE80 | 不变（4 值合批写） |
> | 坐标回环 I | `coord_i` | [1,8] | INT32 | 0x2EE80 | 0x2EEA0 | 不变（直转 CAST_FLOOR 目标） |
> | 层级链 fp16 中转 | `coord_f16` | [1,16] | FP16 | 0x2EEA0 | 0x2EEC0 | 不变 |
> | 像素累加器 | `acc` | [1,64] | FP32 | 0x2EEC0 | 0x2EFC0 | Stage 4 已引入（原 DESIGN §3 表末行之后的运行时扩展，as-built 固化于此） |
> | fp16 装载中转 | `staging` | [56,48] | input_dtype | 0x2F000 | 0x31A00（fp32 键最坏） | 改形扩容（fp32 视图 10752B；fp16 实占 5376B 至 0x30500） |
> | ~~坐标回环 B~~ | ~~coord_b~~ | — | — | ~~0x2EE80~~ | — | 已删除（与 coord_i 区间复用后释放） |
> | ~~分数表 j/i~~ | ~~frac_j/frac_i~~ | — | — | ~~0x2EEE0/0x2F0E0~~ | — | 已删除（各 512B 释放） |
>
> **as-built 高水位**：0x31A00 = 203264 / 253952 = **80.0%**（fp32 datatype key 的 staging 最坏情形；fp16 key 为 0x30500 = 197888 / 78.0%）。容量键 `ub_size=253952` 不变；同地址对仍唯一（out_tile/out_f16，mutex 1）。

> 容量来源：EXPLORE_REPORT §7 + arch-a5.md KB——`950PR_957x.ini` 键 `ub_size=253952`（248KB）；tile 首地址/shape 32B 对齐（`ubblock_size` 32B 量子）。本轮修订（staging 2688→3584B）已在本机 `/usr/local/Ascend/cann/python/site-packages/pypto/lib/configs/950PR_957x.ini` 复核该键仍为 253952；sibling DSL 设备表的 216KB（221184B）口径已被 arch-a5.md 裁定不描述本 SKU 工具链编译预算（「The platform file decides it」），预算余量以 253952 为准（若按 216KB 口径余量为 ~12.6%，不采用）。

### 分配方式选择

- **方案**: `make_tile_group` + `auto_mutex`（全部 12 个组均带 mutex_id；无任何 `make_tile` scratch——坐标回环等小 tile 虽为"单次使用"形态，但参与 MTE2/V/S 多 pipe 依赖且跨循环迭代复用地址，必须由组元数据驱动同步；scatter-owner-model KB 失败签名表「make_tile 无 MTE↔V 同步」的教训）。
- **依据算子**: `pro_ops/vf_api/test_layernorm_tile_group_vf.py`（纯 Vec 多核 + validshape 尾块 + VF 组合范式）、`pro_ops/element_wise/test_add.py`（tile_group/auto_mutex 最小骨架）。
- **理由**: 单 Module 单执行域；strip 的 MTE2(切片 load)→V(gather)、out_tile 的 V(scatter/cast)→MTE3(store)、回环 tile 的 S(setval/getval)↔V(cast) 全部为**核内跨 pipe**依赖，由各组 mutex_id + `@pl.jit(auto_mutex=True)` 自动管理；无跨核依赖（§6）。

### double buffer 地址规划

| 项目 | 结论 | 受影响 tile/地址 |
|------|------|-----------------|
| buffer 数 | 全部 1（单槽） | 全部 tile |
| PONG 地址 | 无 | — |
| 槽位访问 | `current()` | — |

> 不采用 strip 双缓冲：fp32 strip 单槽 168KB，双缓冲 336KB 超 UB 容量；MTE2 与 V 的重叠让位于容量约束（perf_target=null，正确性优先；Stage 5 可缩 strip 换双缓冲）。参照算子 layernorm 亦为输入/输出双缓冲但 gamma/beta 单槽——本算子按容量取全单槽。

---

## §4 循环与 Section 结构（R4 输出）

> 依据 [循环与Section结构设计](../../.opencode/skills/pypto-pro-op-design/references/loop_design.md)。

### 参考样例

- **主要参考样例**: `pro_ops/vf_api/test_layernorm_tile_group_vf.py`（EXPLORE_REPORT §4.4 首推）
- **可复用结构点**: 单 `section_vector` + strided 多核 `for tile_id in pl.range(core_id, num_tiles, num_cores)`；`set_validshape` 先于 load；`vf.full` 初始化跨 `pl.range` 的寄存器累加器（L89-95 `row_sum = vf.add(row_sum, part, preg)` 模式——本算子 acc 寄存器同款）；`vf.update_mask` 动态尾掩码。
- **补充参考**: `attention/dense_lightning_indexer_softmax_lse_v2/op_kernel/*.py`——Ptr 裸参 + TilingData 重建视图（L877-957）、`pl.getval` GM 标量读（resolve_b_s1）、运行时 tile 切片（`reduce_sum_vec_tile[0:1, ci*TKV_HALF:]`）、TilingKey 编译期分支；`pro_ops/fa/test_fa_tilingkey_attn_mask.py`——TilingKey 字典启动多实例。

### 本算子结构说明

- **Module / Section对应关系**: Module 1 → 唯一 `with pl.section_vector():`；tile 组与视图构造在 Section 之前（loop_design「两个Section共享的TileGroup在Section之前声明」；本算子单 Section，统一前置）。
- **结果单元**: 一个工作项 `(roi n, 通道块 cb)` 的输出块 `y[y_row, c0:c0+63, :, :]`（64 通道 × pH×pW 像素）。块内像素再按 PX_PHYS=64 分组暂存写出。
- **跨Tile状态生命周期**: acc 寄存器（每像素一个 64-lane fp32）——像素循环头 `vf.full(0.0)` 初始化，(i,j) chunk 循环内 `vf.add` 更新，像素尾 `vf.muls(inv_area)` 消费；无跨像素/跨工作项状态。分数表 frac_j/frac_i——每工作项头 VF 构建一次，像素循环内只读。strip——每 (像素, chunk) 重载，chunk 内只读。
- **Section结构**: 1 个 section_vector；循环全部在 Section 内；kernel 顶部（Section 外）完成 8+4 个 make_tensor 视图与 12 个 tile 组声明。
- **循环嵌套**（由外到内）:
  ```
  for wid in pl.range(core_id, R*n_cb, num_cores)          # 多核 strided（R5）
    n = wid // n_cb; cb = wid % n_cb                        # 通道块内层 → 同 roi 块分散多核
    [序言 S1–S8：标量链，含 3 个 continue 出口（batch 越界/负积/层级不匹配）]
    for ph in 0..pH-1:                                      # ph_f 回环 1 次/ph
      for pw in 0..pW-1:                                    # pw_f 回环 1 次/pw
        acc = vf.full(0)
        for i0 in 0..sh step SI:                            # y-chunk（SI 每 ROI 算好）
          for j0 in 0..sw step SJ:                          # x-chunk
            [chunk 窗口计算 → strip 64 切片装载 S11]
            for i in i0..min(i0+SI,sh):
              for j in j0..min(j0+SJ,sw):
                [S10 坐标 → S12 4×gather → S13 累加；越界样本 continue]
        [S14 scatter → out_tile 列 (px = ph*pW+pw) 取模 px 组]
    [px 组收尾：每 PX_PHYS 像素或像素末尾 → S15 cast+store]
  ```
  （实现上 px 组循环 `for px0 in 0..pH*pW step PX_PHYS` 包裹 ph/pw 双层，组内像素序 = `ph*pW+pw` 全序；两组循环等价，coder 取组显式版本。）

> **Stage 5 as-built 修订（循环结构，最终实现 = Cand-2；PERFORMANCE_REPORT §6.2）**：
> 像素循环改为 **ph 外层 / pw 内层嵌套**（消除每像素 `px//pooled_w`、`px%pooled_w` uint64
> 除法对与 ph_f/pw_f cast 回环——Exp-A），外层 `ph0` 行组 chunk 化（Exp-C）。结构：
> ```
> for wid in pl.range(core_id, R*n_cb, num_cores)      # 不变（wrapper block_dim 合同不变）
>   [序言 S1–S7：同 Stage 4；S8 分数表已删除；直转 CAST_FLOOR/CEIL；int→float 直接提升]
>   x 窗口一次推导（xc0..x_end, xs_roi；窗口边沿与逐样本表达式位级同构）
>   use_fast = (xs_roi ≤ 48) and (bin_h+2 ≤ ys_cap)    # ys_cap = min(56, 896//c_eff)
>   if use_fast:                                        # 快路径（P0 全覆盖）
>     K = clamp(floor((ys_cap-2)/bin_h), 1, pH)
>     for ph0 in 0..pH step K:                          # 行组 chunk
>       [行组 y 窗口 yc0..y_end, ys_c, pitch=ys_c*48]
>       bar_mte2 → 每通道整窗 TLOAD [ys_c, xs_roi] → strip 运行时 pitch 布局
>       for ph in ph0..ph1:                              # 行内像素
>         for pw0 in 0..pW step PX_PHYS:
>           for pw in pw0..pw0+n_cols:
>             acc = 0; for i in 0..sh:                   # y 侧 FLOOR 每 (ph,i) 一次
>               for j4 in 0..sw step 4:                  # x 侧 4 样本一次 TCVT
>                 [4 样本 pos/weight → _bilinear_acc4_vf（acc 往返 ÷4）]
>               for j in 尾部: [_bilinear_acc_vf]
>             _finish_pixel_vf(acc, out, pw-pw0)
>           [S15: 每行组尾 cast+store（锚点 [y_row*C+c0, ph*pW+pw0]）]
>   else:                                                # 回退路径（宽/高窗；correctness case 覆盖）
>     [Stage 4 同构 chunk 几何（si=floor((ys_cap-2)/bin_h), sj=floor(46/bin_w)），
>      适配宽 strip 视图 + 运行时 pitch；无分数表、无 ph/pw 除法]
> ```
> 已知回退路径代价：c_eff=64 时每通道行预算 28→14，bin_h∈(4,12] 的几何 y-chunk 数翻倍
> （PERFORMANCE_REPORT §6.1 dynamic +2.7% 根因）；P0 全走快路径不受影响，回退仅服务
> 宽/高窗 correctness case（add_chunk_x/xy/frac_fallback）。
- **动态循环上界**: `n_cb=(C+63)//64`、`sh`/`sw`（运行时，sample_num 或 ceil(bin)）、`SI=max(1, floor(26/bin_h_f))`（i32 标量）、`SJ=max(1, floor(22/bin_w_f))`（保证 chunk 窗口 ys≤28、xs≤24；26/22 为窗口预留 2 行/列边界余量）、chunk 数 `(sh+SI-1)//SI`。R=0 时 `pl.range(core_id, 0, num_cores)` 零迭代（KB vec-tensorlist-fixed-arity 实测空 range 语义）。
- **分核信息**: `num_cores = pl.get_block_num()`、`core_id = pl.get_block_idx()`，Section 内读取（纯 Vector kernel，无 subblock 展开问题）。

> 同步点见 §6（不涉及 cross_core）；尾块处理见 §7。

### CV 手动预加载流水设计（`is_fusion=true`时必填）

不适用（is_fusion=false，纯 Vector 单 Section，无 Cube/Vector 交替阶段链）。

---

## §5 分核策略（R5 输出）

> 📌 权威依据：`docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`。

### 分核方式

- **方案**: strided loop 扁平切——`for wid in pl.range(core_id, R*n_cb, num_cores)`，`n = wid // n_cb`、`cb = wid % n_cb`。选择理由：① 工作项 (roi, 通道块) 两两写不相交（owner model，§6），任意分配无写竞争；② 各工作项计算量同阶（同 ROI 尺寸分布），跨步分配最多差 1 项（教程负载均衡性质）；③ (roi, cb) 粒度保证 C=4096/R=1 类极端形状仍有 64 个并行项，避免纯 roi 切分在小 R 大 C 时的核饥饿。
- **host 侧 block_dim**: 纯 Vector Kernel 用 `vector_core_num`：`total_tasks = R * ((C+63)//64)`；`total_tasks > 0` 时 `block_dim = min(get_platform_info().vector_core_num, total_tasks)`；`total_tasks == 0`（R=0，场景 s1）时 `block_dim = 1` 空工作单次启动（`pl.range(core_id, 0, 1)` 零迭代——KB vec-tensorlist-fixed-arity 目标实测语义；禁止 block_dim=0 或跳过启动）。op_host 侧对应 `SetBlockDim(min(aivNum, R*n_cb))`，为 0 时置 1。
- **交付 launch 合同**: 恰好 1 次。唯一 kernel `roi_extractor`（`@pl.jit`）由 `roi_extractor_wrapper`（`custom/roi_extractor/test_roi_extractor.py`）在 host 循环外调用一次：`roi_extractor[None, block_dim, {"HasIndex": hi}](f0..f7, rois, index, y, tiling)`。
- **TensorList ABI（本算子即 8-OPTIONAL 展开形态）**:
  - 冻结合同 `L_MAX = 8`（SPEC §5：canndev `MAX_NUM_FEATURE_MAP=8`），`B_MAX = L_MAX = 8`——**一组** 8 个固定槽位（features_0..features_7），无长度阶梯；每个槽位为**独立 `pl.Ptr[pl.DT_UINT8]` 参数**，地址只存在于 launch 实参，**不进入 tiling 数据**（tiling 只携带 shape/层数/attr 派生量）。
  - 单一 TensorList 形参（features），无多列表长度关系问题；层数 `L` 经 TilingData `l` 字段传递。
  - `N_MAX` 安全性：所有 `DT_INT32/INDEX` 派生式上界——偏移 `b*C*H+c*H+y ≤ 128*4096*8192 ≈ 4.3e12` **超 int32**！偏移算术在 int64（TilingData int 字段为 INDEX/int64_t，dense_lightning `b_idx*s1_dim` 同款）下进行；`wid//n_cb`、`wid%n_cb`、`px//pooled_w`、`px%pooled_w` 为 int64 标量除法/取模（dense_lightning L413-416 在役先例）。**KB vec-tensorlist-fixed-arity 明文约束：除法算术标量保持 DT_INT32——后端无 64 位整数除法（`Div of bitwidth greater than 32 not supported`），且失败与否取决于类型推断（潜伏实例可编译直至某处拓宽除数）**。本设计因偏移值域必须 int64，除法/取模点偏离该默认纪律，不以在役先例豁免，立为活动 requirement **FA-11**（DESIGN_BINDINGS.json）：Stage 4 须目标端实测 int64 除法/取模路径可编译可运行（§7 验证点 VP5）；失败则回退 int32 化方案——除法/取模点操作数显式收窄 DT_INT32（SPEC §5 包络下 `wid < R*n_cb ≤ 640000`、`px < pH*pW ≤ 1048576`、`n_cb ≤ 64`、`pooled_w ≤ 1024` 均 int32 可表示），线性偏移算术保持 int64。
  - wrapper 对真实槽位校验（连续前缀 `features_0..features_{L-1}`、各层 N/C 一致、dtype 耦合 fp16/fp32、rois `[R,5]`、index 长度 =R、pool_mode="avg"、finest_scale∈[1,512]、pooled∈[1,1024]、sample_num∈[0,1024]、spatial_scale 长度≥L）；**缺席槽位复用 features_0 张量填充**（KB 指定形态：不构造 dummy 张量；host 在 tiling 中以 `feat_h/w=1, l=L` 标记缺席，kernel 仅访问 l<L 层，缺席层视图构造于裸指针之上不被解引用——dense_lightning 缺席 OPTIONAL 输入同款处理）；index 缺席时以 rois 张量占位（`HasIndex=0` 编译期死分支，不读取）。
  - **适用边界声明**：本 ABI 不是"连续且 rank 无关的逐元素"扁平化交付——各层 H/W 不同构，故每槽视图为 `make_tensor` + NCHW strides（KB vec-tensorlist-fixed-arity 扁平化变体不适用的声明已在 selection reason 冻结；仓内 dense_lightning 已验证该 Ptr+make_tensor 形态）。
  - **零总任务策略**: R=0 → `block_dim=1` 单次空启动（上文），不在 wrapper 跳过 launch。

### TilingData / TilingKey / workspace 契约（host-kernel 两侧严格一致）

**TilingData dataclass（op_kernel/roi_extractor.py）与 op_host/roi_extractor_tiling.h 的 TILING_DATA_FIELD_DEF 必须字段顺序/类型/对齐逐项一致**（构建系统 force-include codegen 生成的 tiling.h 强制校验，SPEC 接入合同）。数组字段用对应数组形态宏（int64_t[8] / float[9]）：

```python
@dataclass
class RoiExtractorTiling:
    r: int               # R（rois 行数）
    n: int               # N（batch）
    c: int               # C（通道）
    l: int               # 有效层数 1..8（features 连续前缀长度）
    pooled_h: int        # attr
    pooled_w: int        # attr（独立于 pooled_h）
    sample_num: int      # attr（0 = 自适应）
    finest_scale: int    # attr
    aligned: int         # attr bool → 0/1（运行时分支）
    is_fp16: int         # features dtype == fp16 ? 1 : 0（strip 装载路径分支）
    feat_h: int[8]       # 各层 H_l；l≥8 缺席层填 1（FA-04：不构造零尺寸视图）
    feat_w: int[8]       # 各层 W_l；同上
    spatial_scale: float[8]   # 前 L 项有效（attr ListFloat 前 8 项，超出部分填 0）
    b_pow2: float[9]     # B[i]=2^i*finest_scale，i=0..8（host 纯 attr 整数/浮点推导；
                         #   值为 fp16 精确表示的 fp32，kernel 内与 s16 做纯 fp32 比较 ≡ fp16 比较）
```

**TilingKey 裁定**（任务书候选字段 has_index / aligned / sample_num_mode 逐一裁定）：

| 候选字段 | 裁定 | 理由 |
|---|---|---|
| `HasIndex` | **采纳**（bits=1, values=[0,1]） | 选择结构性代码路径：index 张量读取（getval）+ 输出行重映射分支，编译期折叠消除另一分支；binary 交付枚举 2 个合法 key |
| `aligned` | 不采纳（tiling 运行时字段） | 仅影响序言中 4 次标量减法（每 ROI 一次），非结构性路径，运行时分支零收益 |
| `sample_num_mode` | 不采纳（tiling 运行时字段） | 仅影响 sw/sh 标量计算来源，采样循环边界本为运行时值，无代码路径分叉 |
| dtype (fp16/fp32) | 不采纳（`datatype={"rois": "input_dtype"}` 机制承载） | dense_lightning 同款（L859-861）：dtype 特化由 @pl.jit datatype 完成，TilingKey 不重复承载 |
| L (1..8) | 不采纳（tiling.l 运行时字段） | 层访问由运行时层级选择守卫（l<L），无死代码可消除的编译期分叉 |

```python
class RoiTilingKey:
    HasIndex = TilingKeyField(bits=1, values=[0, 1])   # values=[0,1]：编码位 == 实际值，
                                                       # host 直接 SetTilingKey(hasIndex) 按位 OR
```

- **kernel 侧**：`@pl.jit(auto_mutex=True, tiling_key=RoiTilingKey, datatype={"rois": "input_dtype"})`；`if HasIndex == 1:` 读 index 视图（缺席时编译期死分支）。
- **host 侧**（op_host/roi_extractor_tiling.cpp）：`context->SetTilingKey(static_cast<uint64_t>(hasIndex))`——单字段 [0,1] 形态下打包值即位 OR 结果，与 dense_lightning tiling.cpp L437-440 的 DenseLiLseTilingKey 位打包机制同构（多字段时按类定义顺序 `field_value << offset` 累加 OR，本设计仅 bit0）；JIT 侧以 key 字典 `{"HasIndex": hi}` 启动（fa_tilingkey 样例形态）。

**workspace 裁定：不使用**。理由：① 语义无必需辅助张量（SPEC kernel 契约补充：canndev 的 roi_buf 中转是其 tiling 策略实现细节）；② 本设计每工作项状态（strip 168KB + out_tile 16KB + 杂项 ≈ 189KB，见 §3 高水位 193248B）全部驻留单核 UB，无跨核汇聚、无溢出转存（样本累加为寄存器流式，无长轴驻留需求——与 dense_lightning 因 S2 超 UB 才引入 workspace 的场景不同构）；③ owner 分解无跨核写竞争，无需 GM 仲裁区。op_host 侧 workspace 仅为系统保留：`workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize()`（dense_lightning 同款下限）。

**op_host 侧配套**（SPEC 落位合同，Stage 4 交付）：
- `roi_extractor_def.cpp`：8×OPTIONAL features（DataType {ge::DT_FLOAT16, ge::DT_FLOAT}，AutoContiguous）+ REQUIRED rois（同 dtype 集）+ OPTIONAL index（DT_INT32）+ REQUIRED y（同 dtype 集）；ATTR 顺序固定：finest_scale(Int,56)、roi_scale_factor(Float,0)、spatial_scale(ListFloat,{0.25,0.125,0.0625,0.03125})、pooled_height(Int,7)、pooled_width(Int,7)、sample_num(Int,0)、pool_mode(String,"avg")、aligned(Bool,true)；AICore config 仿 dense_lightning（DynamicCompileStaticFlag/DynamicFormatFlag/DynamicRankSupportFlag/DynamicShapeSupportFlag + AddConfig("ascend950")）。
- `roi_extractor_infershape.cpp`：y = [R, C, pooled_height, pooled_width]（**独立使用两 attr**，锁定 SPEC 决策 2 的 canndev 笔误修复）；InferDataType 输出 = features dtype。
- `roi_extractor_tiling.cpp`：镜像 wrapper 参数校验 → 填 TilingData（含缺席层 h/w=1、b_pow2、spatial_scale 前 8 项）→ `SetTilingKey(hasIndex)` → `SetBlockDim(min(aivNum, R*n_cb))`（为 0 置 1）→ workspace = 系统保留。
- CMake/目录：按 SPEC「目标仓落位与目录结构」合同（require_pypto_pro/enable_pypto_kernel/Ascend950PR_9599 条件项/--cce-auto-sync=off，仿 dense_lightning op_host/CMakeLists.txt）。

---

## §6 核间同步（R6 输出）

> 依据 [跨核同步](../../.opencode/skills/pypto-pro-op-design/references/cross_core_synchronization.md)。

### cross_core 涉及判定

- **是否涉及 cross_core**: **不涉及**
- **依据**: ① 纯 Vector 单 Section Kernel，无 Cube↔Vector 执行域交接；② 工作项按 owner model 切分——每个输出元素 `y[y_row, c, ph, pw]` 恰属于一个工作项 `(n, cb)`（index 为 [0,R) 排列时每行恰一写者；重复映射为 SPEC 声明的未定义输入，不设计仲裁——KB vec-scatter-owner-model §1/§3），**无跨 Block/subblock 写竞争、无原子操作、无共享跨核缓冲**；③ 全部核内依赖（MTE2↔V↔S↔MTE3）由 `make_tile_group` mutex 元数据 + auto_mutex 覆盖（§2/§3）。
- **流水实现**: 不适用。

### 共享数据与槽位映射

不涉及。

### 同步点

不涉及。

### event_id 分配表

不涉及（无 cross_core 事件；mutex_id 分配见 §2 TileGroup 表，11 个 ∈ [0,31]）。

---

## §7 尾块处理（R7 输出）

> 📌 权威依据：`docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md`。

### 尾块处理方案

本算子无"矩阵维度 ÷ tile"式静态尾块——所有尾块均由**运行时值驱动的动态窗口**处理，物理 tile 恒为合法编译期常量：

```python
# 1) 工作项数向上取整（通道尾块）
n_cb = (C + CH_BLOCK - 1) // CH_BLOCK          # C=65 → 2 块，末块 c_eff = min(C-c0, 64)
c_eff = pl.min(tiling.c - cb * CH_BLOCK, CH_BLOCK)
preg_ch = vf.update_mask(c_eff, dtype=pl.DT_FP32)   # 尾通道 lane 屏蔽（gather/算术/scatter 全链）

# 2) strip 切片装载：物理子块 [c*28:(c+1)*28, 0:24]，先设有效窗再 load（铁则：set_validshape 先于 load）
slice = strip[c*STRIP_YS:(c+1)*STRIP_YS, 0:STRIP_XS]
pl.set_validshape(slice, [ys, xs])             # ys∈[1,28], xs∈[1,24] 运行时窗口（含 1×1 退化窗）
# fp32 模式：
pl.load(slice, lvl_view, [b*C*H + c*H + yc0, xc0])   # 仅搬有效区，不越界读 GM
# fp16 模式（load 目的 tile 为 staging，同样先设有效窗再 load）：
pl.set_validshape(staging, [ys, xs])           # staging 物理 [28,32]，有效窗与 slice 同窗——
                                               # 约束 GM 搬入 + 定义 cast 源有效区域
pl.load(staging, lvl_view, [b*C*H + c*H + yc0, xc0])
pl.cast(slice, staging, CAST_NONE)             # 源/目的物理 shape/行跨度可不同（cast.md），
                                               # 转换区域 [ys,xs] 内元素按逻辑坐标逐元素对应
# 装载坐标 yc0/xc0 预先 clamp 进 [0,H-1]/[0,W-1] 且 ys/xs≥1 —— 全越界 ROI 的退化窗仍合法

# 3) gather 前恢复 strip 物理满幅窗（消除切片窗口残留歧义）
pl.set_validshape(strip_current, [CH_BLOCK*STRIP_YS, STRIP_XS])

# 4) 像素分组尾块：末组 n_px = pH*pW - px0 < PX_PHYS
pl.set_validshape(out_tile, [c_eff, n_px])     # cast 与 store 均只处理有效区（cast 的转换区域
                                               # 由 out.valid_shape 定义——先设后 cast）

# 5) R 尾块：strided range 自然覆盖（core_id 起、num_cores 步），多出的核零迭代
```

- **compact**: `None`（纯 Vec ND 路径，tail_block_handling「当前A5的Vec ND load/store 不需要 compact」；无分形/矩阵路径）。
- **无效区域是否会被读取**: ① strip 无效区：gather 索引按构造 ≤ 活动范围（活动 lane 偏移 = c*672 + y*24 + x，y<ys、x<xs 恒成立）；非活动通道 lane（≥c_eff）读到的 stale 数据被 `preg_ch` 在算术与 scatter 处屏蔽——**且 mins 钳制保证其索引恒在物理范围内**（gather.md「索引必须在 Tile 有效地址范围内」双保险）。② out_tile 无效区：cast/store 均以 valid_shape 收窄，不写 GM。③ 归约污染：acc 初始化 `vf.full(0)` + `preg_ch` 屏蔽 + 越界样本 `continue`（贡献 0 但分母不变，SPEC §3 语义），无 fillpad 需求。

### 尾块相关验证点（Stage 4 须逐项确认）

| # | 验证点 | 依据 |
|---|--------|------|
| VP1 | 切片与 fp16 路径 `staging` 的 `set_validshape` 均先于各自 `pl.load`（同窗 [ys,xs]）；每 chunk 每通道重设 | tail_block_handling L74-80 |
| VP2 | gather 索引 `vf.mins(off, STRIP_ELEMS-1)` 钳制 + `preg_ch` 屏蔽双保险 | gather.md 约束 + vec-ub-strip-gather「out-of-range is undefined」 |
| VP3 | 64 切片装载完成后对 strip 全 tile 重设物理满幅 valid，再进 gather | 消除切片/父 tile valid 语义歧义（memory-layout「rows stay at declared width」） |
| VP4 | out_tile 的 `set_validshape([c_eff, n_px])` 在 `pl.cast` 之前（转换区域由 out.valid_shape 定义） | cast.md 参数说明 |
| VP5 | 运行时 int64 标量除法/取模（`wid//n_cb`、`wid%n_cb`、`px//pooled_w`、`px%pooled_w`）编译通过且运行正确；失败（`Div of bitwidth greater than 32 not supported` 或运行错误）→ 执行 FA-11 int32 化回退（除法/取模点操作数收窄 DT_INT32，线性偏移保持 int64）并重跑全 case | KB vec-tensorlist-fixed-arity「Keep every scalar in this arithmetic at DT_INT32」（FA-11 活动义务）+ dense_lightning L413-416 在役先例 |
| VP5′ | **Stage 5 as-built**：`px//pooled_w`、`px%pooled_w` 已随 ph/pw 嵌套化删除（§4 修订）；FA-11 int64 除法点收缩为 `wid//n_cb`、`wid%n_cb` 两处（Stage 4 目标端实测通过的同一形态，VP5 证据沿用） | PERFORMANCE_REPORT §6.1 Exp-A；生成物无 px 除法 |
| VP6 | fp16 模式 `pl.cast(out_f16, out_tile, CAST_RINT)` 同地址原地转换的正确性 | cast.md「DT_FP32→DT_FP16 支持完全相同起始地址原地转换」 |
| VP7 | `vf.scatter` 索引 lane*64+px 逐 lane 唯一（px 组内像素不重复） | scatter.md「index 中的值必须唯一」 |

---

## §8 目标测试 case（R7.5 输出 → 交付 develop）

> SPEC 已给出 P0 case（机器合同 p0_cases 1-4 + §10 场景矩阵 s1-s6），在其基础上追加设计导出的基础/边界 case。develop 直接实现，不再自行重算。数据 recipe 严格按 SPEC §10（含 don't-care 行构造与 NaN 排除比对）。

| case 名 | 具体 shape / 参数 | 覆盖场景 | design 适配确认 |
|---------|------------------|---------|----------------|
| `ut_multilevel_fp16`（P0-1） | features [1,16,16,16]/[1,16,9,9]/[1,16,8,8]/[1,16,4,5]，rois [1000,5]，fp16，默认 attr | 4 层选择全区间 + 贴边/零面积/越界/s 边界/batch 越界（995-999 don't-care） | ✅ 层级链 S3 全分支；窗口 ≤18×18 单 chunk；c_eff=16<64 尾通道屏蔽 |
| `st_singlelevel_fp32`（P0-2） | features_0 [5,64,76,76]，rois [150,5]，fp32 | **fp32 特化模式**全程（ datatype 机制） | ✅ staging 旁路（直接 load strip）；无 cast 展宽路径 |
| `dynamic_singlelevel_fp16_index`（P0-3） | features_0 [4,256,200,336]，rois [1200,5]，index=reverse 排列，fp16 | **HasIndex=1 键实例**；L=1 大 RoI 丢弃（~20% don't-care）；R=1200 多核负载 | ✅ y_row=index[n] 编译期分支；丢弃行不写 |
| `multilevel_n2_sample2`（P0-4） | 4 层金字塔 [2,32,48..6]，rois [64,5]，sample_num=2，fp16 | 固定采样数模式 + 跨 batch_idx | ✅ S5 固定分支；R×n_cb=64 项多核 |
| `s1_empty_roi` | 同 P0-1 features，rois [0,5] | **R=0 空工作**：y [0,16,7,7]，block_dim=1 单次空启动 | ✅ pl.range 零迭代（KB 实测语义） |
| `s2_pooled_4x8` | 同 P0-1 features，rois [256,5]，pooled_height=4, pooled_width=8 | **非方形 pooled**（独立 pH/pW，锁定 SPEC 决策 2）；pH*pW=32<64 单 px 组 | ✅ bin_w/bin_h 独立分母；px=ph*8+pw 映射 |
| `s3_l1_drop` | features_0 [1,16,32,32]，rois [16,5]（8 小 8 大），fp16 | L=1 大 RoI 丢弃路径显式触发 | ✅ S3 `L==1 且 s>B[1] → continue` |
| `s6_eight_level` | 8 级金字塔 [1,16,512..4]，rois [512,5]，spatial_scale 8 项 | **L=8 全槽位**（8-OPTIONAL ABI 满载） | ✅ 层视图 if 链到 features_7；tiling float[8] 满载 |
| `add_aligned`（追加） | features [2,64,56,56] fp16，rois [128,5]，pooled 7×7 | **全整除基准**：C=64 恰 1 块（c_eff=64 无尾）、R=128、pH*pW=49<64 单组 | ✅ 无任何尾块路径触发 |
| `add_ctail`（追加） | features [2,65,56,56] fp16，rois [96,5] | **通道尾块**：C=65 → 2 块，末块 c_eff=1 | ✅ preg_ch=update_mask(1)；末块仅 1 活跃通道 |
| `add_multitile`（追加） | features [4,256,200,336] fp16，rois [1024,5] | **跨多工作项**：R×n_cb=4096 项 > 核数，跨步分配 + R 尾余数核 | ✅ strided range 负载均衡性质 |
| `add_chunk_x`（追加） | features_0 [1,64,256,256] fp16，scale 0.25，rois 含 700×700 原图尺寸（特征 175px，bin≈25） | **x-chunk 路径**：bin_w+2=27>24 → SJ 分块；y 不分块 | ✅ S11 窗口 [yc0,yc0+ys)×[xc0,xc0+xs) 分块装载；SI/SJ 自适应 |
| `add_chunk_xy`（追加） | features_0 [1,64,256,256] fp16，scale 0.25，rois 含 800×800 原图尺寸（特征 200px，bin≈28.6→sw=29） | **双轴 chunk**：sw=29>24 且 sh=29>28 → i/j 双分块 + 多寄存器组样本跨 chunk 累加 | ✅ acc 寄存器跨 (i0,j0) chunk 持久（layernorm 累加器同款） |
| `add_frac_fallback`（追加） | 同 add_chunk_xy 但 sample_num=0、构造 sw>128 的极端宽扁 roi（特征宽 >896px） | **分数表回退路径**：frac_j 128 表不覆盖 → 逐样本 [1,8] cast 回环 | ✅ S10 回退分支（仅路径覆盖，不设性能预期） |

> **Stage 5 as-built 修订（§8）**：分数表已删除（Exp-B），`add_frac_fallback` 的覆盖语义
> 变为「sw>128 极端宽扁 roi 走**回退路径**（xs_roi > 48 的逐像素 chunk 几何 + 逐样本算术
> 分数）」——case 本身与断言不变（仍 14/14 PASS，PERFORMANCE_REPORT §6.2），仅路径名称
> 从"分数表回退"改为"宽窗回退"。Stage 4 遗留 F3（64<sw≤128 分数表第二块 base=64 路径的
> 覆盖缺口）随分数表删除而自然消失（Stage 5 探针 `docs/perf/stage5_f3_frac_probe.py`
> 10 行全过，先行闭合后删除）。

> 适配确认依据：strip 窗口预算（P0 包络内最大窗口 ~32×32 → 单 chunk 或 2×1 chunk）；UB 高水位 76.1%（§3）；所有 case 的 don't-care 行按 SPEC §9 NaN 约定排除比对。

---

## §9 综合评估（R8 输出）

### 准确性检查

| 检查项 | 结果 | 证据 |
|--------|------|------|
| API 调用链完整覆盖数学公式 | ✅ | SPEC 公式 (1)→S3、(2)→S4、(3)→S5、(4)→S2、(5)→S10-S13、(6)→S6/S14/S15、bilinear 边界→S10 越界判据/clamp/floor/min 链，逐步映射于 §1 |
| §4 已参照官方样例确定循环与 Section 结构（含参考样例路径与结构说明） | ✅ | layernorm（strided 多核/寄存器累加器/validshape）+ dense_lightning（Ptr 视图/getval/运行时切片/TilingKey）+ fa_tilingkey（key 字典启动），§4 参考样例节 |
| 数据依赖正确（Module 顺序 + sync 点） | ✅ | 单 Module 单向链；核内 pipe 依赖全由 auto_mutex 覆盖（§2 组表）；无跨核依赖（§6） |
| dtype 精度满足要求 | ✅ | 坐标/权重/累加全 fp32；fp16 输入装载时展宽（widen→compute→accumulate，precision.md 顺序）；输出单次 RINT；0.005/0.005 容差链见 §1 数值边界 |
| 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 layout | ✅（不涉及） | 无 vf.reduce_* 调用（累加为逐样本 vf.add 寄存器链，非归约指令） |
| Acc tile 物理 M×N×dtype_bytes ≥ fractal | ✅（不涉及） | 无 Acc/L0C 路径 |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 目标测试 case（≥4）已按 tile 切分确定具体 shape，且逐个验证 design 可适配 | ✅ | §8 共 14 case（P0×4 + s×4 + 追加×6），逐行适配确认 |
| 是否正确处理了尾块（M/N 尾块） | ✅ | 通道尾块（c_eff/preg_ch）、像素组尾块（n_px valid）、strip 动态窗口（ys/xs）、R 余数（strided），§7 |
| 归约轴可能超单Tile时，已给出覆盖完整归约轴的方案 | ✅（不涉及） | 样本轴以寄存器累加 + chunk 流式处理，无跨 tile 归约状态（§0 归约轴结论） |
| 循环边界正确 | ✅ | 全部 ceiling division（n_cb、chunk 数、px 组）；SI/SJ 自适应推导含边界余量（§4） |
| 超越函数在 dtype 范围内无溢出 | ✅ | 仅 vf.sqrt：prod<0 前置丢弃 + fp32 范围分析（§1 表） |
| 窄 dtype（fp16）中间值量级上界在 dtype 范围内 / 升位宽先于增长步 | ✅ | fp16 仅存在于装载中转与输出单次舍入；一切运算 fp32（strip 恒 fp32 策略，§2.1 注） |
| 跨 tile 状态初始化和持久化正确 | ✅ | acc 每像素 vf.full(0) 初始化；frac 表每工作项重建；strip 每 chunk 重载（§4 生命周期） |
| cross_core 同步方案正确 | ✅（不涉及） | §6 判定：owner model 无写竞争、单执行域、无共享跨核缓冲 |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ | §1 调用链的每个 tile 操作数在 §2 表有行、§3 有地址；§4 循环变量与 §1 步骤号一一对应；§5 工作项 = §4 最外层循环 |
| 所有决策有证据支撑 | ✅ | API 引用逐条标注文档路径；结构引用官方样例行号；KB 引用见 DESIGN_BINDINGS.json |
| R3 片上地址范围与逐空间容量检查 | ✅ | UB 193248/253952=76.1%（staging 改形 [28,32] 后复核：余量 23.9%，容量键 `ub_size=253952` 本机 ini 复核一致；216KB 口径被 arch-a5.md 推翻不采用）；唯一同地址对为文档化原地 cast；规则见 R3、证据见 §3 表 |
| `tile_dims` 使用时已关注大 stride 影响 | ✅（不涉及） | 未使用 tile_dims |
| 条件性检查（§6 不涉及 cross_core 时确认无跨域依赖） | ✅ | 纯 Vector 单 Section + owner 写不相交（§6 判定依据①②③） |

### 评估结论

- **整体**: **通过**
- **限制条件**: ① 性能非本轮门禁（perf_target=null）：strip 单缓冲无 MTE2/V 重叠、逐像素小窗装载、标量坐标回环为已知保守点，Stage 5 优化钩子见下；② 极端宽扁 roi（sw>128）走分数表回退慢路径（正确性优先）；③ 运行时 int64 标量除法/取模偏离 KB int32 默认纪律，由活动 requirement FA-11 约束：Stage 4 目标端实测闭合（VP5），失败则 int32 化回退（§5）。
- **修改记录**: R1→R2 迭代中废除"fp16 模式 128 通道块 + astype ZERO/ONE 分裂"候选（DT_UINT16 索引 ≤65535 窗口约束 + layout 语义歧义），改为 **strip 恒 fp32 + 装载处展宽**（CH_BLOCK=64 两模式统一），同步更新 §1/§2/§3；R2→R3 迭代中 strip 由 [64*24,24]（内维 96B）复核对齐后定形 [1792,24]；**stage3-check FAIL 修订（D1/D2/D3）**：D1 staging 改形 [28,24]→[28,32]（fp16 内维 48B→64B 满足 load.md「Tile shape 需要 32 字节对齐」；cast 源/目的物理 shape 不同由 cast.md L103 允许），§1 vector_selection/§2 属性表与对齐注记/§3 地址表与高水位（UB 193248/253952=76.1%，容量键本机 ini 复核；216KB 口径按 arch-a5.md 不采用）/§8/§9/§10 同步；D2 S11 fp16 路径补 `set_validshape(staging,[ys,xs])` 先于 `pl.load`（tail_block_handling L74-80），§7 代码块/VP1/§10 与 TV-02/TV-05 枚举同步；D3 新增活动 requirement FA-11 捕获 KB int32 除法约束（后端无 64 位整数除法），§5/VP5/§9 限制条件改为目标端实测闭合 + int32 化回退，撤销原"告警针对其后端除法路径"的消解主张。
- **Stage 5 优化钩子（非本轮义务，仅记录）**: ph 行内像素共享 y 带 → (ph, 列带) 共享 strip 可减 ~7× 装载次数；i/j 可分离坐标表可减 (sh-1)×sw 次坐标回环；strip 缩容换双缓冲；同 ROI 多通道块坐标序言复用。

> **Stage 5 钩子兑现记录（2026-09-08，PERFORMANCE_REPORT §6/§7/§8）**：
> ① 「(ph, 列带) 共享 strip」→ **已兑现并超越**（Exp-C：ROI 级 x 窗口 + ph 行组整窗共享，
> 装载次数 ut 49×、dynamic 16× 下降，聚合 4.04×）；② 「i/j 可分离坐标表」→ 部分兑现
> （y 侧每 (ph,i) 单 cast + x 侧 4 样本合批 TCVT；VF 域整体接管被 u32 墙阻断，探针 P1/P2）；
> ③ 「strip 缩容换双缓冲」→ 评估后拒绝（BD-3：2×172032B > UB 253952B，半通道拆分恶化
> vec 首管道）；④ 「同 ROI 多通道块序言复用」→ 拒绝（general-07：序言占比 ~0.2%，
> 低于 2% 晋级线，且并行度反向）。

---

## §10 Tile 数据流全景图

```
GM features_0..7 ──┐ (lvl_view = make_tensor [N*C*H_l, W_l]，l 由 S3 选中)
GM rois [R,5] ─────┤
GM index [R] ──────┤ (HasIndex==1 才读)
GM y [R,C,pH,pW] ←─┤ (y_view = [R*C, pH*pW])
                   │
   ════════════════╪════════ Module 1 · section_vector（每工作项 (n, cb)）════════
                   │
 GM rois ─[pl.load 行, set_validshape[1,5] 先行]→ rois_raw[1,16] ─[pl.cast CAST_NONE]→ rois_f32[1,8]
                   │                                              └─[pl.getval ×5]→ 标量链（S2-S5）
 GM index ─[pl.getval]→ y_row（HasIndex==1 编译期分支）                              │
                   │                                                                ▼
                   │        coord_f32a[1,8] ⇄(pl.cast FLOOR/CEIL/TRUNC/RINT)⇄ coord_f32b/coord_i32/coord_f16
                   │        （标量取整/类型回环：b、sw/sh、ph_f/pw_f、逐样本 x_lo/y_lo/lx/ly、s16）
                   │                                                                │
                   │        frac_j/frac_i [1,128] ←[VF: arange→astype→adds→muls→store_align]─(每 ROI 一次)
                   │                                                                │
 GM features_l ─[pl.load 切片×c_eff, set_validshape[ys,xs] 先行（fp16: staging 同窗先行）]→ (fp16: staging[28,32]
                   │        ─[pl.cast CAST_NONE]→) strip 切片 ──→ strip[1792,24] FP32（UB 0x00000, 168KB 驻留）
                   │                                              │ [装载毕: set_validshape(strip) 物理满幅]
                   │                                              ├─[vf.gather ×4/样本, u32 idx = arange×672
                   │                                              │   + adds(pos) + mins 钳制, preg_ch]→ RF v00..v11 (64ch fp32)
                   │                                              │        ├─[vf.muls 标量权重 + vf.add]→ acc 寄存器（跨 chunk 持久）
                   │                                              │        └─ 越界样本: continue（0 贡献, 分母不变）
                   │                                              └──(fp16 装载路径的 staging 循环复用, mutex 2)
                   │
 acc ─[vf.muls inv_area]→ out 列 ─[vf.scatter, idx=arange×64+px, preg_ch]→ out_tile[64,64] FP32（UB 0x2A000）
                   │                                              │ [set_validshape(c_eff, n_px) 先行]
                   │                              (fp16) ─[pl.cast CAST_RINT 原地]→ out_f16[64,64] FP16（同地址, mutex 1 共用）
                   │                                              │
                   └────────────────────[pl.store, 偏移 [y_row*C+c0, px0]]→ GM y_view → y[y_row, c0:c0+63, :, :]
```

### 图例

| 标记 | 含义 |
|------|------|
| `[pl.load]` / `[pl.store]` | MTE2 / MTE3 搬运（GM ↔ UB） |
| `[pl.cast ...]` | V 流水 tile 级类型/舍入转换 |
| `[vf.*]` | VF 寄存器级操作（@pl.vector_function 内） |
| `[pl.getval]` / 下标写 | S 流水标量读写 |
| `⇄` | 标量转换回环（同一小 tile 组上 S/V 交替，auto_mutex 同步） |
| `──→` / `├─` | 数据流向 / 同源分支 |
| `（UB addr）` | §3 地址表对应区段 |

> 全景图每个 tile（rois_raw/rois_f32/coord_*/frac_*/staging/strip/out_tile/out_f16）与每个操作均可在 §2 Tile 属性表、§3 地址映射表、§1 API 序列中找到对应条目；无跨 Module 持久化（单 Module），故无 `---` 虚线元素。

> **Stage 5 as-built 修订（§10）**：上图按 Stage 4 形态绘制。最终实现差异：`frac_j/frac_i`
> 节点已删除（算术分数）；`strip[1792,24]` → `[896,48]` 运行时 pitch（lane 间距
> `arange×(ys_c*48)`）；`coord_f32b` 回环删除（直转 CAST_FLOOR）；gather 每样本 4 次不变，
> 快路径以 4 样本合批 `_bilinear_acc4_vf` 摊薄 acc 往返；其余流向与同步语义不变。


