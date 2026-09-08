# 算子需求规范

## 机器合同

> 下列 JSON 是公式、公开接口、验收配置的唯一机器事实源。正文只解释语义与证据，不复制字段值。

```json machine-contract
{
  "schema_version": 1,
  "op_name": "roi_extractor",
  "formula": "y = roi_extract(features_0..features_{L-1}, rois, index) 逐 ROI n ∈ [0,R):\n(1) 层级选择: s = fp16(sqrt_fp32((rois[n,3]-rois[n,1])*(rois[n,4]-rois[n,2]))); 阈值 B[i]=2^i*finest_scale (i=0..L, fp16 精确表示); l=0 若 s<=B[1]; l=L-1 若 L>1 且 s>B[L-1]; 否则 l 为唯一满足 B[l]<s<=B[l+1] 的层 (1<=l<=L-2); L>=2 时任意 s 恰映射一层; L==1 且 s>2*finest_scale 时该 ROI 输出行不写入(内容未定义)\n(2) 坐标换算(fp32; fp16 输入先转 fp32): x0,y0,x1,y1 = rois[n,1..4]*spatial_scale[l]; aligned=true 时四坐标各减 0.5; roi_w=max(x1-x0,1.0); roi_h=max(y1-y0,1.0); bin_w=roi_w/pooled_width; bin_h=roi_h/pooled_height\n(3) 采样数: sample_num>0 时 sw=sh=sample_num; 否则 sw=ceil(bin_w), sh=ceil(bin_h)\n(4) batch 守卫: b=floor(rois[n,0]); 仅 0<=b<N 时计算, 否则该 ROI 输出行不写入(内容未定义)\n(5) 采样: 对 ph∈[0,pooled_height), pw∈[0,pooled_width), i∈[0,sh), j∈[0,sw): x=x0+(pw+(j+0.5)/sw)*bin_w; yc=y0+(ph+(i+0.5)/sh)*bin_h; acc[c]+=bilinear(features_l[b,c,:,:],x,yc)\n(6) 输出: y_row = (index 提供时) index[n] 否则 n; y[y_row,c,ph,pw] = acc[c]/(sh*sw) 对全部 c∈[0,C); 越界样本贡献 0 且仍计入分母 sh*sw\nbilinear(fm,x,y) (fm 空间尺寸 HxW, x 沿 W, y 沿 H, 输入为原始坐标): 若 x<-1 或 x>W 或 y<-1 或 y>H 返回 0; x=clamp(x,0,W-1); y=clamp(y,0,H-1); x_lo=min(floor(x),W-1); x_hi=min(x_lo+1,W-1); y_lo=min(floor(y),H-1); y_hi=min(y_lo+1,H-1); lx=x-x_lo; hx=1-lx; ly=y-y_lo; hy=1-ly; 返回 fm[y_lo,x_lo]*hy*hx+fm[y_lo,x_hi]*hy*lx+fm[y_hi,x_lo]*ly*hx+fm[y_hi,x_hi]*ly*lx",
  "supported_dtypes": ["float16", "int32"],
  "inputs": [
    {"name": "features_0", "shape": ["N", "C", "H_0", "W_0"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_1", "shape": ["N", "C", "H_1", "W_1"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_2", "shape": ["N", "C", "H_2", "W_2"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_3", "shape": ["N", "C", "H_3", "W_3"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_4", "shape": ["N", "C", "H_4", "W_4"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_5", "shape": ["N", "C", "H_5", "W_5"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_6", "shape": ["N", "C", "H_6", "W_6"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "features_7", "shape": ["N", "C", "H_7", "W_7"], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "rois", "shape": ["R", 5], "dtype": "float16", "value_range": [-65504.0, 65504.0]},
    {"name": "index", "shape": ["R"], "dtype": "int32", "value_range": [0, 9999]}
  ],
  "outputs": [
    {"name": "y", "shape": ["R", "C", 7, 7], "dtype": "float16", "value_range": [-65504.0, 65504.0]}
  ],
  "default_params": {
    "finest_scale": 56,
    "roi_scale_factor": 0.0,
    "pooled_height": 7,
    "pooled_width": 7,
    "sample_num": 0,
    "pool_mode": "avg",
    "aligned": true
  },
  "tolerance": {"atol": 0.005, "rtol": 0.005},
  "dynamic_axes_ranges": {
    "N": [1, 128],
    "C": [1, 4096],
    "R": [1, 10000],
    "H_0": [1, 8192], "W_0": [1, 8192],
    "H_1": [1, 8192], "W_1": [1, 8192],
    "H_2": [1, 8192], "W_2": [1, 8192],
    "H_3": [1, 8192], "W_3": [1, 8192],
    "H_4": [1, 8192], "W_4": [1, 8192],
    "H_5": [1, 8192], "W_5": [1, 8192],
    "H_6": [1, 8192], "W_6": [1, 8192],
    "H_7": [1, 8192], "W_7": [1, 8192]
  },
  "shape_constraints": [
    "L=有效层数：features_0..features_7 中实际提供的个数，必须构成 features_0..features_{L-1} 连续前缀，1<=L<=8；层数经 TilingData 传递",
    "各有效层 N、C 维必须一致；H_i、W_i 可不同；有效层与 rois、y 的 dtype 必须一致（float16 或 float32 二选一，经 @pl.jit datatype 特化；机器合同以 float16 为 canonical dtype，float32 为等价第二模式，见 §5/§7）",
    "index 恒为 int32，长度必须等于 R，值域 [0,R)；提供 index 时输出行号=index[n]，否则=n；重复映射行为未定义",
    "spatial_scale 为 ListFloat，长度必须 >= L，实际使用前 L 项；默认 [0.25, 0.125, 0.0625, 0.03125]（非标量参数，故不入 default_params，全部 p0_case 均用默认值）",
    "y = [R, C, pooled_height, pooled_width]；pooled_height 与 pooled_width 为两个独立 attr（默认 7/7，支持区间 [1,1024]，canndev 动态实现存在 pool_w=pooled_h 笔误，重实现按接口契约正确支持独立值）；机器合同将 pooled 两维按默认值 7 具体化，非默认 pooled 尺寸场景见 §10 场景矩阵",
    "机器合同 p0_case 约定：features_i 形状 [N,C,1,1] 表示该层缺席（真实层在 p0_case 中不使用 1x1 尺寸）；所有 p0_case 的 index 形状 [R] 为占位，case 名以 _index 结尾表示该 case 实际提供 index 输入",
    "R=0（空 RoI）合法：y=[0,C,pooled_height,pooled_width]，kernel 不写任何数据并正常返回；机器合同符号区间仅覆盖可具体化为正数维的 R>=1，R=0 场景见 §10",
    "finest_scale ∈ [1,512]（保证全部层级阈值 2^i*finest_scale 在 fp16 有限值域内精确表示）；sample_num ∈ [0,1024]；pool_mode 仅支持 avg，非 avg 报参数错误；roi_scale_factor 接受但不参与计算（与 canndev 参考实现一致）"
  ],
  "p0_cases": [
    {
      "name": "ut_multilevel_fp16",
      "params": {"finest_scale": 56, "roi_scale_factor": 0.0, "pooled_height": 7, "pooled_width": 7, "sample_num": 0, "pool_mode": "avg", "aligned": true},
      "input_shapes": {
        "features_0": [1, 16, 16, 16],
        "features_1": [1, 16, 9, 9],
        "features_2": [1, 16, 8, 8],
        "features_3": [1, 16, 4, 5],
        "features_4": [1, 16, 1, 1],
        "features_5": [1, 16, 1, 1],
        "features_6": [1, 16, 1, 1],
        "features_7": [1, 16, 1, 1],
        "rois": [1000, 5],
        "index": [1000]
      },
      "output_shapes": {"y": [1000, 16, 7, 7]}
    },
    {
      "name": "st_singlelevel_fp32",
      "params": {"finest_scale": 56, "roi_scale_factor": 0.0, "pooled_height": 7, "pooled_width": 7, "sample_num": 0, "pool_mode": "avg", "aligned": true},
      "input_shapes": {
        "features_0": [5, 64, 76, 76],
        "features_1": [5, 64, 1, 1],
        "features_2": [5, 64, 1, 1],
        "features_3": [5, 64, 1, 1],
        "features_4": [5, 64, 1, 1],
        "features_5": [5, 64, 1, 1],
        "features_6": [5, 64, 1, 1],
        "features_7": [5, 64, 1, 1],
        "rois": [150, 5],
        "index": [150]
      },
      "output_shapes": {"y": [150, 64, 7, 7]}
    },
    {
      "name": "dynamic_singlelevel_fp16_index",
      "params": {"finest_scale": 56, "roi_scale_factor": 0.0, "pooled_height": 7, "pooled_width": 7, "sample_num": 0, "pool_mode": "avg", "aligned": true},
      "input_shapes": {
        "features_0": [4, 256, 200, 336],
        "features_1": [4, 256, 1, 1],
        "features_2": [4, 256, 1, 1],
        "features_3": [4, 256, 1, 1],
        "features_4": [4, 256, 1, 1],
        "features_5": [4, 256, 1, 1],
        "features_6": [4, 256, 1, 1],
        "features_7": [4, 256, 1, 1],
        "rois": [1200, 5],
        "index": [1200]
      },
      "output_shapes": {"y": [1200, 256, 7, 7]}
    },
    {
      "name": "multilevel_n2_sample2",
      "params": {"finest_scale": 56, "roi_scale_factor": 0.0, "pooled_height": 7, "pooled_width": 7, "sample_num": 2, "pool_mode": "avg", "aligned": true},
      "input_shapes": {
        "features_0": [2, 32, 48, 48],
        "features_1": [2, 32, 24, 24],
        "features_2": [2, 32, 12, 12],
        "features_3": [2, 32, 6, 6],
        "features_4": [2, 32, 1, 1],
        "features_5": [2, 32, 1, 1],
        "features_6": [2, 32, 1, 1],
        "features_7": [2, 32, 1, 1],
        "rois": [64, 5],
        "index": [64]
      },
      "output_shapes": {"y": [64, 32, 7, 7]}
    }
  ],
  "perf_target": null
}
```

## 语义说明

### 1. 功能与分类

RoiExtractor 是检测网络 ROI head 的多层特征抽取算子（FPN RoI Extract / RoI Align）：对每个 RoI 先按其在**原图坐标系**下的面积开方值选择一个 FPN 层级，再将 RoI 坐标换算到该层特征图坐标系，做对齐式双线性采样并按采样数平均，输出每个 RoI 的池化特征。语义兼容 mmdetection `SingleRoIExtractor`（RoIAlignV2 行为）。

计算分类：gather（4 点取样）+ 双线性插值 + 平均池化的**纯 Vector 算子**（无 matmul / 无 Cube 依赖）。本任务为"仓内接入"型开发：ops-transformer 仓原本不存在该算子，语义唯一依据为 canndev 内置算子 RoiExtractor（TBE 动态实现），须保持算子语义不变；接入形态参照 `attention/dense_lightning_indexer_softmax_lse_v2`（PyPTO-Pro kernel + op_host C++ tiling/def + CMake 集成），目标芯片 Ascend950PR（a5，编译目标 ascend950）。

### 2. 公式符号与依据

| 符号 | 含义 | 依据 |
|---|---|---|
| `L` | 有效 FPN 层数（1~8），= 实际提供的 features_i 个数 | canndev `MAX_NUM_FEATURE_MAP=8`；op_proto `DYNAMIC_INPUT(features)` 1~8 层 |
| `N, C` | 各层公共的 batch 维与通道维（各层必须一致） | tbe/impl/dynamic/roi_extractor.py `feats_n_dim/feats_c1_dim` 单值假设 |
| `H_l, W_l` | 第 l 层特征图空间尺寸（各层可不同） | 同上 `feats_h_dims/feats_w_dims` |
| `R` | RoI 数（rois 第 0 维） | op_proto rois [num_rois, 5] |
| `rois[n, k]` | 第 n 行第 k 列：k=0 为 batch_idx，k=1..4 为 x0,y0,x1,y1（原图绝对坐标） | op_proto 注释与 TBE `_move_roi_data_to_ub` 列布局 |
| `s` | `fp16(sqrt((x1-x0)*(y1-y0)))`，层级选择用的面积尺度 | roi_extractor.py `map_roi_levels` L438-508：fp32 乘积 → 高精度 rsqrt+rec 求 sqrt → 转 fp16 |
| `B[i]` | `2^i * finest_scale`，i=0..L 的层级阈值（fp16 精确表示） | roi_extractor.py L92 `pow_table` |
| `spatial_scale[l]` | 第 l 层坐标换算乘法系数（原图坐标 × spatial_scale[l] = 该层特征图坐标） | op_proto attr 语义 + roi_align_compute.py `_calc_fm_start_end_coordinate` |
| `bin_w, bin_h` | 单个输出像素对应的 RoI 宽/高子区间 | `_calc_grid_size` |
| `sw, sh` | 每个输出像素内沿 W/H 的采样数 | `_calc_grid_size`：sample_num>0 用固定值，否则 ceil(bin) |
| `bilinear(fm, x, y)` | 带边界钳制的双线性插值，见公式 | roi_align_compute.py `_calc_bilinear_interpolate_coordinate` L581-696 |

依据置信度：接口与 attr 合同为 ✓ 高（op_proto/runtime 原文）；算法细节为 ⚠ 中（TBE 源码逐行解读，已对照伪代码复核，见 §11）。

### 3. 算法描述

公式字段给出逐 ROI 的六步算法，展开说明如下（编号与公式一致）：

**(1) 层级选择**——`s` 的乘积与开方在 fp32 中计算（fp16 rois 先转 fp32），结果按 round-to-nearest 转 fp16；`B[i] = 2^i * finest_scale` 在 fp16 中精确表示（默认 finest_scale=56 时 B=[56,112,224,448,896,...]）。比较在 **fp16** 中进行，规则：
- `s <= B[1]`（即 `s <= 2*finest_scale`）→ 层 0；
- `L >= 2` 且 `s > B[L-1]` → 层 L-1；
- 否则唯一满足 `B[l] < s <= B[l+1]` 的层 l（1 ≤ l ≤ L-2）。

**边界归属：s 恰等于 B[i] 时归入低层 i-1**（`s == B[1]` → 层 0，因层 0 条件为闭区间上界）。`L >= 2` 时任意有限 s 恰好映射一层（无丢弃）。**`L == 1` 时只有层 0 条件 `s <= 2*finest_scale` 生效，`s > 2*finest_scale` 的 RoI 不匹配任何层，输出行不写入**（保留 canndev 语义，见 §12 决策 3）。

**(2) 坐标换算**——先乘 `spatial_scale[l]`，`aligned=true` 时再对 x0,y0,x1,y1 四坐标**各减 0.5**（顺序：先乘后减，与 canndev `_calc_fm_start_end_coordinate` → `_calc_roi_size` 一致）。`roi_w = max(x1-x0, 1.0)`、`roi_h = max(y1-y0, 1.0)` 保护零面积/反转坐标。`bin_w = roi_w/pooled_width`、`bin_h = roi_h/pooled_height`（注意 bin_w 用 pooled_width——修复 canndev 动态版笔误，见 §12 决策 2）。

**(3) 采样数**——`sample_num > 0` 时 `sw = sh = sample_num`（固定采样）；否则自适应 `sw = ceil(bin_w)`、`sh = ceil(bin_h)`。

**(4) batch 守卫**——`b = floor(rois[n,0])`（fp32 floor 转 int）；仅 `0 <= b < N` 时计算该 RoI，否则输出行不写入（canndev `compute_one_roi` 的 `if fm_idx >= 0 and fm_idx < feature_n` 守卫）。

**(5) 采样与双线性**——采样点 `x = x0 + (pw + (j+0.5)/sw)*bin_w`、`yc = y0 + (ph + (i+0.5)/sh)*bin_h`（与 canndev `grid_distance*(grid_offset+grid_idx+0.5)` 展开等价）。双线性按公式定义执行边界处理：
- 原始坐标 `x < -1` 或 `x > W`（y 同理对 H）→ 该样本贡献 0（注意判据是 **-1 与 W**，不是 0 与 W-1：`x ∈ (-1, 0)` 会钳到 0 并取第 0 列，`x == W` 会钳到 W-1 并取最后一列）；
- 否则 `x = clamp(x, 0, W-1)`，`x_lo = min(floor(x), W-1)`，`x_hi = min(x_lo+1, W-1)`，`lx = x-x_lo`，`hx = 1-lx`（y 同理）；
- 四点加权 `fm[y_lo,x_lo]*hy*hx + fm[y_lo,x_hi]*hy*lx + fm[y_hi,x_lo]*ly*hx + fm[y_hi,x_hi]*ly*lx`。

**越界样本贡献 0 且仍计入平均分母 sh*sw**（canndev 将权重预除 sh/sw 后累加，数学上等价于固定分母平均；与 mmdet RoIAlign 一致，不是"跳过越界样本再平均"）。

**(6) 输出行映射**——提供 index 时输出行号 = `index[n]`，否则 = `n`；数值语义不变，仅改变写出行地址。每个 `(y_row, c, ph, pw)` 元素为覆盖写（该 RoI 行内累加完成后一次写出），无跨 RoI 累加。

canndev 实现细节差异（语义等价性说明，供 Stage 2/3 参考）：
- canndev 用 `rsqrt+rec` 高精度近似指令实现 sqrt（非 IEEE 精确舍入），与"fp32 精确 sqrt 后转 fp16"存在 ulp 级差异；仅在 s 恰位于 fp16 舍入边界时可能翻转层级。重实现与 golden 统一采用 **fp32 IEEE sqrt → fp16 round-to-nearest** 语义，测试数据构造避免 s 落在 fp16 舍入歧义区间（见 §10 数据 recipe，边界用例取整数面积使 sqrt 精确）；
- canndev fp16 模式将 spatial_scale 先转 fp16 再参与乘法；默认 scale 均为 2 的幂（0.25/0.125/0.0625/0.03125），fp16 表示精确，两种做法完全一致。重实现统一按 fp32 乘法（attr 本义即 fp32 列表）；非 2 的幂 scale 属 P2 场景不进验收门禁（见 §12 决策 9）。

### 4. 数据流说明

```
features_0..7 (NCHW, GM) ─┐
rois [R,5] (GM) ──────────┤→ 逐 RoI:
index [R] (int32, 可选) ──┘   ① 读 rois[n,:] → fp32 化
                              ② 层级选择（fp16 比较）→ l
                              ③ 坐标换算 → x0,y0,x1,y1 (fp32) → bin_w,bin_h → sw,sh
                              ④ batch 守卫 → b
                              ⑤ 逐 (ph,pw,i,j) 从 features_l[b] gather 4 点 → 双线性加权累加
                              ⑥ 除以 sh*sw → 写 y[index[n] 或 n, :, ph, pw]
y [R,C,pH,pW] (GM) ←──────────┘
```

- 特征图按需随机访问（每样本 4 个相邻点），无整图搬运要求；canndev 的 workspace 中转（roi_buf/roi_feats_buf）与 L1 整图缓存是其 tiling 策略的实现细节，重实现是否使用 workspace 由 Stage 3 决定（语义上无必需的辅助张量）。
- 层级选择在原图坐标上进行（与 spatial_scale 无关）；坐标换算才使用 spatial_scale[l]。
- 输出与输入无别名；除 index 提供重复值（未定义行为）外无写冲突。

### 5. 接口语义

**新算子公开接口（本 SPEC 的接口合同）**：

```
RoiExtractor(features_0..features_7 [OPTIONAL×8], rois [REQUIRED],
             index [OPTIONAL], y [OUTPUT];
             attr 顺序固定: finest_scale, roi_scale_factor, spatial_scale,
                            pooled_height, pooled_width, sample_num,
                            pool_mode, aligned)
```

**偏离声明（8-OPTIONAL 展开）**：canndev 原接口为 `DYNAMIC_INPUT(features)`（TensorList，1~8 层）。PyPTO-Pro 公开 API 无 int→ptr 转换，kernel 内无法解码 TensorList 描述符（C++ AscendC 专属能力，此为编排器已验证的框架约束），故新算子将 features 展开为 **8 个 OPTIONAL 输入 features_0..features_7**（对应 canndev `MAX_NUM_FEATURE_MAP=8`），层数 L 经 TilingData 传递。多层 FPN 特征提取**语义不变**；aclnn 接口形态由 `aclTensorList` 变为 8 个可选 `aclTensor*`。有效层必须构成 `features_0..features_{L-1}` 连续前缀（host 侧校验），空缺中间层为非法输入。

各输入语义：
- `features_i`：第 i 层特征图，逻辑 NCHW `[N, C, H_i, W_i]`，fp16/fp32；各有效层 N、C 一致，H/W 可不同。format ND（逻辑 NCHW；canndev 内部 5HD 布局是其实现细节，公开接口按逻辑 NCHW 语义）。
- `rois`：`[R, 5]` = `[batch_idx, x0, y0, x1, y1]`，**原图绝对坐标**，dtype 与 features 一致。
- `index`：int32 `[R]`，输出行映射：提供时输出行号 = `index[n]`，否则 = n 自身位置。长度必须等于 R（host 校验）；值域 [0,R) 由调用方保证（canndev 不校验，越界导致 GM 越界写，行为未定义）；重复映射行为未定义。
- `y`：`[R, C, pooled_height, pooled_width]`，dtype 与 features 一致。

**dtype 耦合规则**：features、rois、y 三者 dtype 必须一致，取值 float16 或 float32 二选一，经 `@pl.jit datatype` 机制特化（参考 dense_lightning `datatype={"query_index": "input_dtype"}` 模式；具体以哪个输入驱动特化由 Stage 3 决定）。index 恒为 int32。机器合同以 float16 为 canonical dtype（UT 主用例、精度路径更严格），**float32 为等价第二模式**：接口形状、公式、边界语义完全相同，仅值 dtype 不同（ST 主用例即 fp32，见 §10 case 2）。

attr 语义（顺序固定，默认值取自 canndev op_proto）：
| attr | 类型 | 默认 | 语义 |
|---|---|---|---|
| finest_scale | Int | 56 | 层级阈值基数：B[i] = 2^i * finest_scale |
| roi_scale_factor | Float | 0 | **接受但不参与计算**（canndev 参考实现同样未使用，保持一致） |
| spatial_scale | ListFloat | {0.25, 0.125, 0.0625, 0.03125} | 各层坐标换算乘法系数，长度 ≥ L，用前 L 项 |
| pooled_height | Int | 7 | 输出 H 维（bin_h 分母） |
| pooled_width | Int | 7 | 输出 W 维（bin_w 分母），与 pooled_height 独立 |
| sample_num | Int | 0 | 每输出像素每轴采样数；0 = 自适应 ceil(bin) |
| pool_mode | String | "avg" | 池化模式，仅支持 "avg"，其他值 host 报参数错误 |
| aligned | Bool | true | true 时坐标换算后四坐标各减 0.5 |

### 6. 功能与可选参数依据

| 特性/参数 | 需要 | 来源与置信度 | 优先级 | 影响 |
|---|---|---|---|---|
| 多层 FPN（L=1..8） | 是 | op_proto DYNAMIC_INPUT 1~8 层 ✓高 | P0 | 层级选择 + 逐层 spatial_scale |
| index 输出重映射 | 是（可选输入） | op_proto OPTIONAL_INPUT + TBE balance_compute L426-427 ⚠中 | P0 | 输出行地址映射，不改数值 |
| aligned（-0.5 对齐） | 是 | op_proto 默认 true + `_calc_roi_size` ⚠中 | P0 | 坐标换算 |
| sample_num 自适应/固定 | 是 | op_proto 默认 0 + `_calc_grid_size` ⚠中 | P0 | 采样数 |
| pool_mode="avg" | 是 | op_proto 默认 + TBE init 强校验 ✓高 | P0 | 仅 avg，非 avg 报错 |
| roi_scale_factor 保留不使用 | 是 | op_proto 默认 0；TBE 读入 tiling 但计算未用 ⚠中 | P1 | 接口兼容性（保持 attr 存在与顺序） |
| 独立 pooled_h/pooled_w | 是（接口契约） | op_proto + runtime infershape 独立使用两 attr ✓高；canndev 动态 TBE 有笔误 ⚠中 | P0 | 非方形输出；修复决策见 §12 |
| L=1 大 RoI 丢弃 | 是（保留语义） | TBE `where_and_nonzero` L549-556 ⚠中 | P1 | 边界语义（决策见 §12） |
| fp16/fp32 双 dtype 模式 | 是 | op_proto TensorType({DT_FLOAT16, DT_FLOAT}) ✓高 | P0 | datatype 特化 |
| 动态 shape（R/N/C/H/W/L 运行时可变） | 是 | canndev 动态实现 + ST 动态用例 ✓高 | P0 | TilingData 传递全部维度 |

optional 输入 index 四项分析：公式位置=第(6)步输出行映射；计算作用=纯地址重映射（数值不变）；依赖/互斥=仅当提供时生效，无互斥参数，长度须等于 R；省略时默认语义=输出行号 = roi 自身位置 n（恒等映射）。

### 7. 精度语义

- **验收容差**：atol=0.005、rtol=0.005（来源：canndev UT `PrecisionStandard(0.005, 0.005)`，test_roi_extractor_impl.py L71）。比对基准为 CPU FP32 golden（Stage 2 产出）；don't-care 行（见 §9）排除比对。
- **精度路径**（canndev 已验证路径，重实现必须遵守的语义 dtype 链）：
  - 坐标链全程 **fp32**：fp16 rois/features 读入后先转 fp32；spatial_scale 乘法、-0.5、roi_w/h、bin、采样点、floor、lx/ly 全部 fp32；
  - 层级选择在 **fp16**：fp32 sqrt 结果转 fp16 后与 fp16 阈值比较；
  - fp32 模式：权重、乘加、累加全程 fp32，输出 fp32；
  - fp16 模式：双线性权重允许 fp32 计算后转 fp16 用于乘加（canndev 行为：权重预除 sh/sw 后 fp16 乘加累加）；重实现推荐 fp32 累加后单次舍入到 fp16（精度更优），两种路径均须通过 0.005/0.005 容差——累加中间 dtype 属实现选择，由 Stage 3 在容差约束下决定（见 kernel 契约补充）。
- **除法语义**：`bin_w = roi_w/pooled_width` 等除法按 fp32 除法（或乘以 host 预计算倒数，canndev 用 pooled_w_reciprocal——倒数与除法在容差内等价，Stage 3 决定）。
- **sqrt 语义**：统一 fp32 IEEE sqrt → fp16 round-to-nearest（canndev 的 rsqrt+rec 近似视为 ulp 级等价；测试数据避开歧义区间，见 §3）。

### 8. 动态 Shape 与约束依据

- `R ∈ [1, 10000]`（机器合同可具体化区间）：检测框架单次前向 RoI 数（mmdet max_per_img ≤ 1000，多 batch 放大一个量级）；canndev UT/ST 用例 1000/150/1200。**R=0 合法**（空 RoI，输出 [0,C,pH,pW]，不写数据），机器合同因正数维限制无法具体化，见 §9/§10。
- `N ∈ [1, 128]`：特征图 batch（用例 1/4/5）。
- `C ∈ [1, 4096]`：FPN 通道包络（用例 16/64/256，检测网络 ≤ 4096）。
- `H_i, W_i ∈ [1, 8192]`：特征图空间尺寸（动态 ST 200×336）。canndev 动态实现受"fh*fw ≤ 8192"的整图搬运约束（roi_align_compute.py L1192 注释），重实现按需 gather 无此约束，语义上解除该限制（8192 单维包络）。
- `L ∈ [1, 8]`：canndev MAX_NUM_FEATURE_MAP=8；L=8 时 spatial_scale 须显式传 ≥8 项（默认仅 4 项）。
- 层间约束：各有效层 N、C 一致（host 校验）；H/W 独立。
- pooled_height/pooled_width ∈ [1,1024]、sample_num ∈ [0,1024]、finest_scale ∈ [1,512]（2^(L-1)*finest_scale ≤ 65504 保证全部阈值 fp16 精确，L=8 时上界即 512）：支持包络，超界 host 报参数错误。

### 9. 边界条件处理

| 边界场景 | 数学行为 | 比对处理 |
|---|---|---|
| R=0（空 RoI） | y=[0,C,pH,pW]，kernel 不写数据，正常返回 | 输出为空，无数据可比 |
| 零面积 RoI（x1==x0 或 y1==y0） | roi_w/roi_h 经 max(Δ,1)=1 保护，正常计算 | 正常比对 |
| 反转坐标（x1<x0） | 同上 max(Δ,1)；若仅单维反转则面积积为负 → sqrt 负数 → NaN → 层级不匹配 → 行不写入 | don't-care 行 |
| 贴边 RoI（坐标在特征图边界） | 双线性钳制语义（x==W 取最后一列等） | 正常比对 |
| 部分越界坐标（x<-1 或 x>W 等） | 该样本贡献 0，分母不变；x∈(-1,0) 钳到 0 列 | 正常比对 |
| s 恰等于 B[i] | 归属低层 i-1（闭区间上界） | 正常比对（数据构造使 sqrt 精确） |
| L==1 且 s>2*finest_scale | 输出行不写入（保留 canndev 丢弃语义） | don't-care 行 |
| batch 索引越界（b<0 或 b≥N） | 输出行不写入 | don't-care 行 |
| index 值越界 [0,R) | 未定义（canndev 会 GM 越界写；调用方合同保证合法） | 不构造 |
| index 值重复 | 未定义（写冲突） | 不构造 |
| rois 含 NaN/Inf | 未定义（canndev：NaN 面积→层级不匹配→行不写入；NaN 坐标→硬件 max/min 传播） | 不构造（值域声明为有限值） |
| sample_num=0 且 bin<1 | ceil 后 sw=sh=1，正常计算 | 正常比对 |

don't-care 行约定：CPU golden 对这些行填 NaN，比对时跳过 NaN 行；NPU 侧输出这些行内容未定义（不应读取）。

### 10. P0 与性能目标依据

**机器合同 p0_case 执行规格**（形状见机器合同；dtype 与数据构造在此定义，golden 生成器按此执行）：

| case | 执行 dtype | index | rois 数据 recipe（确定性要求） |
|---|---|---|---|
| ut_multilevel_fp16 | fp16 | 缺席 | 原图 64px（features_0 16×16 ÷ 0.25）。行 0..959：随机正常 rois，s 值覆盖全部 4 个层级区间（(0,112], (112,224], (224,448], >448）每区间 ≥200 个；行 960..969：贴边（触及 0 或 63 边界）；行 970..979：零面积（x1==x0 及 y1==y0 两类）；行 980..989：部分越界（负坐标、超 63 坐标）；行 990..994：s 恰在边界——构造整数面积 w×h = 112²=12544（→s=112 归层 0）、224²=50176（→s=224 归层 1）、448²=200704（→s=448 归层 2）各 ≥1 个（允许坐标越界，层级选择与图幅无关）；行 995..999：batch 越界（b=-1、b=1，N=1）→ don't-care |
| st_singlelevel_fp32 | fp32 | 缺席 | 原图 304px（76÷0.25）。150 行随机正常 rois + 少量贴边/越界（坐标范围 [0,304) 为主） |
| dynamic_singlelevel_fp16_index | fp16 | **提供**，取 [0,1200) 的非恒等排列（如 index[n]=1199-n） | 原图 800×1344（200/0.25, 336/0.25）。约 80% 行 s≤112（正常计算），约 20% 行 s>112（L=1 丢弃 → don't-care，验证丢弃语义与 index 重映射共存） |
| multilevel_n2_sample2 | fp16 | 缺席 | 原图 192px（48÷0.25）。64 行随机正常 rois，跨 batch_idx 0/1，s 覆盖 4 层；sample_num=2 固定采样（aligned 走默认 true 路径，aligned=false 场景见下方场景矩阵 P1） |

**P0 场景矩阵**（机器 case 之外的 P0 验收场景，Stage 2 golden 与 Stage 4 JIT 验证必须覆盖）：

| # | 场景 | 规格 | 预期 |
|---|---|---|---|
| s1 | 空 RoI | features 同 ut_multilevel_fp16，rois [0,5] | y [0,16,7,7]，不写数据，正常返回 |
| s2 | 非方形 pooled | features 同 ut_multilevel_fp16，rois [256,5]，pooled_height=4, pooled_width=8 | y [256,16,4,8]，独立 pH/pW 正确（锁定 §12 决策 2） |
| s3 | L=1 大 RoI 丢弃 | features_0 [1,16,32,32] fp16，rois [16,5]（8 行 s≤112、8 行 s>112） | s>112 行 don't-care，其余行正常比对 |
| s4 | index 非恒等重映射 | 机器 case 3 覆盖（reverse 排列） | 输出行按 index 重排 |
| s5 | 边界/零面积/越界/s 边界/batch 越界 | 机器 case 1 数据 recipe 覆盖 | 正常比对 + don't-care 行 |
| s6 | 8 层特征 | features [1,16,256,256]→[1,16,128,128]→…→[1,16,4,4]（8 级 2 倍金字塔），spatial_scale 8 项（0.25 起逐级减半），rois [512,5]，fp16 | 正常比对 |

**P1 场景**（建议覆盖，不阻塞交付）：fp32 多层（case 1 形状 fp32 执行）、pooled 14×14、aligned=false + sample_num=0 组合、finest_scale=224（边界整体移动）。**验收规格（P0 硬门禁）**：
1. **编译产物**：`bash build.sh --pkg --soc=ascend950 --ops=roi_extractor` 实际编译成功，产出 kernel `.o` 与最终算子二进制（run 包内 binary），op_host def/infershape/tiling 与 op_graph proto 注册齐全；
2. **JIT 精度测试**：机器 case 1-4 + 场景 s1-s6 在 NPU（TILE_FWK_DEVICE_ID=6，Ascend950PR）上以 `@pl.jit` 编译执行，与 CPU FP32 golden 对齐，容差 0.005/0.005（don't-care 行排除）；
3. dtype 覆盖：fp16 与 fp32 两模式均被执行（case 1/3/4 fp16，case 2 fp32）。

**性能目标**：`perf_target: null`——用户未给出数值性能目标，无 Stage 5 性能指标硬门禁（仅正确性交付）。按无人值守默认规则披露：若进入 Stage 5，将采用逐 P0 case 的默认理想参考 `golden_reference_ratio >= 1.0`，该值为系统默认而非用户要求。

### 11. 参考与来源

| 来源 | 用途 | 置信度 |
|---|---|---|
| `/data/w00938615/code_repo/canndev/ops/built-in/op_proto/inc/nn_detect_ops.h` L1973-2013 | 接口、attr 顺序与默认值 | ✓ 高 |
| `/data/w00938615/code_repo/canndev/ops/built-in/op_proto/runtime/roi_extractor.cc` L42-84 | 输出 shape 推导（独立 pooled_height/pooled_width） | ✓ 高 |
| `/data/w00938615/code_repo/canndev/ops/built-in/tbe/impl/dynamic/roi_extractor.py` | 层级选择（map_roi_levels L438-508、where_and_nonzero L528-568、pow_table L92）、index 重映射（balance_compute L426-427）、fp16 阈值比较 | ⚠ 中（逐行解读） |
| `/data/w00938615/code_repo/canndev/ops/built-in/tbe/impl/dynamic/roi_align_compute.py` | 坐标换算/采样/双线性/平均（L452-507、L581-696）；**L72 `self.pool_w = op.pooled_h` 笔误证据**；权重预除 sh/sw（L689-696） | ⚠ 中（逐行解读） |
| `canndev/ops/built-in/tests/ut/ops_test/RoiExtractor/test_roi_extractor_impl.py` | 0.005/0.005 精度标准；4 层 fp16 用例 | ✓ 高 |
| `canndev/ops/built-in/tests/st/RoiExtractor/*.json` | 单层 fp32（150 rois）、动态 fp16+index（1200 rois）用例 | ✓ 高 |
| 编排器任务书技术约束（8-OPTIONAL 展开、TilingData/TilingKey 一致性机制、CMake 落位、Ascend950PR） | 已验证框架约束与落位合同 | ✓ 高（已批准合同） |
| `attention/dense_lightning_indexer_softmax_lse_v2/`（仓内参照实现） | 接入形态模板（@pl.jit、datatype 特化、op_host/CMake 结构） | ✓ 高 |

### 12. 自动决策

无人值守模式（编排器代用户授权 + 任务书已给完整需求），非阻塞字段的决策如下（均可在评审时回溯）：

1. **8-OPTIONAL 展开**（采用）：features 声明为 8 个 OPTIONAL 输入。来源：编排器已验证的 PyPTO-Pro 框架约束（无 TensorList 描述符解码能力）。影响：aclnn 形态由 aclTensorList 变 8×可选 aclTensor*；语义不变；层数经 TilingData 传递。
2. **独立 pooled_height/pooled_width**（采用，修复 canndev 笔误）：bin_w 用 pooled_width、bin_h 用 pooled_height、输出 [R,C,pH,pW]。依据：op_proto 与 runtime infershape 的接口契约独立使用两 attr（✓高）；canndev 动态 TBE `roi_align_compute.py` L72 `pool_w = op.pooled_h` 为实现笔误（pooled_h≠pooled_w 时输出形状正确但计算错误）。影响：非方形 pooled 从"错误输出"变为"按契约正确"；用 s2 场景锁定。
3. **L=1 大 RoI 丢弃语义**（采用，保留 canndev 行为）：L==1 且 s>2*finest_scale 时输出行不写入（内容未定义）。理由：canndev 是唯一语义依据，其 L=1 分支即不写入（where_and_nonzero 仅 fm_idx==0 有 `s<=B[1]` 条件）；改写为零值或钳制到层 0 都是无依据的语义变更。影响：测试以 don't-care 行处理，场景 s3 显式验证触发路径。
4. **层级边界归属**（采用，保留 canndev 行为）：s 恰等于 B[i] 时归入低层 i-1（闭区间上界语义）。依据：层 0 条件为 `s <= B[1]`、中间层为 `B[l] < s <= B[l+1]`。
5. **canonical dtype 取 float16**（采用）：机器合同以 fp16 为 canonical、int32 为 index dtype；float32 为等价第二模式（公式/接口/边界完全相同），在 §5/§7/§8 与 kernel 契约补充中显式记录。理由：SPEC 机器合同的 dtype 字段为单一值（验证器限制）；fp16 为 UT 主用例且精度路径更严格。影响：Stage 2 golden 须按 §10 的执行 dtype 表为 case 2 生成 fp32 数据；Stage 3 须实现 fp16/fp32 双特化。
6. **机器合同哨兵约定**（采用）：features_i 缺席 ⇔ p0_case 中形状 [N,C,1,1]（真实层不用 1×1）；index 形状 [R] 为占位、case 名 `_index` 后缀表示实际提供；pooled 维按默认 7 具体化（非默认场景进 §10 场景矩阵）。理由：验证器要求全量输入正数维且 rank 一致，无法表达缺席/空维/参数化输出维；约定在 shape_constraints 中机器可查，无歧义。
7. **roi_scale_factor 不参与计算**（采用，与 canndev 一致）：attr 保留（接口兼容），计算忽略。
8. **sqrt 语义统一为 fp32 IEEE sqrt → fp16 舍入**（采用）：canndev 的 rsqrt+rec 近似视为 ulp 级等价；边界测试数据取整数面积使 sqrt 精确，规避舍入歧义。
9. **spatial_scale 按 fp32 乘法**（采用）：canndev fp16 模式先将 scale 转 fp16 的怪癖仅对非 2 的幂 scale 产生 ulp 级差异；默认值与全部 P0/P1 用例均为 2 的幂（精确一致）。非 2 的幂 scale 属 P2，不进验收门禁。
10. **数值包络**（采用）：features/rois/y 取 fp16 有限值域 [-65504,65504] 作合同包络（UT 实际生成 [-10,10]）；R≤10000、C≤4096、H/W≤8192、pooled≤1024、sample_num≤1024、finest_scale≤512（fp16 阈值可表示性）。超包络 host 报参数错误（对 finest_scale 为新增保护性校验，防止静默退化）。
11. **解除 canndev fh*fw≤8192 空间限制**（采用）：重实现按需 gather 4 点，无整图搬运需求；单维上限取 8192 包络。落点由 Stage 3 设计确认。

## kernel 契约补充

| 字段 | 裁定 |
|---|---|
| **辅助张量语义** | 无模型参数、无公开辅助张量。index 为公开可选输入（纯输出行重映射）。canndev 的 workspace 中转（roi_buf/roi_feats_buf）与 L1 整图缓存是其 tiling 策略实现细节，非语义必需；重实现是否申请 workspace 由 Stage 3 决定。无 dense_lightning 式 metadata 输入（分核/循环参数走 TilingData）。 |
| **cast 边界链** | 输入（fp16 模式）：rois/features fp16 → 坐标链入口转 fp32；spatial_scale 乘法、-0.5、roi/bin/采样点/floor/lx-ly 全程 fp32。层级链：fp32 乘积与 sqrt → fp16（round-to-nearest）→ fp16 阈值比较。权重：fp32 计算（fp16 模式允许转 fp16 用于乘加）。累加：fp32（推荐）或 fp16（canndev 行为，权重预除 sh/sw）。输出：单次舍入到 features dtype（fp16/fp32）。fp32 模式全程 fp32。具体 cast 指令选型归 Stage 3，验收以 0.005/0.005 判定。 |
| **累加/写回语义** | 每个 (y_row, c, ph, pw) 元素 = 该 RoI 行内 sh*sw 个样本累加后**覆盖写**（一次写出）；无跨 RoI 累加、无原子操作、无初始化要求（don't-care 行不写）。分母恒为 sh*sw（越界样本计 0）。index 重映射仅改目标行地址。 |
| **目标设备** | Ascend950PR（a5，编译目标 ascend950）——任务书已指定，非默认假设。JIT 验证用 TILE_FWK_DEVICE_ID=6（device 0 有告警勿用）。op_host 配置目录 `op_host/config/ascend950/`。 |
| **topology/tile/同步** | 由 Stage 3 设计，本 SPEC 不猜测。供参考的客观事实：纯 Vector 算子（无 Cube/matmul）；数据并行维度 = RoI × 通道 × 输出像素；每样本 4 点 gather + 双线性加权；无跨核数据依赖（index 重复映射属未定义输入，不设计保护）。 |

**目标仓落位与目录结构**（仓内接入合同，Stage 3/4 遵守）：

```
detection/                                    # 新建顶层模块（仿 attention/）
├── CMakeLists.txt                            # 仿 attention/CMakeLists.txt（子目录遍历 + ASCEND_OP_NAME 过滤）
└── roi_extractor/
    ├── CMakeLists.txt                        # 仿 dense_lightning 模块级 CMakeLists（子目录遍历）
    ├── op_host/
    │   ├── CMakeLists.txt                    # require_pypto_pro(roi_extractor) + enable_pypto_kernel(roi_extractor)
    │   │                                     #   （BUILD_OPEN_PROJECT 分支内、add_modules_sources 之前）+ add_op_to_compiled_list
    │   │                                     #   + Ascend950PR_9599 条件编译项 + --cce-auto-sync=off（仿 dense_lightning op_host/CMakeLists.txt）
    │   ├── roi_extractor_def.cpp             # OpDef：8×OPTIONAL features + rois + OPTIONAL index + y + 8 ATTR（顺序固定）；ascend950 AICore config
    │   ├── roi_extractor_infershape.cpp      # y = [R, C, pooled_height, pooled_width]（独立两 attr）
    │   ├── roi_extractor_tiling.cpp          # TilingData 填充（含 L、各层 H/W、spatial_scale）+ SetTilingKey 位打包 + workspace 规划
    │   ├── roi_extractor_tiling.h            # TILING_DATA_FIELD_DEF 结构 + REGISTER_TILING_DATA_CLASS（字段与 .py dataclass 完全一致）
    │   └── config/ascend950/
    │       ├── roi_extractor_binary.json     # 二进制登记（构建产物）
    │       └── roi_extractor_simplified_key.ini
    ├── op_kernel/
    │   └── roi_extractor.py                  # @pl.jit kernel：@dataclass Tiling + TilingKeyField + datatype 特化 + make_tensor/pl.make_tile_group
    ├── op_graph/
    │   └── roi_extractor_proto.h             # REG_OP 原型（op_host def 的 graph 侧镜像，8 OPTIONAL + attr 顺序一致）
    ├── examples/
    │   └── test_aclnn_roi_extractor.cpp      # aclnn 调用示例
    └── README.md
```

根集成（三处）：① 根 `CMakeLists.txt` L57 `ASCEND_ALL_MODULE_NAME` 追加 `detection`；② `cmake/custom_build.cmake` 模块块（仿 L292-295 attention 块）增加 `should_add_module("detection" ADD_DETECTION)` + `add_subdirectory(detection)`；③ 编译入口 `bash build.sh --pkg --soc=ascend950 --ops=roi_extractor`。

**Tiling/TilingKey 硬约束**（编排器已验证，Stage 3 必须遵守）：
- Host 侧 TilingData（roi_extractor_tiling.h 的 TILING_DATA_FIELD_DEF）必须与 op_kernel/roi_extractor.py 中 `@dataclass` Tiling 类字段**顺序/类型/对齐完全一致**（构建系统 force-include pypto codegen 生成的 `roi_extractor_tiling.h`）；
- TilingKey 由 kernel 侧 `@pl.jit tiling_key=<Class>`（TilingKeyField 位域）与 host 侧 `roi_extractor_tiling.cpp` 的 `context->SetTilingKey()` 手工位打包保持一致——打包的是 values 列表中的**索引值左移 offset**，字段按类定义顺序分配位（参考 dense_lightning tiling.cpp L437-440 与 DenseLiLseTilingKey）；
- dtype 特化经 `@pl.jit datatype={"<某输入>": "input_dtype"}` 机制（参考 dense_lightning `datatype={"query_index": "input_dtype"}`）；候选 TilingKey 维度（dtype fp16/fp32、L、has_index 等）由 Stage 3 设计。

---
*生成时间: 2026-09-04*
*确认状态: 已确认（编排器代用户授权的无人值守模式；接口/算法/精度/验收均已按任务书与 canndev 源码冻结，非阻塞决策见 §12 自动决策）*
