# pdist TTK 三通路测试结果（67 例泛化用例集）

- 日期: 2026-09-11（2026-09-07 首轮）
- 设备: Ascend 950PR × 8 卡（TTK 自动分卡）
- 算子包: ops-transformer build_out（custom_transformer vendor，含 pypto MIX_AIC_1_2 二进制；aclnn 入口复用 CANN 内置 aclnnPdist）
- golden: `../assets/golden.py`（TestSpec：对齐 shape 走 numpy 分块，非对齐走 torch 小算子拼接；NaN 语义对齐 GPU/CUDA——p=0 NaN 维度计为不相等，p=inf 忽略 NaN 维度、全 NaN 行对输出 0）
- 容差: 元素规则 rtol=atol=1e-3（同 mix_tolerance），全局绝对上限经 TestSpec compare 钩子按输出幅值缩放 `max(固定值, 2×ulp(max|golden|))`（fp16 固定 2.0 / fp32 固定 1.0），覆盖大幅值输出处 1~2 个出口舍入 ulp
- 用例: `../st/arch35/ttk_kernel_pdist_st.csv`（kernel/geir 共用）、`ttk_aclnn_pdist_st.csv`

## 结果

| 通路 | 命令 | 结果 |
|---|---|---|
| kernel（直连 launch） | `ttk kernel -i ttk_kernel_pdist_st.csv --plugin ../assets/` | **67/67 PASS** |
| aclnn（引擎 API） | `ttk aclnn -i ttk_aclnn_pdist_st.csv --plugin ../assets/` | **67/67 PASS** |
| geir（GE 图，二进制复用） | `ttk geir -i ttk_kernel_pdist_st.csv -b release --plugin ../assets/` | **67/67 PASS**（wrapper 回退 0 次） |

注：kernel 通路在线编译（DYN）为共享 kernel_meta 的并发 make，清缓存后全量重编时偶发
`pypto compile failed (make); missing objs` 资源竞争（与精度无关，复跑即过）。

## 用例覆盖（67 例）

1. PBranch 全覆盖：p=0/0.5/1/2/3/inf 精确链 + p=1.5/7.0 general exp/ln 链
2. tile 边界：最小 shape、N=63/65/129/193（jseg/VP 尾）、M=127/129/256（M 尾）、M=1、双轴尾、multitile、partial、M=8192/65535、两个 perf 形态
3. dtype：fp16 × 各 p 分支 + 宽幅值域
4. 数据形态：全零/固定值/必现 0/必现 7/纯负/宽幅/极窄幅/微小值/NaN × 4 分支
5. 随机采样 8 例（seed=20260907 固定可复现）
6. 对齐交叉 2×2（M%128 × (N-1)%64）
7. 大N/大M 组合 4 例

## 精度结论

- 逐元素 rtol=atol=1e-3 全例 100% 达标（matched_ratio=1.0）
- NaN 语义对齐 GPU/CUDA（本轮 2026-09-11 生效）：p=0 NaN 维度计为不相等（有限计数）；p=inf 忽略 NaN 维度（部分 NaN → 非 NaN 最大差，全 NaN 行对 → 0）；p∈{0.5,1,2} 与一般 p NaN 自然传播/显式输出 NaN
- p=0 与 p=inf 为精确运算（计数/比较 max），kernel 与 golden 逐 bit 一致
- 其余分支 max_abs_error 为输出量级 × 输出 dtype ulp 包络（fp16 出口 CAST_RINT 1 ulp；fp32 累加 ~8 ulp；p=0.5 相对 ~6e-6）
- 三通路误差特征一致（逐例差异源于各通路独立随机输入，非系统性偏差）

## 复跑前提

```bash
source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
export ASCEND_CUSTOM_OPP_PATH=<vendor 包路径>          # kernel/geir 必需
# aclnn 入口为 CANN 内置 libopapi.so（setenv 后默认可见），无需追加 vendor op_api/lib
cd ops-test-kit
python3 -m ttk kernel -i <csv> --plugin <assets>       # kernel
python3 -m ttk aclnn  -i <csv> --plugin <assets>       # aclnn
python3 -m ttk geir   -i <csv> -b release --plugin <assets>  # geir 必须二进制复用
# 若曾跑过 geir 在线模式: 先清 atc_data/kernel_cache/Ascend*/te_<op>_*（防 JIT 缓存静默回退）
```

## 文件

- `ttk_kernel_pdist_result.csv` / `ttk_aclnn_pdist_result.csv` / `ttk_geir_pdist_result.csv`：逐例结果（含 precision_metrics 明细）
