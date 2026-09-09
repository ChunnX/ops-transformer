# pdist TTK 三通路测试结果（67 例泛化用例集）

- 日期: 2026-09-07
- 设备: Ascend 950PR × 8 卡（TTK 自动分卡）
- 算子包: ops-transformer build_out（custom_transformer vendor，含 pypto MIX_AIC_1_2 二进制）
- golden: `../assets/golden.py`（TestSpec：对齐走 numpy 分块，非对齐走 torch 小算子拼接；容差 mix_tolerance rtol=atol=1e-3，max_abs_error_limit fp32=1.0 / fp16=2.0）
- 用例: `../st/arch35/ttk_kernel_pdist_st.csv`（kernel/geir 共用）、`ttk_aclnn_pdist_st.csv`

## 结果

| 通路 | 命令 | 结果 |
|---|---|---|
| kernel（直连 launch） | `ttk kernel -i ttk_kernel_pdist_st.csv --plugin ../assets/` | **67/67 PASS** |
| aclnn（引擎 API） | `ttk aclnn -i ttk_aclnn_pdist_st.csv --plugin ../assets/` | **67/67 PASS** |
| geir（GE 图，二进制复用） | `ttk geir -i ttk_kernel_pdist_st.csv -b release --plugin ../assets/` | **67/67 PASS**（wrapper 回退 0 次） |

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
- 20/67 例逐 bit 精确（p=0/inf、NaN 语义、重复行、微型 shape 等精确分支）
- max_abs_error 全部为输出量级 × 输出 dtype ulp 包络（fp16 出口 CAST_RINT 1 ulp；fp32 累加 ~8 ulp；p=0.5 相对 ~6e-6）
- 三通路误差特征一致（逐例差异源于各通路独立随机输入，非系统性偏差）

## 复跑前提

```bash
source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
export ASCEND_CUSTOM_OPP_PATH=<vendor 包路径>          # kernel/geir 必需
export LD_LIBRARY_PATH=<vendor>/op_api/lib:$LD_LIBRARY_PATH  # aclnn 需追加
cd ops-test-kit
python3 -m ttk kernel -i <csv> --plugin <assets>       # kernel
python3 -m ttk aclnn  -i <csv> --plugin <assets>       # aclnn
python3 -m ttk geir   -i <csv> -b release --plugin <assets>  # geir 必须二进制复用
# 若曾跑过 geir 在线模式: 先清 atc_data/kernel_cache/Ascend*/te_<op>_*（防 JIT 缓存静默回退）
```

## 文件

- `ttk_kernel_pdist_result.csv` / `ttk_aclnn_pdist_result.csv` / `ttk_geir_pdist_result.csv`：逐例结果（含 precision_metrics 明细）
