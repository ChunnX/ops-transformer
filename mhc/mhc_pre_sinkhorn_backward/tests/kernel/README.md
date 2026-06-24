# MhcPreSinkhornBackward — Kernel 模式精度测试 (TTK)

基于 ops-test-kit (TTK) kernel 模式，对 `mhc_pre_sinkhorn_backward` 做上板精度验证，
覆盖 deterministic 模板（bf16 / fp16 两套 dtype），1000 条用例。

## 文件
- `mhc_pre_sinkhorn_backward_kernel_cases.csv` — 1000 条用例（B=1, S=bs, n=4, iters=20）。
  - bs 覆盖 1~16384；d≤7168（重点 4096 / 7168）；bf16 / fp16 各 500 条。
  - `attributes` 含 `'#enable_deterministic_mode':1`，走 deterministic 模板。
  - 容差：fp32 (rtol,ptol)=(1e-4,1e-4)、fp16=(1e-3,1e-3)、bf16=(5e-3,5e-3)，atol=1e-5。
- `golden_kernel.py` — kernel 级 numpy/torch golden 插件（`__golden__` 注册），
  数学参考 `../assets/golden_src.py`，SinkhornGrad 适配算子 sum_out/norm_out 的 (2*iters,B,S,n[,n]) 布局。
- `gen_cases.py` — 用例生成脚本。

## 运行
```bash
cd <ops-test-kit>
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
python3 -m ttk kernel \
  -i <this_dir>/mhc_pre_sinkhorn_backward_kernel_cases.csv \
  --plugin <this_dir>/golden_kernel.py \
  -b release --pc 2 -o results.csv
```
前置：先用 `bash build.sh --pkg --soc=ascend950 --ops=mhc_pre_sinkhorn_backward` 编译并安装算子包。

## 本次修复的源码问题
1. arch35 base tiling `SetDynUBufSize(ubSize_)` 用满 UB(253952B)，MIX(AIC:AIV=1:2) 走 SIMT
   动态共享内存，超过 950 的 229376B 上限 → kernel 拉起报 ACL_ERROR_RT_PARAM_INVALID。
   已 clamp 到 SIMT 可用预算 221184B。
2. grad 矩阵乘默认开 HF32（`SetHF32(true,1)`），fp32 输出 grad_phi 达不到 1e-4 标准。
   one_stage / deterministic 两条路径均改为 `SetHF32(false,0)`，走全 fp32 matmul。
