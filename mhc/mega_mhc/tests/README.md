# mega_mhc 验证

## 无设备测试

```sh
python -m unittest discover -s mhc/mega_mhc/tests -p test_torch_contract.py -v
g++ -std=c++17 -O2 mhc/mega_mhc/tests/host/test_cube_plan.cpp \
  mhc/mega_mhc/op_host/op_tiling/arch35/mega_mhc_cube_plan.cpp -o /tmp/mega_mhc_plan_test
/tmp/mega_mhc_plan_test
```

Python 合约测试直接执行生产 validator/Meta 函数的 AST，以 metadata 对象代替 Tensor；
不代表真实 Torch dispatcher/JIT 验证。Host 测试直接调用生产 BuildCubePlan，覆盖9500组
M/D/核数/投影模式/四五模板，另增加836组ND/BS分核检查，检查 M/K 覆盖、partial 边界、尾块、L1/L0容量及两种 producer UB。

## 950 验收

先构建并安装算子包、加载对应厂商环境，再安装当前仓库的 Torch 扩展。

```sh
python -m unittest discover -s mhc/mega_mhc/tests -p test_mega_mhc_npu.py -v
```

真实调用 `cann_ops_transformer.mega_mhc`，对照独立非融合参考公式，保留阶段舍入点，
检查固定配置重复调用逐位一致。没有 CANN/Torch NPU 时测试明确 skip。
阈值见测试实现；A5-222/CANN-0830、device 6 的分批验收结果见通用分块实验记录。
以下为测试矩阵定义，不表示全量已通过；五算子既有精度例外仍以实验记录为准。

当前数值测试矩阵为672组：FP16/BF16 × D2560/5120/6144/7168 × 四/五算子 × FP32/HF32 ×
M=1/15/16/17/31/32/33/64/128/160/192/512/768/1024/1536/1537/2048/3072/3073/4096及[B,S]=[2,17]。M=3073覆盖通用分块的尾块。
另有64组零值、小/大幅值与epsilon测试。
通过 `MEGA_MHC_NPU_DEVICE` 选择设备，默认1；选算子构建为自定义包时，设置
`MEGA_MHC_TORCH_PACKAGE=cann_ops_transformer_custom`。请从源码 `torch_extension` 目录之外启动，
避免当前目录的打包中间产物遮蔽安装包。

A5-222本轮测试使用 `workspace/setenv.sh`、`workspace/CANN-0830/cann-9.2.0/set_env.sh`
及 `workspace/megamhc/install/vendors/custom_transformer/bin/set_env.bash`，先加载环境再启用shell严格模式。
系统Python缺少开发头文件；JIT使用机器已有的 `/opt/conda/envs/py311/include/python3.11`（通过CPATH）。
测试源码、安装包和日志均隔离在 `/home/l00947498/workspace/megamhc`。

当前默认 GM 路径在 device 6、32 AIC/64 AIV 下通过上述176组数值/确定性与64组边界配置。
目标 M=3072、D=5120 的四/五算子、FP16/BF16、FP32/HF32 八种组合也已单独验证。
实际使用前仍需确认所选设备空闲且健康；历史设备启动异常不代表当前卡的状态。

```sh
MEGA_MHC_NPU_DEVICE=6 python -m unittest discover \
  -s mhc/mega_mhc/tests -p test_mega_mhc_npu.py -v
```

`MEGA_MHC_TEST_AIC_LIMIT` 仅用于测试进程资源配额，不是默认实现的一部分。
UB→L1 等非默认传输路径需要独立验收，不能从 GM 结果推定。

扩展 D 仍存在未通过精度阈值的案例，上述历史验收结果不代表新增矩阵全部通过。

通过 `MEGA_MHC_TEST_SCHEDULE_MODE=0/1/2` 分别对 M-K/ND/BS 运行完整数值矩阵。
ND/BS 仅验证了部分场景，仍存在精度验收例外；定义测试矩阵不代表全量已通过。
