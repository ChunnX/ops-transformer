# SoC 类型映射表

## torch_npu 显示名称与编译参数对照

| torch_npu 显示名称 | 编译参数 `--soc=` | 硬件型号 | 说明 |
|-------------------|------------------|---------|------|
| Ascend910_93 | `ascend910_93` | Atlas A3 | 训练/推理 |
| Ascend910_9362 | `ascend910_93` | Atlas A3 变体 | 训练/推理 |
| Ascend910B1 | `ascend910b` | Atlas A2 | 训练/推理 |
| Ascend910_910B | `ascend910b` | Atlas A2 变体 | 训练/推理 |
| Ascend910_91 | `ascend910b` | Atlas A2 变体 | 训练/推理 |
| Ascend950PR | `ascend950` | Ascend 950PR | 推理 |
| Ascend950DT | `ascend950` | Ascend 950DT | 推理 |
| KirinX90 | `ascend950` | Kirin X90 | 推理 |
| Ascend310P | `ascend310p` | Atlas 推理卡 | 推理 |

## 查询方法

```bash
# 方法 1: 使用 npu-smi
npu-smi info
# 查看 Device Name 字段

# 方法 2: 使用 torch_npu（推荐）
python3 -c "import torch_npu; print(torch_npu.npu.get_device_name(0))"

# 方法 3: 查看设备信息文件
cat /usr/local/Ascend/latest/version.info
```

## 架构说明

| 编译参数 | 架构代号 | 特点 |
|---------|---------|------|
| ascend910b | dav-2201 (arch32) | Atlas A2, 支持 flash attention |
| ascend910_93 | arch32 | Atlas A3, 更高性能 |
| ascend950 | arch35 | 推理专用, 可用模拟器 |
| ascend310p | arch20 | 推理卡, 不支持训练算子 |

## 注意事项

1. **必须使用正确的 SoC 版本编译**
   - 编译参数错误会导致运行时 ERROR 561103
   - kernel binary 会安装到对应 SoC 目录下

2. **同一硬件可能有多个显示名称**
   - `Ascend910_93` 和 `Ascend910_9362` 都对应 `ascend910_93`
   - `Ascend910B1`、`Ascend910_910B`、`Ascend910_91` 都对应 `ascend910b`

3. **不支持跨 SoC 运行**
   - 为 `ascend910b` 编译的算子无法在 `ascend910_93` 上运行
   - 必须针对目标设备重新编译

## kernel 安装目录

编译后 kernel 会安装到：

```
${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${soc}/
```

例如：
- `ascend910b` → `kernel/ascend910b/${算子名}/`
- `ascend910_93` → `kernel/ascend910_93/${算子名}/`

## 验证 kernel 安装

```bash
# 检查是否安装了对应 SoC 的 kernel
ls ${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/${soc}/${算子名}/

# 应看到：
# *.o       - kernel object 文件
# *.json    - kernel config 文件
```