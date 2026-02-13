# aclnnSTFT

## 功能描述

短时傅里叶变换（STFT）。从 CANN ops-math/math/stft 迁移至 NPUKernelBench。

### 接口说明

- 输入：x（1D/2D 信号）, plan（FFT plan）, window（可选窗函数）
- 属性：n_fft, hop_length, win_length, normalized, onesided, return_complex
- 输出：y（STFT 结果，2D/3D/4D）

### 精度验证说明

当前 validation 使用占位 plan（zeros），与 PyTorch torch.stft 参考在数值上存在偏差。若需严格精度通过，需提供与 kernel 约定格式一致的 FFT plan 生成逻辑。
