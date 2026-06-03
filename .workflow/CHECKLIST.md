# CHECKLIST — 检查清单

> 各阶段的质量检查项。

## 审查原则

- 基于事实，不基于假设
- 可验证，可复现
- 明确标注：✅ 通过 / ❌ 不通过 / ⬜ 不适用

## 阶段一：项目脚手架 + op_host 基础

### 目录与构建
- [ ] 目录结构完整（op_host/, op_api/, tests/, docs/）
- [ ] CMakeLists.txt 可被父目录正确解析
- [ ] 不破坏已有算子编译

### def.cpp
- [ ] 9 个必选输入 + 2 个可选输入全部定义
- [ ] 5 个输出全部定义
- [ ] head_nums 为 REQUIRED ListInt
- [ ] epsilon 为 OPTIONAL float (默认 1e-6)
- [ ] k_cache/v_cache/k_scale_cache 标记 in-place/alias
- [ ] ascend950 平台配置正确

### infershape.cpp
- [ ] q_out shape = [T, Nq, D]
- [ ] q_scale shape = [T, Nq]
- [ ] cache 输出 shape = 对应输入 shape
- [ ] qkv.shape[1] == Nq+Nk+Nv 校验
- [ ] cache head 维与 head_nums 一致校验
- [ ] D 为偶数校验
- [ ] dtype 校验完整
- [ ] cos_sin.shape[1] == D 校验
- [ ] slot_mapping.shape == [T] 校验
- [ ] cu_seq_len rank == 1 校验

## 阶段二：op_api

### L2 公共 API
- [ ] GetWorkspaceSize 签名正确
- [ ] Execute 签名正确
- [ ] 空指针参数校验
- [ ] dtype 校验
- [ ] optionalRotation != nullptr 校验
- [ ] optionalVScale != nullptr 校验
- [ ] headNums 非空、长度 3、值正整数校验
- [ ] executor 创建和传递
- [ ] in-place 输出 alias 处理

### L0 内部实现
- [ ] OP_TYPE_REGISTER 注册
- [ ] 可选输入 nullptr 时创建 placeholder
- [ ] aclIntArray → aclTensor 转换
- [ ] INFER_SHAPE 调用
- [ ] ADD_TO_LAUNCHER_LIST_AICORE 调用

## 阶段三：测试 + 文档 + 验收

### op_api UT
- [ ] 正常调用成功
- [ ] 参数错误返回正确错误码
- [ ] dtype 错误返回正确错误码
- [ ] 空指针返回正确错误码
- [ ] 可选输入缺失返回正确错误码
- [ ] headNums 非法（长度!=3、值非正）返回正确错误码

### 文档
- [ ] ACLNN 接口文档完整（输入输出/shape/dtype/约束/调用方式）
- [ ] README 描述算子功能、支持平台、编译方式

### 编译验证
- [ ] 所有源文件编译通过
- [ ] UT 全部 pass
- [ ] 不影响已有算子

## 通用检查项

### 代码规范
- [ ] 文件命名与注释匹配
- [ ] Header guard 与文件名匹配
- [ ] 无遗留 TODO/FIXME 未处理
