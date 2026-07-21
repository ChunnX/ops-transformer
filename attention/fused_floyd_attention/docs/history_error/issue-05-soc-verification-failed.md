# SoC 版本验证失败

## 现象

```
SoC version ascend950 verification failed. This SoC is not configured
through the AddConfig API of the OpDef class.
aclnnFusedFloydAttentionGetWorkspaceSize failed! error code: 361001
```

## 根因

ATK 的 `get_opp_lib_path`（acl_wrapper.py:461-510）按顺序搜索 API .so：
1. `ATK_CUSTOM_OPP_PATH`（直接返回 .so 路径）
2. `ASCEND_CUSTOM_OPP_PATH/vendors/{prefix}{suffix}/op_api/lib/libcust_opapi.so`
3. built-in `${ASCEND_OPP_PATH}/lib64/libopapi_{xx}.so`

`set_env.bash` 把 `ASCEND_CUSTOM_OPP_PATH` 设成了 vendor 目录本身
（`.../vendors/custom_transformer`），但 ATK 期望的是**包含 `vendors/` 的 cann 根目录**。
路径被拼成 `.../vendors/custom_transformer/vendors/custom_transformer/...`（不存在），
ATK 回退到 built-in `libopapi_transformer.so`。

内置 .so 虽含 `aclnnFusedFloydAttentionGetWorkspaceSize`，但**没有为 FusedFloydAttention 注册
ascend950 SoC 支持** → 验证失败。真正带 ascend950 配置的自定义
`libcust_opapi.so` 未被加载。

## 修复

设置环境变量直接指向自定义 opapi .so：

```bash
export ATK_CUSTOM_OPP_PATH=/home/j00946653/cann618/cann-9.1.0/vendors/custom_transformer/op_api/lib/libcust_opapi.so
```
