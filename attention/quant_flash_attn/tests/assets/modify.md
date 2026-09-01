首先参考https://gitcode.com/cann/ops-transformer/pull/9872和https://gitcode.com/cann/ops-transformer/pull/10560这两笔PR的修改，现在要把quant_flash_attn也这样修改，主要修改点如下：

1、类似地把__spec__的key替换成torch.ops.cann_ops_transformer.quant_flash_attn
2、把wrapper中用到的一些对象提取到npu_preprocess中，舍弃掉wrapper

注：
spec的作用如下：
__spec__ dict 里的 key（api_name）就是 CSV 用例表里 api_name 列填的值，TTK 用它做查找 key 把用例和 spec 类关联起来。
作用：用例 ↔ spec 类的匹配桥梁
CSV 用例 (api_name="torch.ops.cann_ops_transformer.quant_flash_attn")
        ↓
get_spec_attr(api_name, "golden"/"tolerance"/"compare"/..., plugin_path)
        ↓
__spec__ dict 查找 → "QuantFlashAttnSpec" 类
        ↓
取出类里的 golden/tolerance/compare/pre_compare/third_party/customize_inputs 等属性
E2E 模式下，TTK 在 framework_api/profiling.py 多处用 testcase.api_name 去 get_spec_attr 查 tolerance/golden/compare/pre_compare（profiling.py:544,614,620,629,...）。
三个 key 的含义
__spec__ = {
    # 标准 torch.ops 自定义算子：CSV 的 api_name 直接填这个
    "torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnSpec",
    "torch.ops.cann_ops_transformer.quant_flash_attn_metadata": "QuantFlashAttnMetadataSpec",
    # 普通 Python 函数路径：CSV 的 api_name 填这个，TTK 当作可调用对象解析
    "qfa_main_wrapper.run_main": "QfaMainSpec",
}
- 前两个：torch.ops.<ns>.<op> 形式，参数从算子自带 _schemas（FunctionSchema）自动解析，无需手写签名
- 第三个 qfa_main_wrapper.run_main：非 torch API，是一个普通 Python 模块函数路径，TTK 通过 _resolve_function 解析（importlib + getattr），用 inspect.signature / docstring 拿签名。这通常用于把某个 wrapper 函数（如 qfa 的 main 入口）作为被测 API，配一个 spec 提供 golden/compare
关键点
1. key 必须与 CSV api_name 完全一致（SKILL.md:200），不一致就找不到 spec，会回退到默认行为（E2E 默认在 CPU 跑同 API 作 golden）
2. 类名（如 QuantFlashAttnSpec） 必须在该 spec 文件中定义，loader 通过 AST 扫描 __spec__ 建索引，命中时才 exec 该文件
3. 多 key 可指向同一个 spec 文件：一个文件可以注册多个算子的 spec
4. 注册名 vs 解析签名：注册名只用于"找 spec 类"；签名解析是另一套机制（见上一问），两者独立
简言之：__spec__ 的 key 就是用例表里 api_name 列的字符串，是连接 CSV 用例与 spec 资产（golden/tolerance/compare 等）的查找索引。