"""Prompt construction for Ascend C code generation — ported from old framework."""

from __future__ import annotations

import os
import re
import textwrap
from typing import Optional, Tuple

import pandas as pd

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import BuildHandle
from npukernelbench.code_gen.prompts.sft_samples import (
    KERNEL_ONLY_EXAMPLE_CODE,
    KERNEL_ONLY_CODE_REQUIREMENT,
    FULL_EXAMPLE_CODE,
    FULL_CODE_REQUIREMENT,
)
from npukernelbench.utils.io import safe_read_file


def build_code_gen_prompt(
    task: BuildHandle,
    config: FullConfig,
) -> Tuple[str, str]:
    """Build (sys_prompt, user_prompt) for Ascend C code generation."""
    kernel_name = task.metadata.kernel_name
    template_path = task.paths.template_path

    # 1. Load template files
    if config.kernel_only_mode:
        required_files = [
            os.path.join(template_path, "op_kernel", f"{kernel_name}.cpp"),
        ]
    else:
        required_files = [
            os.path.join(template_path, "op_kernel", f"{kernel_name}.cpp"),
            os.path.join(template_path, "op_host", f"{kernel_name}.cpp"),
        ]

    template_contents = [safe_read_file(fp) or "" for fp in required_files]

    # 2. Load api_desc
    api_desc = safe_read_file(task.paths.api_desc_path) or ""

    # 3. Load test case data
    test_case_data = _load_test_case_data(task.paths.test_cases_path, config)

    # 4. Enhance api_desc with dtype info
    api_desc = _enhance_api_description(api_desc, test_case_data, config.static_shape_mode)

    # 5. Select SFT example and code requirements based on mode
    if config.kernel_only_mode:
        example_content = KERNEL_ONLY_EXAMPLE_CODE
        code_requirement = KERNEL_ONLY_CODE_REQUIREMENT
    else:
        example_content = FULL_EXAMPLE_CODE
        code_requirement = FULL_CODE_REQUIREMENT

    # 6. Generate answer template
    answer_template = _generate_answer_template(
        kernel_name, template_contents, config.kernel_only_mode,
    )

    # 7. Build user prompt
    user_prompt = textwrap.dedent(f"""\
你是一个精通Ascend C编程的专家。你的任务是根据提供的文档内容生成符合特定任务需求的Ascend C代码。

【任务背景】
我将提供3个关键文件，描述了待实现算子的信息，请仔细阅读每一个：
1. api_desc.md - 待实现算子的接口描述和公式定义相关内容，包含了算子的功能要求和数学表达式
2. test_cases.csv - 其设定了待实现算子需要重点考虑的datatype和shape信息，用于更合适的tiling设计和性能优化
3. hardware.txt - 其给出了运行待实现算子的硬件型号规格信息

【输出格式要求】
- 同时输出思维过程和答案，思维链进行合理的分析推导，答案部分必须严格按照提供的模板格式输出
- 不要修改已有的函数名、类定义、模板参数、命名空间等
- 确保答案部分的代码在XML标签内，每个文件对应一个标签块
- 确保检查算子参考示例写法正确注册了tiling结构，并根据输入shape信息，设计tiling参数及数值

【代码实现要点】
实现代码时，请确保：
{code_requirement}

{example_content}

接下来我将提供待实现算子所需文件信息，请参考这些信息按照要求完成所有指定文件的实现。
【输入文件】
下面是第一个文件："api_desc.md"，是待实现算子的接口描述和公式定义：
{api_desc}

下面是第二个文件："test_cases.csv"，其设定了待实现算子需要重点考虑的典型shape以及DataType信息，用于设计最优tiling策略：
{test_case_data}

下面是第三个文件："hardware.txt"，其给出的是硬件信息，待实现算子将要在该硬件上运行：
【
# [Platform Info configuration begin]
#**************************************************************************************
#
[version]
SoC_version=Ascend910_9392
Short_SoC_version=Ascend910_93   [注：Ascend910_93为NPU型号名]

[SoCInfo]
ai_core_cnt=24
cube_core_cnt=24
vector_core_cnt=48

[AICoreSpec]
ub_size=196608
】

【输出模板】
你需要输出的文件模板如下，请对其进行填补：
{answer_template}

【最终任务】
请分析所有提供的信息，特别关注：
 - 算子描述中的功能要求、公式定义和实现细节
 - 测试用例中提供的datatype和shape等信息
 - 输入准备代码中的参数构造和使用方法

请提供完整的思维过程和符合要求的代码答案。"
    """)

    sys_prompt = ""
    return sys_prompt, user_prompt


def _load_test_case_data(csv_path: str, config: FullConfig) -> Optional[pd.DataFrame]:
    """Load test case CSV, return DataFrame or None."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, sep=";")
        n = config.n_case
        if n and n > 0:
            df = df.head(n)
        return df
    except Exception:
        return None


def _enhance_api_description(
    api_desc: str,
    test_case_data: Optional[pd.DataFrame],
    static_shape_mode: bool,
) -> str:
    """Inject dtype info from test_cases into api_desc."""
    if test_case_data is None or test_case_data.empty:
        return api_desc
    if "dtype" not in test_case_data.columns:
        return api_desc

    dtype_info = f"\n## 支持的数据类型\n- {test_case_data['dtype']}\n\n"
    return api_desc.replace("## 功能描述", dtype_info + "## 功能描述")


def _generate_answer_template(
    kernel_name: str,
    template_contents: list[str],
    kernel_only_mode: bool,
) -> str:
    """Generate the answer template that LLM should fill."""
    if kernel_only_mode:
        kernel_impl = template_contents[0] if template_contents else ""
        # Remove tiling parameter from workspace signature
        cpp_code = re.sub(r"(workspace[^)]*\))", "workspace)", kernel_impl)
        return textwrap.dedent(f"""\
<kernel_impl>
```cpp
{cpp_code}
```
</kernel_impl>
""")
    else:
        kernel_impl = template_contents[0] if len(template_contents) > 0 else ""
        host_impl = template_contents[1] if len(template_contents) > 1 else ""

        # Remove auto-set BlockDim lines
        host_impl = host_impl.replace(
            "context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());",
            "",
        )
        host_impl = host_impl.replace(
            "context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());",
            "",
        )

        op_name = kernel_name.split("_")[0]
        op_name = "".join(w.capitalize() for w in op_name.split("_"))

        return textwrap.dedent(f"""\
================== op_host/{kernel_name}_tiling.h ==================
```cpp
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF(TilingData)

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS({op_name}, TilingData)
}}
```
================== op_host/{kernel_name}.cpp ==================
```cpp
#include "{kernel_name}_tiling.h"
{host_impl}
```
================== op_kernel/{kernel_name}.cpp ==================
```cpp
{kernel_impl}
```

""")
