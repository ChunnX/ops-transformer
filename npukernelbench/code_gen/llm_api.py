"""LLM API wrapper — ported from old framework's call_api_vllm."""

from __future__ import annotations

import os
import time
import traceback

from openai import OpenAI

from npukernelbench.config.models import FullConfig
from npukernelbench.utils.logger import init_logger

logger = init_logger(__name__)


def call_llm(
    sys_prompt: str,
    user_prompt: str,
    config: FullConfig,
) -> tuple[str, str]:
    """Call LLM API with DeepSeek reasoning format.

    Returns:
        (reasoning_content, generated_content)
    """
    base_url = config.chat.base_url
    api_key = os.environ.get("LLM_API_KEY", "YOUR_API_KEY")
    model_name = config.chat.model_path or config.chat.api_model

    client = OpenAI(api_key=api_key, base_url=base_url)

    formatted_prompt = (
        f"<|im_start|>user\n({user_prompt})<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    for attempt in range(config.chat.api_max_retries):
        try:
            completion = client.completions.create(
                model=model_name,
                prompt=formatted_prompt,
                stream=False,
                n=1,
                top_p=0.9,
                temperature=config.chat.temperature,
                max_tokens=config.chat.max_tokens,
            )
            content = completion.choices[0].text
            return "", content

        except Exception as e:
            logger.warning(
                f"LLM API call failed (attempt {attempt + 1}/{config.chat.api_max_retries}): {e}"
            )
            if attempt < config.chat.api_max_retries - 1:
                time.sleep(config.chat.api_retry_delay)
            else:
                logger.error(f"LLM API call failed after {config.chat.api_max_retries} attempts")
                raise

    return "Generation Failed", "Generation Failed"


def call_llm_stub(
    sys_prompt: str,
    user_prompt: str,
    config: FullConfig,
) -> tuple[str, str]:
    """Stub LLM call for testing — returns a minimal valid response.

    Returns (reasoning, content) with a placeholder kernel_impl block.
    """
    # Try to read a reference answer if available
    from npukernelbench.utils.io import safe_read_file
    from npukernelbench.models.task import BuildHandle
    import re

    # Extract kernel_name from the prompt heuristically
    kernel_match = re.search(r"op_kernel/(\w+)\.cpp", user_prompt)
    kernel_name = kernel_match.group(1) if kernel_match else "custom_op"

    stub_content = f"""<kernel_impl>
```cpp
// STUB: This is a placeholder response from call_llm_stub.
// Replace with actual LLM API call when model is available.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void {kernel_name}(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace)
{{
    // Stub implementation
}}
```
</kernel_impl>"""

    logger.info("Using stub LLM response (no real model connected)")
    return "", stub_content
