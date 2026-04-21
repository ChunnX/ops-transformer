"""Kernel code parser — extracts tiling/kernel from LLM output."""

from __future__ import annotations

import re
from typing import Optional, Tuple


class KernelParser:
    """Parse generated content into tiling and kernel components."""

    def __init__(self, kernel_name: str, kernel_only: bool = True):
        self.kernel_name = kernel_name
        self.kernel_only = kernel_only

    def parse(self, gen_content: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse gen_content → (tiling_def, tiling_content, kernel_content).

        In kernel_only mode, returns (None, None, kernel_impl).
        In full mode, returns (tiling_h, tiling_cpp, kernel_cpp).
        """
        if self.kernel_only:
            return self._parse_kernel_only(gen_content)
        else:
            return self._parse_full(gen_content)

    def _parse_kernel_only(self, content: str) -> Tuple[None, None, str]:
        match = re.search(r"<kernel_impl>(.*?)</kernel_impl>", content, re.DOTALL)
        if not match:
            raise RuntimeError("Missing <kernel_impl> tag in generated content")

        code = _extract_code_block(match.group(1).strip())
        if not code:
            raise RuntimeError("Empty <kernel_impl> content")
        return None, None, code

    def _parse_full(self, content: str) -> Tuple[str, str, str]:
        kn = self.kernel_name
        patterns = [
            rf"================== op_host/{kn}_tiling\.h ==================\n```cpp\n(.*?)\n```",
            rf"================== op_host/{kn}\.cpp ==================\n```cpp\n(.*?)\n```",
            rf"================== op_kernel/{kn}\.cpp ==================\n```cpp\n(.*?)\n```",
        ]
        results = []
        for pat in patterns:
            m = re.search(pat, content, re.DOTALL)
            if not m:
                raise RuntimeError(f"Missing section for pattern: {pat[:50]}...")
            code = _extract_code_block(m.group(1).strip())
            if not code:
                raise RuntimeError(f"Empty content for pattern: {pat[:50]}...")
            results.append(code)

        return results[0], results[1], results[2]


def _extract_code_block(text: str) -> str:
    """Extract code from ```cpp ... ``` if present, otherwise return as-is."""
    m = re.search(r"```cpp\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()
