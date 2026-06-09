#!/usr/bin/env python3
"""
单独运行某个测试case，方便调试
用法：python run_single_case.py <case_id>
例如：python run_single_case.py 1
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from precision_test_general import run_single_test, TEST_CASES


def main():
    if len(sys.argv) != 2:
        print(f"用法: python {sys.argv[0]} <case_id>")
        print(f"可用case_id: 0~{len(TEST_CASES)-1}")
        for i, case in enumerate(TEST_CASES):
            print(f"  {i}: {case['name']}")
        sys.exit(1)

    case_id = int(sys.argv[1])
    if case_id < 0 or case_id >= len(TEST_CASES):
        print(f"错误: case_id必须在 0~{len(TEST_CASES)-1} 范围内")
        sys.exit(1)

    results = run_single_test(case_id)

    if results:
        print("\n" + "=" * 80)
        print("精度判断标准: 相对误差 < 2e-3 为达标")
        print("=" * 80)
        max_rel_err = max(
            results["dQueryIndex"]["rel_err"],
            results["dKeyIndex"]["rel_err"],
            results["dWeights"]["rel_err"],
            results["loss"]["rel_err"],
        )
        status = "✓ 达标" if max_rel_err < 2e-3 else "✗ 未达标"
        print(f"结果: {status} (max_rel_err={max_rel_err:.6e})")


if __name__ == "__main__":
    main()