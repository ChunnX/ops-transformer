from __future__ import annotations

import unittest
from pathlib import Path

from mc2.tools.moe_ep_profiler.analyze_cycle import DEFAULT_STAGE_NAMES
from mc2.tools.moe_ep_profiler.validate_instrumentation import (
    business_tokens,
    inspect_source,
)


def source() -> str:
    return """
#include "moe_ep_profiler.h"
enum class MoeEpCombineProfileStage : uint32_t { KERNEL=0, ADDRESS_TABLE=1 };
MOE_EP_PROFILE_DECLARE();
void Init() { MOE_EP_PROFILE_INIT(context, MoeEpProfileOp::COMBINE, cores); init(); }
void Process() {
MOE_EP_PROFILE_MARK(MOE_EP_PROFILE_EVENT(MoeEpProfileOp::COMBINE, MoeEpProfileEngine::SCALAR,
    MoeEpCombineProfileStage::KERNEL, MoeEpProfileBoundary::BEGIN));
business();
MOE_EP_PROFILE_MARK(MOE_EP_PROFILE_EVENT(MoeEpProfileOp::COMBINE, MoeEpProfileEngine::SCALAR,
    MoeEpCombineProfileStage::KERNEL, MoeEpProfileBoundary::END));
MOE_EP_PROFILE_FLUSH();
}
"""


class InstrumentationLintTest(unittest.TestCase):
    def test_build_switch_resets_cached_on_and_keeps_wheel_abi_in_step(self) -> None:
        root = Path(__file__).resolve().parents[4]
        build = (root / "build.sh").read_text(encoding="utf-8")
        setup = (root / "torch_extension/setup.py").read_text(encoding="utf-8")
        wrapper = (root / "mc2/common/torch_extension/elastic_buffer.py").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            build,
            r'(?s)if \[\[ "\$\{ENABLE_MOE_EP_PROFILING\}" == "TRUE" \]\]; then\s+'
            r'CUSTOM_OPTION="[^"\n]*-DENABLE_MOE_EP_PROFILING=ON"\s+else\s+'
            r'(?:#[^\n]*\n\s*)*CUSTOM_OPTION="[^"\n]*-DENABLE_MOE_EP_PROFILING=OFF"\s+fi',
        )
        self.assertRegex(
            build,
            r"(?s)export CANN_OPS_MOE_EP_PROFILING_BUILD=1\s+else\s+"
            r"unset CANN_OPS_MOE_EP_PROFILING_BUILD\s+fi",
        )
        self.assertIn(
            'os.environ.get("CANN_OPS_MOE_EP_PROFILING_BUILD", "").strip() == "1"',
            setup,
        )
        self.assertIn('"_MOE_EP_PROFILING_BUILD = %s" % _moe_ep_profiling_build', setup)
        self.assertRegex(
            wrapper,
            r'if _MOE_EP_PROFILING_BUILD:\s+args.append\("-DENABLE_MOE_EP_PROFILING=1"\)',
        )

    def test_stage_ids_preserve_legacy_and_append_new_paths(self) -> None:
        combine = DEFAULT_STAGE_NAMES["moe_ep_combine"]
        self.assertEqual(combine[1], "ADDRESS_TABLE")
        self.assertEqual(
            [combine[index] for index in range(4, 8)],
            [
                "FLAG_SOURCE_INIT",
                "LOCAL_METADATA_COPY",
                "LOCAL_BARRIER",
                "CORE_ASSIGNMENT",
            ],
        )
        dispatch = DEFAULT_STAGE_NAMES["moe_ep_dispatch_epilogue"]
        self.assertEqual(
            [dispatch[index] for index in range(6, 10)],
            ["COUNT_BARRIER", "RANK_ROW_STARTS", "CACHED_RANK_OFFSETS", "OUTPUT_SYNC"],
        )

    def test_good_anchors_and_pairs(self) -> None:
        report = inspect_source("moe_ep_combine", source())
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["lexical_marker_sites"], 2)

    def test_missing_end_and_early_flush_fail(self) -> None:
        bad = source().replace(
            "MoeEpProfileBoundary::END", "MoeEpProfileBoundary::BEGIN"
        )
        errors = inspect_source("moe_ep_combine", bad)["errors"]
        self.assertTrue(any("unmatched" in error for error in errors))
        self.assertTrue(any("Flush" in error for error in errors))

    def test_stripping_keeps_business_sync_and_handles_nested_macros(self) -> None:
        baseline = "void Init() { init(); } void Process() { business(); }"
        self.assertEqual(business_tokens(source()), business_tokens(baseline))
        changed = source().replace("business();", "business(); SyncAll<true>();")
        self.assertNotEqual(business_tokens(changed), business_tokens(baseline))


if __name__ == "__main__":
    unittest.main()
