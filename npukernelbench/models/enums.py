"""Enumerations used across the framework."""

from enum import Enum


class TestStage(str, Enum):
    CODE_GEN = "code_gen"
    COMPILE = "compile"
    PRECISION = "precision"
    PERFORMANCE = "perf"

    @property
    def result_attr(self) -> str:
        """Attribute name on BuildHandle / TestHandle for this stage's result."""
        mapping = {
            TestStage.CODE_GEN: "code_gen_result",
            TestStage.COMPILE: "compile_result",
            TestStage.PRECISION: "precision_result",
            TestStage.PERFORMANCE: "perf_result",
        }
        return mapping[self]


class CompileMode(str, Enum):
    MSOPGEN = "msopgen"
    CANN_OPS = "cann-ops"
