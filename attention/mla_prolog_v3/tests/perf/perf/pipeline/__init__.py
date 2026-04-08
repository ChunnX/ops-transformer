"""perf.pipeline - AIC/AIV pipeline DAG, timeline, and roofline analysis.

Public API re-exports.
"""

from .dag import PipelineStage, PipelineResult, build_pipeline_dag
from .kernel_time import estimate_kernel_time
from .timeline import format_timeline
from .roofline import RooflineResult, roofline_analysis, format_roofline_report
