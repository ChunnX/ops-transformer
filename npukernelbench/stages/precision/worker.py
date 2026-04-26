"""Subprocess entry point for precision evaluation."""

from __future__ import annotations

import pickle
import sys

from npukernelbench.pipeline.context import PipelineContext
from npukernelbench.stages.precision.stage import PrecisionStage


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m npukernelbench.stages.precision.worker INPUT.pkl OUTPUT.pkl", file=sys.stderr)
        return 2

    input_path, output_path = args
    with open(input_path, "rb") as f:
        payload = pickle.load(f)

    config = payload["config"]

    stage = PrecisionStage(config)
    if "tasks" in payload:
        results = []
        for task in payload["tasks"]:
            stage.execute_inline(task, PipelineContext(config=config))
            results.append(task.precision_result)
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        return 0

    task = payload["task"]
    stage.execute_inline(task, PipelineContext(config=config))
    with open(output_path, "wb") as f:
        pickle.dump(task.precision_result, f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
