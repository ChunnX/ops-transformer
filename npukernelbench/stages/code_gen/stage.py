"""Code generation stage — LLM or reference code loading."""

from __future__ import annotations

import os

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import BuildHandle
from npukernelbench.models.results import CodeGenResult
from npukernelbench.pipeline.context import PipelineContext
from npukernelbench.stages.code_gen.parser import KernelParser
from npukernelbench.utils.io import safe_read_file, save_json_file, load_json_file
from npukernelbench.utils.logger import init_logger, add_file_handler


class CodeGenStage:
    """Load or generate kernel source code for each BuildHandle."""

    name = "code_gen"

    def __init__(self, config: FullConfig):
        self.config = config

    def execute(self, task: BuildHandle, ctx: PipelineContext) -> None:
        logger = init_logger(f"codegen.{task.short_id}")
        os.makedirs(os.path.dirname(task.paths.code_gen_log), exist_ok=True)
        add_file_handler(task.paths.code_gen_log, logger=logger)

        if self.config.chat.active:
            self._llm_generate(task, logger)
        else:
            self._load_reference(task, logger)

    def execute_batch(self, tasks: list[BuildHandle], ctx: PipelineContext) -> None:
        for task in tasks:
            self.execute(task, ctx)

    def _llm_generate(self, task: BuildHandle, logger) -> None:
        """Use LLM to generate code."""
        prompt_file = task.paths.prompt_save_file

        # 1. Check for existing results
        if os.path.exists(prompt_file):
            data = load_json_file(prompt_file)
            if data and data.get("gen_content"):
                gen_content = data["gen_content"]
                reasoning = data.get("reasoning_content", "")
                task.gen_content = f"{reasoning}\n{gen_content}" if reasoning else gen_content
                self._parse_content(task, logger)
                task.code_gen_result = CodeGenResult(
                    success=True,
                    log_file=task.paths.code_gen_log,
                    gen_content=gen_content,
                    reasoning_content=reasoning,
                )
                logger.info(f"Loaded existing code gen for {task.short_id}")
                return

        # 2. Build prompt
        from npukernelbench.code_gen.prompts.builder import build_code_gen_prompt
        sys_prompt, user_prompt = build_code_gen_prompt(task, self.config)

        # Save prompt for debugging
        os.makedirs(os.path.dirname(prompt_file), exist_ok=True)

        # 3. Call LLM
        from npukernelbench.code_gen.llm_api import call_llm
        reasoning, content = call_llm(sys_prompt, user_prompt, self.config)

        if not content or content == "Generation Failed":
            logger.error(f"LLM generation failed for {task.short_id}")
            task.code_gen_result = CodeGenResult(
                success=False,
                log_file=task.paths.code_gen_log,
            )
            return

        # 4. Save generation data
        gen_data = {
            "gen_codes_sys_prompt": sys_prompt,
            "gen_codes_user_prompt": user_prompt,
            "gen_content": content,
            "reasoning_content": reasoning,
        }
        save_json_file(gen_data, prompt_file)

        # 5. Parse and set results
        task.gen_content = f"{reasoning}\n{content}" if reasoning else content
        self._parse_content(task, logger)
        task.code_gen_result = CodeGenResult(
            success=True,
            log_file=task.paths.code_gen_log,
            gen_content=content,
            reasoning_content=reasoning,
        )
        logger.info(f"Code gen completed for {task.short_id}")

    def _load_reference(self, task: BuildHandle, logger) -> None:
        """Load reference answer code (no LLM needed)."""
        task.code_gen_result = CodeGenResult(success=True, log_file=task.paths.code_gen_log)
        logger.info(f"Reference mode: code gen skipped for {task.short_id}")

    def _parse_content(self, task: BuildHandle, logger) -> None:
        """Parse gen_content into tiling/kernel parts."""
        parser = KernelParser(
            kernel_name=task.metadata.kernel_name,
            kernel_only=self.config.kernel_only_mode,
        )
        task.tiling_def, task.tiling_content, task.kernel_content = parser.parse(task.gen_content)
