# Whitebox helper failure reproductions

Helper (single entry point):

`/home/z50058944/code/skills-dev/plugins-official/ops-registry-invoke-glacier/.agents/skills/ascendc-whitebox-design/references/whitebox_designer.py`

All commands below must be run after:

`source /home/z50058944/.config/glacier/env.sh`

## 1. Coverage audit expands a fixed-rank heterogeneous-input contract

Minimal inputs and generated output:

- `tests/whitebox/probe_optional/design_facts.json`
- `tests/whitebox/probe_optional/opspec.json`
- `tests/whitebox/probe_optional/design_doc_cases.json`

Reproduction:

```bash
/home/z50058944/venv/bin/python "$HELPER" coverage-audit \
  tests/whitebox/probe_optional/design_doc_cases.json \
  tests/whitebox/probe_optional/design_facts.json \
  tests/whitebox/probe_optional/opspec.json
```

Observed non-zero report:

- `validate-inputs` reports `scenario_ranks: [2]`.
- Generated `target_sigs` contain only `rank=2` and `dtype=float16`.
- `coverage-audit` nevertheless reports expected rank `[2,3,4,5,6,7,8]`.
- It also reports expected dtype `int32`, although `int32` is the fixed index dtype and is not legal for `permuted_tokens`.
- The report therefore has `ok: false`, missing ranks/dtype and cannot be satisfied without violating the OpDef contract.

The public workflow describes this path as `coverage-audit -> build_model(dims)`. The
whitebox-designer hard rule forbids this agent from reading or grepping helper internals,
so the exact private caller line is intentionally left for the root task that owns the
helper fix.

## 2. Optional input absence crashes materialization

Minimal inputs:

- `tests/whitebox/audit/repro_optional_absent/design_facts.json`
- `tests/whitebox/audit/repro_optional_absent/opspec.json`

Reproduction:

```bash
/home/z50058944/venv/bin/python "$HELPER" validate-inputs \
  tests/whitebox/audit/repro_optional_absent/design_facts.json \
  tests/whitebox/audit/repro_optional_absent/opspec.json
/home/z50058944/venv/bin/python "$HELPER" design-to-ttk \
  tests/whitebox/audit/repro_optional_absent/design_facts.json \
  tests/whitebox/audit/repro_optional_absent/opspec.json \
  tests/whitebox/audit/repro_optional_absent/design_doc_cases.json \
  tests/whitebox/audit/repro_optional_absent/ttk_design_doc_cases.csv --case-limit 20
```

Observed stack:

```text
whitebox_designer.py:7339 in <module>
whitebox_designer.py:7335 in main
whitebox_designer.py:6495 in _cmd_design_to_ttk
whitebox_designer.py:6381 in generate_design_doc_cases_and_ttk
whitebox_designer.py:6339 in generate_design_doc_cases
whitebox_designer.py:5349 in gen_cases
_vendored_scripts/wb_scenario_gen.py:624 in gen_cases
TypeError: 'NoneType' object is not iterable
```

The absent optional input is declared, rather than post-edited, as
`semantic_extract.extreme_inputs[0].shapes.probs = null`. Preflight returns `ok: true`
with no warning, then generation raises the exception above.

## 3. First full-suite shape-rule failure (preserved evidence)

The first full attempt used attr-dependent input rules to express TopK. Preflight returned
`ok: true`, but `validate-shapes` checked 97 cases and returned 194 violations: every
`sorted_indices` and `probs` rule was not evaluable. Its regular CSV had 42 rows and its
oversized CSV had 55 rows. This output was rejected and must not be executed.
