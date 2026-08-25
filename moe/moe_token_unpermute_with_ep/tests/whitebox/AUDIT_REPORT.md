# MoeTokenUnpermuteWithEp Kernel whitebox generation audit

Final unified-entry result: **PASS**. The generated source contains 106 cases. The regular executable CSV contains 105 cases; one directed 134,212,224-element case is isolated in the oversized CSV.

All commands below were run after `source /home/z50058944/.config/glacier/env.sh`, with:

```text
PY=/home/z50058944/venv/bin/python
HELPER=/home/z50058944/code/skills-dev/plugins-official/ops-registry-invoke-glacier/.agents/skills/ascendc-whitebox-design/references/whitebox_designer.py
```

## Unified generation

```text
$PY "$HELPER" validate-inputs tests/whitebox/design_facts.json tests/whitebox/opspec.json
$PY "$HELPER" design-to-ttk tests/whitebox/design_facts.json tests/whitebox/opspec.json tests/whitebox/design_doc_cases.json tests/whitebox/ttk_design_doc_cases.csv
```

Result: input preflight `ok=true`, 0 errors, 0 warnings. The helper emitted 105 regular rows and isolated one oversized row; `design_doc_cases.json` contains all 106 source cases.

## Final helper gates

```text
$PY "$HELPER" validate-ttk tests/whitebox/ttk_design_doc_cases.csv tests/whitebox/opspec.json
$PY "$HELPER" validate-ttk tests/whitebox/ttk_design_doc_cases_oversized.csv tests/whitebox/opspec.json
$PY "$HELPER" validate-shapes tests/whitebox/design_doc_cases.json tests/whitebox/design_facts.json tests/whitebox/opspec.json
$PY "$HELPER" coverage-audit tests/whitebox/design_doc_cases.json tests/whitebox/design_facts.json tests/whitebox/opspec.json
```

- Regular `validate-ttk`: PASS, 105 unique rows, 26 columns, three formal input slots.
- Oversized `validate-ttk`: PASS, one row, 26 columns, three formal input slots.
- `validate-shapes`: PASS, 106 checked, 0 violations.
- `coverage-audit`: PASS, 106 cases, 1050 expected signatures independently recomputed, 98 actual signatures, no missing axes or attributes, 0 input-integrity violations.
- Structural axes: size S/M/L; dtype FP16/BF16/FP32; alignment aligned/nonaligned/prime-nonaligned; fixed rank 2.
- Attributes: TopK 1/2/8/511/512; fallback, zero-start EP, and nonzero-start EP ranges; padded mode false; restore shape empty.

## Independent semantic audit

```text
$PY tests/whitebox/semantic_audit.py
```

Result: PASS. All 106 source cases and all 106 CSV rows were checked. The audit verifies exact rank/dtype/formal-slot contracts, present-probs and absent-probs shape coupling, output shape/dtype, fallback effective TopK=1, local EP span, runtime-valid fallback indices, outside-range zero-contribution scenarios, CSV/source equality, oversized isolation, all nine tokens/probs dtype pairs, all three absent-probs token dtypes, absent-probs TopK 1/2/8/511/512, tiling keys 0/1/2/3, signed zero, finite positive/negative values, large finite values, Inf, NaN, and the deterministic-repeat design obligation. It found 0 errors and 50 cases whose generated runtime EP indices include an outside-range contribution.

The initially generated invalid absent-probs fallback case and the helper regressions that exposed it remain preserved under `audit/` as failure evidence. The final source and CSVs were regenerated from the single declarative `design_facts.json` + `opspec.json` input; no generated case or CSV row was hand-edited.

Kernel-mode non-contiguous tensors are outside the raw GM-buffer interface and remain assigned to the ACLNN route. The `deterministic_repeat` case is present, while byte-identical repeated execution remains a downstream runner obligation.
