# Transfer-test scan reconciliation

`scan_report.md` is the unmodified output of `ops-transfer-test-scan`. Its
overall result is not PASS. The machine report and the actual route evidence
must be read separately:

- The scan was rerun after platform-isolation commit `289067d40`. The separate
  source/build isolation audit is PASS, including the Ascend910B automatic
  ACLNN compile route; this does not rewrite any scanner FAIL as PASS.

- `3-ascend950-ini`, `6-legacy-tiling`, and `7-def-driven-dtype` are PASS.
- Prototype and Infer items are FAIL against the scanner's prescribed
  `op_graph` layout. This repository delivers OpDef and InferShape from
  `op_host`; no new `op_graph` interface was fabricated for this test task.
- Golden is reported FAIL because the scanner searches its registered-operator
  directory convention. The actual standard Golden is
  `tests/assets/golden.py`, and its fixed-vector CPU self-check and all final
  TTK precision suites pass.
- GEIR is reported REMEDIATE because the scanner searches for the literal
  `RunGraph` and verification strings only in the dynamic `.cpp`; the real
  calls and checks are shared by the static and dynamic examples through
  `moe_token_unpermute_with_ep_geir_test_common.h`. Direct ES execution is
  independently archived as static 14/14 and dynamic 6/6 PASS.
- The scan skill's unknown-rank validator additionally requires three distinct
  concrete Rank signatures. This operator's actual protocol fixes ranks to
  tokens/probs rank 2 and indices rank 1, so the test covers three different
  concrete sizes under `[-2]` but does not invent illegal ranks.
- Items 9--13 remain UNAVAILABLE to this scanner. The installed TTK exposes
  only `kernel`, `aclnn`, `e2e`, `info`, and `list`; `python -m ttk geir
  --help` exits 2 as an invalid command. Raw evidence is
  `/home/z50058944/moe_ttk_geir_subcommand_blocker.log`, SHA-256
  `02828816f5766c857884a577171be75492ec8e2a6b8d5845bcbb9697dc2202b9`.
  The ACLNN L2 route was still fully executed and all 35 probes were proven to
  be rejected by the target API or target kernel. Its generated/manifest/run
  counts are 35/35/35 with zero skip and are recorded in
  `tests/results/aclnn/negative/rejection_audit.json`, but that evidence cannot
  be substituted for the scanner's required `ttk geir` evidence format.

These limitations are reported as FAIL/REMEDIATE/UNAVAILABLE rather than
rewritten as PASS.
