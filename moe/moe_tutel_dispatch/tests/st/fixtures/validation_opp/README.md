# TTK schema-validation fixture

This directory contains only the minimum `op-info` metadata needed by
`python -m ttk <kernel|geir> --validate` before the operator package exists.
It is not a deployable OPP and must never be used as execution evidence.

Set `ASCEND_CUSTOM_OPP_PATH` to this `validation_opp` directory only for CSV
schema validation. CP4/CP6 execution must use the real built and deployed
Ascend950 package; `SOC_NOT_SUPPORT` under the stock package is expected before
that deployment.
