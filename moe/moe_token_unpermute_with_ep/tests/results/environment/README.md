# TTK environment preflight

The `ttk-env-check` preflight was run before the operator suites, after sourcing
`/home/z50058944/.config/glacier/env.sh`. It verified the fixed TTK, Python and
CANN paths and executed the CANN-provided `aclnnAdd` connectivity case on the
device.

- Device: `Ascend950DT_9592`
- Generated/manifest/executed: `1/1/1`
- Result: `1 PASS`, `0 FAIL`, `0 SKIP`
- Archived result: `ttk_env_check_result.csv`
- Archived result SHA-256: `d56d4fb17e6c4ec37b1817741dccac9e5a6982953a38f5e4d6096a1a97c7a019`

The skill's original temporary workspace and command logs are intentionally
not included in Git.

One later operator-smoke preparation attempt was launched from the code-repo
working directory. TTK generated `warmup.o` in that directory's
`kernel_meta/`, while its configured lookup directory remained
`/home/z50058944/ops-test-kit/kernel_meta`; it therefore stopped before any
operator case ran. The generated object's SHA-256 matched the `sha256` recorded
in `warmup.json`, which isolated this as a working-directory issue. Canonical
final runs are launched from `/home/z50058944/ops-test-kit`; the aborted
preparation attempt is not counted as an operator result.
