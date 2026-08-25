# Ascend950 custom-kernel provenance

The implementation baseline is `3951ec00487871ea02568d2b8566fe3798bd6b20`.
The final test run also includes the independent follow-up fixes `f04eecc40`,
`b0d50e8a1`, `5d71404bd`, and the platform-isolation correction
`289067d40`; no commit was amended or rewritten.

The validated run package has SHA-256
`596cb40f6e33ccb51a6c7da4226984715afe3f22c83030dc4d6fffd57eb435e3`
and is installed at
`/home/z50058944/Ascend/moe_token_unpermute_with_ep_validation_isolation_289067d40`.
The repository and installed `_apt.cpp` both hash to
`2a55a9090c0f6605cbed53772304934cc1c0ce377cfc2d0456bf304366aa7030`.

ACLNN debug plog
`/home/z50058944/ascend/log/debug/plog/plog-1784582_20260825134440191.log`
records the isolated custom object in `SelectBin` at line 75425, registers the
same path at line 90153, and launches tiling key 3 at line 90158. The three
installed object hashes are recorded in `kernel_provenance.json`.

GEIR debug plogs independently prove both dynamic descriptors:

- `unknown_dim_minus_1`: isolated custom root at line 81018, custom DSL path
  at line 83291, custom dynamic module at line 83364, and three launches at
  lines 90082/90367/90654 of
  `plog-1843544_20260825135455036.log`.
- `unknown_rank_minus_2`: isolated custom root at line 8410, custom DSL path
  at line 10643, custom dynamic module at line 10713, and three launches at
  lines 16819/17105/17392 of
  `plog-1843544_20260825135456926.log`.

GEIR compiles the installed source into ATC cache objects; their SHA-256 values
and paths are recorded in the JSON evidence. The TTK Kernel bridge under
`tests/ttk_bridge/` supplies only the signature lookup required by this TTK
version and raises if invoked as a compute implementation. It does not provide
or emulate the tested kernel.

The platform-isolation audit additionally proves that non-3510 selected source
branches are line-equivalent to `origin/master`. A post-correction Ascend910B
package compile generated the automatic ACLNN source and all three legacy
Kernel objects without compiling the 950 manual API or defining the 950 Host
macro. Ascend910B runtime was not executed because no Ascend910B device is
assigned to this environment.

Raw build, run, device, and plog files remain outside Git. The committed JSON
keeps exact paths, line numbers, hashes, and final-result hashes so the route
can be audited without mistaking the system builtin for the custom `_apt`
implementation.
