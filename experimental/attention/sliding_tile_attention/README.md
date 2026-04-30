# SlidingTileAttention

SlidingTileAttention is the BNSD forward operator used by FastVideo STA kernels. It computes scaled dot-product attention with a 3D tiled image window and optional global text tokens.

## Interface

- Inputs: `q`, `k`, `v`, all BNSD `[batch, heads, seq, dim]`, `float16` or `bfloat16`.
- Output: `y`, same shape and dtype as `q`.
- Attrs:
  - `window_size`: required `ListInt`. Accepts `[t, h, w]` broadcast to every head, or flattened per-head triples `[h0_t, h0_h, h0_w, h1_t, ...]`.
  - `text_length`: optional `Int`, default `0`.
  - `has_text`: optional `Bool`, default `true`.
  - `seq_shape`: optional `String`, default `30x48x80`.

## Hardware

This implementation registers AICore configs for `ascend910b` and `ascend910_93`, covering Atlas A2 training series targets used by the task.

## Current Kernel

The first implementation is a correctness-oriented scalar Ascend C path:

1. Map each core to a strided set of `(batch, head, query)` tokens.
2. For image queries, locate the 6x8x8 tile and clamp the configured 3D tile window to image tile bounds.
3. For text queries, attend to all real image and text tokens.
4. Compute stable softmax with a max pass, denominator pass, and value accumulation pass.

This gives a functional baseline for integration and golden comparison. Performance work should replace the scalar per-query loops with UB-resident score blocks, vectorized dot/exp/reduce, and double-buffered V accumulation.

## Tests

The pytest reference test always validates the CPU golden implementation. On a machine with `torch_npu` and a registered Python entry for this operator, it also runs the NPU operator and compares it with the CPU result:

```bash
python3 -m pytest -rA -s experimental/attention/sliding_tile_attention/tests/test_sliding_tile_attention_reference.py
```

If the Python binding uses a non-default name, set `SLIDING_TILE_ATTENTION_NPU_OP` to a dotted callable path, for example:

```bash
SLIDING_TILE_ATTENTION_NPU_OP=torch.ops.npu_ops_transformer_ext.sliding_tile_attention python3 -m pytest -rA -s experimental/attention/sliding_tile_attention/tests/test_sliding_tile_attention_reference.py -m npu
```

For a direct CANN ACLNN launch, build and install the custom operator package, source the generated custom OPP environment, then run the eager example:

```bash
bash build.sh --pkg --experimental --soc=ascend910b --ops=sliding_tile_attention
./build_out/cann-ops-transformer-*linux*.run --install-path=${install_path}
source ${install_path}/vendors/custom_transformer/bin/set_env.bash
bash build.sh --run_example sliding_tile_attention eager cust --soc=ascend910b --vendor_name=custom
```

The ACLNN example calls `aclnnSlidingTileAttentionGetWorkspaceSize` and `aclnnSlidingTileAttention`, copies the NPU output back to host, and checks it against an in-process CPU golden.
