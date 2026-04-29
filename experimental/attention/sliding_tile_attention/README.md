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
