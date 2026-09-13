# GLM-5.3 cache-owned adaptive K2/K3 spike

Date: 2026-09-13

This is a scoped research result, not a default runtime policy.  It tests
whether a request-local acceptance window can retain the second native-MTP
draft token on profitable content while falling back to one draft token when
acceptance decays.

## Configuration

- Apple M3 Ultra, 256 GiB, macOS 26.5.2
- target: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- target-matched Q4 MTP sidecar: 3.9 GiB
- temperature zero, batch one, per-task thinking budgets
- six tasks: coding, closed-book knowledge, multi-step math, exact instruction
  following, constrained creative writing, and long-context retrieval
- source base: cache-owned MTP merge plus strict Q4 `lm_head` remap at
  `ddea4644`

The candidate starts with two draft tokens.  After at least 12 complete
two-draft rounds, it falls back to one draft token if a rolling 64-round
window accepts less than 65% of proposed drafts.  It never promotes again in
the same request.  The gate is enabled only for singleton greedy decode with
`MLX_VLM_MTP_ADAPTIVE_K23=1` and block-total 3.

## Result

All fixed-width and adaptive runs passed 6/6 tasks.  Complete reasoning and
final responses were byte-identical across widths and across both adaptive
runs.

| Mode | Coding | Knowledge | Math | Instruction | Creative | Long context | Category median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed block-total 2 | 36.17 | 34.34 | 32.63 | 33.42 | 33.23 | 11.62 | 33.33 |
| fixed block-total 3 | 37.14 | 37.25 | 33.51 | 35.88 | 31.74 | 11.95 | 34.69 |
| adaptive run 1 | 36.97 | 37.20 | 33.24 | 35.86 | 32.57 | 11.98 | 34.55 |
| adaptive run 2 | 36.98 | 37.24 | 33.31 | 35.92 | 32.50 | 11.98 | 34.61 |

Relative to fixed block-total 2, adaptive run 1 improved the paired six-task
median by 1.027x.  Five categories improved; creative retained 0.980x.  Against
fixed block-total 3, adaptive improved creative by 1.026x while the other five
categories stayed within 0.9%.  The two adaptive runs reproduced within
0.23% on every category.

Fixed block-total-3 acceptance was 81.4%, 88.6%, 72.8%, 93.0%, 60.9%, and
94.8% in table order.  The adaptive gate therefore retained two drafts for
every category except creative writing.  Creative used 66 two-draft rounds,
then completed with one draft; both adaptive runs made the identical choice
and reported 202 total rounds, 268 proposed drafts, and 186 accepted drafts.

The host had unrelated `mediaanalysisd`, `fseventsd`, and pytest load during
the campaign.  Absolute throughput should be repeated on an idle host before
a release claim; the exact output, acceptance counts, adaptive choice, and
near-identical adaptive repeats were stable despite that load.

## Replay-overlap follow-up

Launching the replay-head seed with `mx.async_eval` immediately before the
verified block's final yield raised the adaptive path again without changing
its decisions:

| Mode | Coding | Knowledge | Math | Instruction | Creative | Long context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive + async run 1 | 37.63 | 37.69 | 33.88 | 36.52 | 33.05 | 12.02 |
| adaptive + async run 2 | 37.67 | 37.56 | 33.73 | 36.45 | 32.98 | 12.03 |

Against the original adaptive run, the first repeat improved the paired
six-task median 1.016x and every task was at least 1.003x.  Against the
same-source non-speculative AR artifact, the two repeats improved the paired
median 1.322x and 1.320x.  Both remained 6/6 and byte-identical.  Their
per-task round/proposal/accept counters were also exactly identical to the
original adaptive run, showing that the gain came from scheduling rather than
a changed K decision or output trajectory.

The replay launch is isolated for review on
`perf/glm53-async-mtp-replay`; unlike this adaptive policy, it has no tuned
acceptance threshold.

## Exact-prefix replay dogfood

The cache-owned MTP state also unlocks a much larger repeated-prompt win than
the decode-only figures above.  With memory APC enabled, disk APC disabled,
and two resident checkpoint entries, an English construction-contract prompt
was submitted twice through the streaming chat server:

| Prompt | Cold TTFT | Warm TTFT | Warm cached tokens | Cold / warm total |
| --- | ---: | ---: | ---: | ---: |
| 1,614 tokens | 20.952 s | 1.481 s | 1,613 | 21.143 / 1.677 s |
| 3,324 tokens | 12.397 s | 0.090 s | 3,323 | 13.572 / 1.192 s |

For the 3,324-token fixture, user-visible TTFT improved 137.9x and total request
time improved 11.39x.  The complete streamed-output SHA-256 matched between
cold and warm requests.  Server telemetry measured the warm prefill itself at
0.035 s for the 1,614-token fixture and reported the exact cached-token counts.

A 9,352-token version did not restore under the same two-entry automatic
memory budget (`cached_tokens=0`; repeated TTFT about 32.4 s).  This is an
important product boundary: claim the measured 3.3K exact-prefix result, not
unbounded prompt replay.  Longer checkpoints need a separately qualified
resident-memory policy or disk tier.

## Four-stream server dogfood

The cache-owned server can coalesce simultaneous speculative requests into a
multi-row batch.  Four identical warm-prefix requests, each capped at 256
tokens, completed in 15.416 s: 66.43 aggregate tok/s versus 40.04 tok/s for a
single request (1.66x aggregate scaling).  Every concurrent row produced the
same reasoning/output hash.

The batch trajectory was not byte-identical to the singleton trajectory at
temperature zero.  The same prompt reproduced the boundary without MTP: AR
reached 57.95 aggregate tok/s versus 27.43 tok/s singleton on the 128-token
fixture, but the dynamically admitted batch produced two output hashes (three
rows shared one and one row differed), neither guaranteed to match singleton.
This localizes the determinism gap to GLM backbone batch-size/dynamic-admission
numerics rather than speculative acceptance or rollback.  Treat cross-batch
byte invariance as a separate P1; semantic output remained sound in this
fixture, but do not advertise exact singleton/batch equality.

An environment-gated spike forced every batch-one-token affine projection
through `singleton_quantized_linear`.  It did not restore equality and reduced
AR aggregate throughput from 57.95 to 50.07 tok/s (13.6%).  The remaining
sources include GLM recurrent, MoE, and batch-state-transition paths; do not
ship a global singleton-QMM fallback.

### Layer-local batch-invariance diagnosis

A follow-up offline probe compared the same eight-token prompt at B=1 and B=4
after every GLM layer.  Stock execution was exact through the embedding, first
diverged at linear-attention layer 0 (`max_abs=0.00054931640625`), and reached
`max_abs=9.625` at layer 44 and `2.25` at the logits.  Sparse-index top-k sets
remained exact at every attention layer, excluding index selection and MoE
routing as the initial cause.

Three short-block reduction geometries explain the complete difference:

- quantized and narrow dense projections execute with a different B>1
  reduction than independent B=1 rows;
- hyperconnection `_mix` uses tokenwise B=1 matmul but batched B>1 matmul;
- fused SDPA changes its reduction geometry with batch size.

A correctness oracle that independently reused the B=1 operation for each row
at those three sites produced bit-exact output at all 45 layers, the final
norm, and the complete logits (`max_abs=0` throughout).  This proves the P1 is
locally repairable, but the row loop is not a production implementation.  The
next step is a parallel kernel with the same per-row reduction order, followed
by the dynamic-admission server repro and an aggregate-throughput gate.  Do
not trade away the measured 1.66x four-request scaling merely to obtain exact
bytes.

### Parallel decode qualification

The row-loop oracle was replaced with parallel Metal kernels for affine-Q4
linear and per-head `MultiLinear` projections.  Short-block hyperconnection
normalization now uses the existing fixed-row fused kernel for every batch
size, and sparse decode reads selected positions directly from the physical KV
cache while respecting its logical length.  No Python row loop remains in the
candidate path.

With a B=1 cache prefetched once and copied exactly to B=4, stock T=1 decode
first differed at layer 0 (`max_abs=0.000213623046875`) and reached
`0.00390625` by the first sparse-attention layer.  The parallel candidate was
bit-exact at every one of 45 layers, every sparse top-k result, the final norm,
and all 154,880 logits (`max_abs=0`, identical argmax token 43799).  This test
isolates decode arithmetic from prefill and scheduler admission.

Target-shape microbenchmarks on the same M3 Ultra showed:

- affine-Q4 `MultiLinear` (B=4): 0.290 ms native, 0.259 ms parallel, 0.308 ms
  row loop;
- sparse attention over an 8,192-token latent cache with 2,048 selected
  positions: 1.060 ms for gather plus SDPA versus 0.726 ms direct indexed
  attention.

End-to-end effects are smaller because these operators are only part of a
decode step.  On a 62-token arithmetic prompt with 128 generated tokens, warm
singleton latency improved from 3.99 s to 3.87 s (3.1%).  Four-request
aggregate throughput was neutral: 53.63 versus 53.39 tok/s.  On a
14,476-token prompt, output was byte-identical and decode was 38.2 versus 38.4
tok/s, within run noise.  Do not claim the operator microbenchmark speedups as
whole-model gains.

The server still does not promise singleton-versus-batched-prefill byte
identity: prompts longer than the short-block limit use native prefill
reductions.  Coalesced rows remained mutually identical, and coding,
knowledge, and constrained creative-writing dogfood all produced correct final
answers when run with a 128-token thinking budget.  Treat this candidate as a
decode determinism improvement with no measured concurrency regression, not a
complete resolution of dynamic-admission determinism.

## Rejected: shared cross-request cost EWMA

A follow-up reused the K1/K2 wall-time EWMA across requests while keeping
acceptance request-local.  It was rejected.  The first six-task pass reached
37.93, 38.16, 32.96, 36.27, 33.66, and 12.05 tok/s, but the second pass changed
the exact same coding trajectory from 199 rounds / 395 drafts to 250 rounds /
309 drafts.  Instruction throughput fell from 36.27 to 34.27 tok/s.  Outputs
remained byte-identical, so this was a selection-stability failure rather than
a quality failure.

The lazy replay cost can cross a request/round boundary.  Persisting that
naively attributed EWMA lets one request poison the next request's width
choice.  Do not reintroduce shared cost state until timing assigns replay work
to the depth that produced it (or uses device events that measure the intended
pipeline without forcing a synchronization).

## Rejected: reuse an accepted proposal cache prefix

A cache-level spike tried to retain the MTP cache entries produced while
drafting when the target accepted the complete proposal block, then replay
only the remaining suffix.  This reduced a three-draft toy replay from four
MTP inputs to two, but a direct baseline comparison found that the resulting
next-round seed hidden state was not equal.

Token acceptance is insufficient to make the cache entries reusable.  Draft
proposal steps advance from the previous MTP hidden state, while committed
replay must advance from the corresponding target-verified hidden states.
Those inputs differ even when every proposed token matches.  Keep the current
abort-and-replay behavior unless a future MTP architecture explicitly proves
that its proposal and committed hidden-state paths are identical.

## Rejected: fused quantized argmax for the draft head

The MTP head only consumes the winning token, so avoiding a materialized vocab
projection appeared promising.  A direct microbenchmark used the cached
target's real affine-Q4 `lm_head` geometry (4096 inputs, 154,880 outputs,
group size 64, BF16 input) and compared the current `linear` plus `argmax`
path with the optimized affine quantized-argmax kernel.  Across three 100-call
runs, current-path medians were 0.672-0.677 ms while fused-argmax medians were
0.701-0.705 ms.  Tokens matched exactly, but the candidate was 3.5-4.4%
slower.  Do not add a GLM-specific draft-head path for this kernel on the
current MLX/runtime combination.

## Target-verifier phase profile

The production-geometry benchmark's optional synchronized phase profiler
localized the remaining speculative decode cost on the same target, sidecar,
and 128-token greedy prompt:

| Mode | Throughput | Verify | Commit | Draft | Accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 25.95 tok/s | — | — | — | — |
| MTP block-total 2 | 38.66 tok/s | 89.5% | 9.1% | 0.4% | 1.0% |
| MTP block-total 3 | 36.10 tok/s | 88.5% | 6.7% | 4.0% | 0.8% |

The verifier took 39.749 ms per call at two target tokens and 51.903 ms at
three target tokens.  This makes target short-block execution, not the MTP
head, the next material optimization target.  The profiler changes execution
scheduling, so use its phase shares for localization and the unsynchronized
benchmark for release throughput claims.

Do not use a full Metal GPU capture on this 181.7 GB checkpoint.  Even a
single verifier call caused Xcode's capture layer to snapshot about 21 GB of
buffers before the workload reached the measured step.  The trace was stopped
and deleted without touching model weights or caches.  Prefer scoped operator
benchmarks and explicit phase timers on this model.

## Rejected: legacy single-token KDA chain fusion

The closed `mlx-vlm` PR #2105 reported roughly 9–10.5% decode gains before the
shared gated-delta and current GLM short-block work landed.  Its lossless
single-token Metal kernel was ported as an opt-in spike to the current source
and measured again on the real 181.7 GB checkpoint:

| Path | Current | Legacy KDA fusion | Delta | Token parity |
| --- | ---: | ---: | ---: | :---: |
| AR | 26.08 tok/s | 26.26 tok/s | +0.7% | yes |
| MTP block-total 2 | 39.81 tok/s | 39.94 tok/s | +0.3% | yes |

Both changes are inside the observed position/run noise.  A randomized
layer-level port was also no longer bit-exact against the current eager path:
the largest output difference was `9.16e-5` and the largest recurrent-state
difference was `5.39e-4`.  The old headline therefore does not transfer to the
current runtime, and its 1,100+ lines should not be revived.  Multi-token
target verification remains the relevant KDA/MoE surface.

## Reproduction

Start the server from this branch with the already-local target and sidecar:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MLX_VLM_MTP_ADAPTIVE_K23=1 \
python -m mlx_vlm.server \
  --model /path/to/target-snapshot \
  --draft-model /path/to/q4-mtp-sidecar \
  --draft-kind mtp --draft-block-size 3 \
  --host 127.0.0.1 --port 8465 --max-tokens 1024 --enable-thinking
```

Run the Rapid-MLX `scripts/benchmark_glm53_real_tasks.py` six-task gate and
compare its artifact to a fixed block-total-2 control from the same source.

## Promotion gate

Do not upstream the current environment-gated policy as a generic default.
The 65% crossover is qualified only for this Q4 GLM target and sidecar, the
policy is singleton-greedy only, and it has no recovery probe after falling
back.  A production controller should learn the per-depth cost curve, export
its selected-depth telemetry, and be tested under batched scheduling before
replacing the explicit block-size control.
