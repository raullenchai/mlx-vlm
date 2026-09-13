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
