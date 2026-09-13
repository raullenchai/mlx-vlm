# GLM-5.3 asynchronous MTP replay spike

Date: 2026-09-13

## Change

The MTP cache commit builds the small replay-head graph before yielding the
last token in a verified block.  Previously that graph remained lazy until the
generator was resumed, leaving the device idle while the caller detokenized,
checked stops, and framed the streamed token.  This spike calls
`mx.async_eval` on the next-round seed before the yield.  It does not add a
synchronization or change token selection.

## Configuration

- Apple M3 Ultra, 256 GiB, macOS 26.5.2
- target: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- target-matched 3.9 GiB Q4 MTP sidecar
- fixed block-total 3, temperature zero, batch one
- six graded tasks: coding, knowledge, math, instruction following, creative
  writing, and long-context retrieval
- source base: cache-owned MTP conflict-resolution merge plus strict quantized
  `lm_head` remap at `ddea4644`

No model was downloaded or relocated during the run.

## Result

| Mode | Coding | Knowledge | Math | Instruction | Creative | Long context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synchronous/lazy control | 37.14 | 37.25 | 33.51 | 35.88 | 31.74 | 11.95 |
| async replay run 1 | 37.50 | 37.97 | 33.46 | 36.43 | 32.14 | 12.02 |
| async replay run 2 | 37.44 | 37.68 | 33.99 | 35.90 | 31.78 | 11.96 |

Against the same-source control, the paired six-task median improved 1.011x in
run 1 and 1.005x in run 2.  The slowest individual ratio was 0.999x in run 1
and 1.001x in run 2.  Both candidates passed 6/6 graders, and their complete
reasoning and final outputs were byte-identical to the control and to one
another.

The same launch composed with the request-local K2/K3 acceptance experiment
improved its paired median 1.016x, with all six task ratios at least 1.003x.
That combined path reached a paired median 1.322x over non-speculative AR in
run 1 and 1.320x in run 2, again with byte-identical outputs.  The adaptive
controller remains a separate experiment; these numbers do not make its 65%
threshold a generic policy.

The host still had unrelated `fseventsd` load.  Two repeats bound the observed
effect, but the approximately 0.5-1.1% fixed-width gain should be treated as a
small pipeline improvement rather than a release headline.

## Verification

- `python -m pytest mlx_vlm/tests/test_speculative.py mlx_vlm/tests/test_server.py -q`
  — 405 passed
- Ruff on the changed source and test files
- `git diff --check`
- two complete six-task E2E runs, each 6/6 with exact output comparison

