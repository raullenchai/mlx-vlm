"""MTP proposal / target verification / prefix acceptance / cache commit."""

import logging
import os
import time
from functools import partial

import mlx.core as mx

from ..models.cache import BatchKVCache, BatchQuantizedKVCache
from .adaptive import K23AcceptanceGate, K23EvController
from .cache_state import SpeculativeCache, iter_leaf_caches
from .sampling import accept_greedy, accept_sampled

logger = logging.getLogger(__name__)


def _adaptive_k23_enabled():
    return os.environ.get("MLX_VLM_MTP_ADAPTIVE_K23", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _ev_k23_enabled():
    return os.environ.get("MLX_VLM_MTP_EV_K23", "").lower() in {
        "1",
        "true",
        "yes",
    }


def mtp_rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    prompt_tokens,
    first_bonus,
    max_tokens,
    sampler,
    draft_block_size=None,
    token_dtype=mx.int32,
    stop_check=None,
    eos_token_ids=None,
    greedy_sampling=False,
    row_ids=None,
    phase_observer=None,
    logits_processors=None,
    token_context=None,
    state=None,
    compute_logprobs=False,
):
    """Yield one token per live row. ``max_tokens`` includes ``first_bonus``.

    The drafter is deterministic, so matching a draw from the target is exact
    rejection sampling for a point-mass proposal. Sampling needs no draft RNG.
    Both batch-one and ragged batches use this same loop.
    """
    batch = first_bonus.size
    limits = [max_tokens] * batch if isinstance(max_tokens, int) else list(max_tokens)
    if len(limits) != batch:
        raise ValueError("max_tokens must provide one limit per row.")
    if max(limits, default=0) <= 1:
        return
    count = (
        draft_model.config.block_size if draft_block_size is None else draft_block_size
    ) - 1
    if count < 1:
        raise ValueError(
            "MTP block size must be at least 2 (one draft plus one target token)."
        )
    immediate_yield = any(
        getattr(processor, "requires_immediate_decode_yield", False)
        for processors in logits_processors or []
        for processor in processors or []
    )
    ev_controller = (
        K23EvController()
        if _ev_k23_enabled()
        and count == 2
        and batch == 1
        and greedy_sampling
        and not immediate_yield
        else None
    )
    adaptive = (
        K23AcceptanceGate()
        if ev_controller is None
        and _adaptive_k23_enabled()
        and count == 2
        and batch == 1
        and greedy_sampling
        and not immediate_yield
        else None
    )
    target = getattr(model, "language_model", model)
    row_ids = list(range(batch)) if row_ids is None else row_ids
    padding = None
    for entry in iter_leaf_caches(prompt_cache):
        if isinstance(entry, (BatchKVCache, BatchQuantizedKVCache)):
            padding = entry.left_padding.tolist()
            break
    if batch > 1 and padding is None:
        raise ValueError("Batched MTP requires batch prompt caches.")
    forward = partial(draft_model, target_model=target)
    if phase_observer:
        phase_observer("start", [])
    if state is None:
        state = SpeculativeCache.create(prompt_cache, draft_model, batch)
        state.bonus = first_bonus.astype(token_dtype).reshape(-1, 1)
        state.prefill(prompt_tokens, hidden, forward)
    contexts = (
        token_context
        if token_context is not None
        else [
            row[(padding[i] if padding else 0) :]
            for i, row in enumerate(prompt_tokens.tolist())
        ]
    )
    state.tokens = [
        list(context) + [token]
        for context, token in zip(contexts, first_bonus.reshape(-1).tolist())
    ]
    if phase_observer:
        phase_observer("draft_prefill", [state.seed.token, state.seed.hidden])
    produced = [1] * batch
    stopped = [False] * batch
    eos = eos_token_ids or set()
    for row, token in enumerate(first_bonus.reshape(-1).tolist()):
        stopped[row] = token in eos or bool(stop_check and stop_check(row, token))

    try:
        while any(
            not done and n < limit for done, n, limit in zip(stopped, produced, limits)
        ):
            budgets = [
                0 if stopped[i] else max(0, limits[i] - n)
                for i, n in enumerate(produced)
            ]
            available_depth = min(count, max(budgets) - 1)
            depth = 0 if immediate_yield else available_depth
            controller = ev_controller or adaptive
            if controller is not None:
                depth = min(controller.pick(), available_depth)
            stats_before = (
                state.stats[0].snapshot() if controller is not None else None
            )
            compute_started = time.perf_counter() if ev_controller is not None else None
            proposals = state.propose(depth, forward)
            if phase_observer:
                phase_observer("draft", [proposals])
            output = target(
                state.verify_inputs(proposals),
                cache=state.target,
                return_hidden=True,
                position_ids=state.positions(proposals.shape[1] + 1),
            )
            state.record_verification(output.hidden_states[-1])
            if phase_observer:
                phase_observer("verify", [output.logits, output.hidden_states[-1]])
            if greedy_sampling and not (logits_processors and any(logits_processors)):
                accepted = accept_greedy(
                    proposals, output.logits, budgets, compute_logprobs=compute_logprobs
                )
            else:
                accepted = accept_sampled(
                    proposals,
                    output.logits,
                    budgets,
                    sampler,
                    row_ids,
                    produced,
                    processors=logits_processors,
                    contexts=state.tokens,
                    greedy=greedy_sampling,
                    compute_logprobs=compute_logprobs,
                )
            round_compute_ms = (
                (time.perf_counter() - compute_started) * 1000.0
                if compute_started is not None
                else 0.0
            )
            rows = accepted.tokens
            if phase_observer:
                phase_observer("accept", [])
            for row, values in enumerate(rows):
                for pos, token in enumerate(values):
                    if token in eos or (stop_check and stop_check(row, token)):
                        rows[row] = values[: pos + 1]
                        stopped[row] = True
                        break
            emitted = [[] for _ in rows]
            width = max(map(len, rows))
            committed = False
            try:
                for pos in range(width):
                    tokens = []
                    for row, values in enumerate(rows):
                        token = values[pos] if pos < len(values) else None
                        tokens.append(token)
                        if token is not None:
                            emitted[row].append(token)
                            produced[row] += 1
                    if pos + 1 == width:
                        state.commit(emitted, forward)
                        has_next_round = any(
                            not done and n < limit
                            for done, n, limit in zip(stopped, produced, limits)
                        )
                        if controller is not None and has_next_round:
                            # The replay head feeds only the next round.  Start
                            # it before yielding the final token so device work
                            # overlaps the caller's detokenization / framing.
                            mx.async_eval(
                                state.seed.token,
                                state.seed.hidden,
                                state.bonus,
                            )
                        if controller is not None and stats_before is not None:
                            stats_after = state.stats[0].snapshot()
                            record_args = {
                                "depth": depth,
                                "accepted": stats_after[1] - stats_before[1],
                                "drafted": stats_after[2] - stats_before[2],
                            }
                            if ev_controller is not None:
                                # Acceptance has synchronized this round's
                                # proposal + target verification.  Do not force
                                # the replay head here: doing so changes the
                                # pipeline being measured.  Any lazy replay work
                                # that remains is naturally paid at the next
                                # round's proposal boundary.
                                record_args["wall_ms"] = round_compute_ms
                            controller.record(**record_args)
                        if phase_observer:
                            phase_observer(
                                "commit", [state.seed.token, state.seed.hidden]
                            )
                        committed = True
                    yield tokens, {
                        "round_pos": pos,
                        "round_len": width,
                        "logprobs": (
                            [
                                (
                                    accepted.logprobs[row][pos]
                                    if token is not None
                                    else None
                                )
                                for row, token in enumerate(tokens)
                            ]
                            if accepted.logprobs is not None
                            else None
                        ),
                    }
            finally:
                if not committed:
                    state.commit(emitted, forward)
    finally:
        state.abort()
        if adaptive is not None:
            logger.info("MTP adaptive K2/K3: %s", adaptive.summary())
        if ev_controller is not None:
            logger.info("MTP EV K2/K3: %s", ev_controller.summary())
