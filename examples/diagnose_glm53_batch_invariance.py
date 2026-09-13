"""Locate the first GLM-5.3 layer that differs between B=1 and B=N.

This is an offline diagnostic for Blaizzy/mlx-vlm#2242.  It intentionally
uses a short, identical prompt in every row and no cache, separating backbone
batch-shape numerics from scheduler admission and speculative rollback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
from mlx_vlm.models.gated_delta import gated_delta_update
from mlx_vlm.models.linear import linear


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _tokens(processor, prompt: str, length: int) -> mx.array:
    tokenizer = _tokenizer(processor)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("prompt encoded to zero tokens")
    while len(token_ids) < length:
        token_ids.extend(token_ids)
    return mx.array([token_ids[:length]], dtype=mx.int32)


def _delta(single: mx.array, batched: mx.array) -> tuple[bool, float]:
    reference = batched[:1]
    mx.eval(single, reference)
    exact = bool(mx.array_equal(single, reference).item())
    maximum = float(mx.max(mx.abs(single.astype(mx.float32) - reference)).item())
    return exact, maximum


def _first_layer_detail(backbone, single, batched, single_mask, batch_mask):
    layer = backbone.layers[0]
    rows = []

    def record(site, one, many):
        exact, maximum = _delta(one, many)
        rows.append({"site": site, "exact": exact, "max_abs": maximum})

    one_c, one_post, one_comb = layer.attn_hc(single)
    many_c, many_post, many_comb = layer.attn_hc(batched)
    record("layer.0.attn_hc.collapsed", one_c, many_c)
    record("layer.0.attn_hc.post", one_post, many_post)
    record("layer.0.attn_hc.comb", one_comb, many_comb)
    one_n = layer.input_layernorm(one_c)
    many_n = layer.input_layernorm(many_c)
    record("layer.0.attn_norm", one_n, many_n)
    attn = layer.self_attn
    one_qkv = linear(attn.qkv_proj, one_n)
    many_qkv = linear(attn.qkv_proj, many_n)
    record("layer.0.linear.qkv_proj", one_qkv, many_qkv)
    one_qkv = attn.qkv_conv(one_qkv, mask=single_mask, cache=None)
    many_qkv = attn.qkv_conv(many_qkv, mask=batch_mask, cache=None)
    record("layer.0.linear.qkv_conv", one_qkv, many_qkv)
    one_shape = (1, one_n.shape[1], attn.num_heads, attn.head_dim)
    many_shape = (
        many_n.shape[0],
        many_n.shape[1],
        attn.num_heads,
        attn.head_dim,
    )
    one_q, one_k, one_v = (
        value.reshape(one_shape) for value in mx.split(one_qkv, 3, axis=-1)
    )
    many_q, many_k, many_v = (
        value.reshape(many_shape) for value in mx.split(many_qkv, 3, axis=-1)
    )
    eps = 1e-6 / attn.head_dim
    one_q = attn.scale**2 * mx.fast.rms_norm(one_q, None, eps)
    many_q = attn.scale**2 * mx.fast.rms_norm(many_q, None, eps)
    one_k = attn.scale * mx.fast.rms_norm(one_k, None, eps)
    many_k = attn.scale * mx.fast.rms_norm(many_k, None, eps)
    record("layer.0.linear.q_norm", one_q, many_q)
    record("layer.0.linear.k_norm", one_k, many_k)
    one_fbg = linear(attn.fbg_a_proj, one_n)
    many_fbg = linear(attn.fbg_a_proj, many_n)
    record("layer.0.linear.fbg_a_proj", one_fbg, many_fbg)
    cuts = (attn.head_dim, attn.head_dim + attn.num_heads)
    one_fa, one_b, one_ga = mx.split(one_fbg, cuts, axis=-1)
    many_fa, many_b, many_ga = mx.split(many_fbg, cuts, axis=-1)
    one_a = linear(attn.f_b_proj, one_fa).reshape(one_shape)
    many_a = linear(attn.f_b_proj, many_fa).reshape(many_shape)
    record("layer.0.linear.f_b_proj", one_a, many_a)
    one_b = one_b.reshape(1, one_n.shape[1], attn.num_heads)
    many_b = many_b.reshape(many_n.shape[0], many_n.shape[1], attn.num_heads)
    one_out, _ = gated_delta_update(
        one_q,
        one_k,
        one_v,
        one_a,
        one_b,
        attn.A_log.reshape(attn.num_heads, 1),
        attn.dt_bias.reshape(attn.num_heads, attn.head_dim),
        mask=single_mask,
        use_kernel=True,
        lower_bound=attn.lower_bound,
    )
    many_out, _ = gated_delta_update(
        many_q,
        many_k,
        many_v,
        many_a,
        many_b,
        attn.A_log.reshape(attn.num_heads, 1),
        attn.dt_bias.reshape(attn.num_heads, attn.head_dim),
        mask=batch_mask,
        use_kernel=True,
        lower_bound=attn.lower_bound,
    )
    record("layer.0.linear.gated_delta", one_out, many_out)
    one_gate = linear(attn.g_b_proj, one_ga).reshape(one_shape)
    many_gate = linear(attn.g_b_proj, many_ga).reshape(many_shape)
    record("layer.0.linear.g_b_proj", one_gate, many_gate)
    one_out = (attn.o_norm(one_out) * mx.sigmoid(one_gate)).reshape(
        1, one_n.shape[1], -1
    )
    many_out = (attn.o_norm(many_out) * mx.sigmoid(many_gate)).reshape(
        many_n.shape[0], many_n.shape[1], -1
    )
    record("layer.0.linear.gated_norm", one_out, many_out)
    one_out = linear(attn.o_proj, one_out)
    many_out = linear(attn.o_proj, many_out)
    record("layer.0.linear.o_proj", one_out, many_out)
    one_branch = layer.self_attn(one_n, single_mask, None)
    many_branch = layer.self_attn(many_n, batch_mask, None)
    record("layer.0.linear_attention", one_branch, many_branch)
    one_a = hc_expand(one_branch, single, one_post, one_comb)
    many_a = hc_expand(many_branch, batched, many_post, many_comb)
    record("layer.0.attn_hc.expanded", one_a, many_a)

    one_c, one_post, one_comb = layer.ffn_hc(one_a)
    many_c, many_post, many_comb = layer.ffn_hc(many_a)
    record("layer.0.ffn_hc.collapsed", one_c, many_c)
    one_n = layer.post_attention_layernorm(one_c)
    many_n = layer.post_attention_layernorm(many_c)
    record("layer.0.ffn_norm", one_n, many_n)
    one_branch = layer.mlp(one_n)
    many_branch = layer.mlp(many_n)
    record("layer.0.mlp", one_branch, many_branch)
    one_f = hc_expand(one_branch, one_a, one_post, one_comb)
    many_f = hc_expand(many_branch, many_a, many_post, many_comb)
    record("layer.0.ffn_hc.expanded", one_f, many_f)
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def _basic_layer_detail(
    layer, index, single, batched, single_mask, batch_mask, single_topk, batch_topk
):
    rows = []

    def record(site, one, many):
        exact, maximum = _delta(one, many)
        rows.append({"site": site, "exact": exact, "max_abs": maximum})

    one_c, one_post, one_comb = layer.attn_hc(single)
    many_c, many_post, many_comb = layer.attn_hc(batched)
    record(f"layer.{index}.attn_hc.collapsed", one_c, many_c)
    record(f"layer.{index}.attn_hc.post", one_post, many_post)
    record(f"layer.{index}.attn_hc.comb", one_comb, many_comb)
    one_n = layer.input_layernorm(one_c)
    many_n = layer.input_layernorm(many_c)
    record(f"layer.{index}.attn_norm", one_n, many_n)
    if layer.is_linear:
        from mlx_vlm.models.glm5_next import language as glm_language

        attn = layer.self_attn
        one_qkv = glm_language.linear(attn.qkv_proj, one_n)
        many_qkv = glm_language.linear(attn.qkv_proj, many_n)
        record(f"layer.{index}.linear.qkv_proj", one_qkv, many_qkv)
        one_conv = attn.qkv_conv(one_qkv, mask=single_mask, cache=None)
        many_conv = attn.qkv_conv(many_qkv, mask=batch_mask, cache=None)
        record(f"layer.{index}.linear.qkv_conv", one_conv, many_conv)
        one_fbg = glm_language.linear(attn.fbg_a_proj, one_n)
        many_fbg = glm_language.linear(attn.fbg_a_proj, many_n)
        record(f"layer.{index}.linear.fbg_a_proj", one_fbg, many_fbg)
        one_shape = (1, one_n.shape[1], attn.num_heads, attn.head_dim)
        many_shape = (
            many_n.shape[0],
            many_n.shape[1],
            attn.num_heads,
            attn.head_dim,
        )
        one_q, one_k, one_v = (
            value.reshape(one_shape) for value in mx.split(one_conv, 3, axis=-1)
        )
        many_q, many_k, many_v = (
            value.reshape(many_shape) for value in mx.split(many_conv, 3, axis=-1)
        )
        eps = 1e-6 / attn.head_dim
        one_q = attn.scale**2 * mx.fast.rms_norm(one_q, None, eps)
        many_q = attn.scale**2 * mx.fast.rms_norm(many_q, None, eps)
        one_k = attn.scale * mx.fast.rms_norm(one_k, None, eps)
        many_k = attn.scale * mx.fast.rms_norm(many_k, None, eps)
        record(f"layer.{index}.linear.q_norm", one_q, many_q)
        record(f"layer.{index}.linear.k_norm", one_k, many_k)
        cuts = (attn.head_dim, attn.head_dim + attn.num_heads)
        one_fa, one_b, one_ga = mx.split(one_fbg, cuts, axis=-1)
        many_fa, many_b, many_ga = mx.split(many_fbg, cuts, axis=-1)
        one_a = glm_language.linear(attn.f_b_proj, one_fa).reshape(one_shape)
        many_a = glm_language.linear(attn.f_b_proj, many_fa).reshape(many_shape)
        record(f"layer.{index}.linear.f_b_proj", one_a, many_a)
        one_b = one_b.reshape(1, one_n.shape[1], attn.num_heads)
        many_b = many_b.reshape(many_n.shape[0], many_n.shape[1], attn.num_heads)
        one_out, _ = gated_delta_update(
            one_q,
            one_k,
            one_v,
            one_a,
            one_b,
            attn.A_log.reshape(attn.num_heads, 1),
            attn.dt_bias.reshape(attn.num_heads, attn.head_dim),
            mask=single_mask,
            lower_bound=attn.lower_bound,
        )
        many_out, _ = gated_delta_update(
            many_q,
            many_k,
            many_v,
            many_a,
            many_b,
            attn.A_log.reshape(attn.num_heads, 1),
            attn.dt_bias.reshape(attn.num_heads, attn.head_dim),
            mask=batch_mask,
            lower_bound=attn.lower_bound,
        )
        record(f"layer.{index}.linear.gated_delta", one_out, many_out)
        one_gate = glm_language.linear(attn.g_b_proj, one_ga).reshape(one_shape)
        many_gate = glm_language.linear(attn.g_b_proj, many_ga).reshape(many_shape)
        record(f"layer.{index}.linear.g_b_proj", one_gate, many_gate)
        one_out = (attn.o_norm(one_out) * mx.sigmoid(one_gate)).reshape(
            1, one_n.shape[1], -1
        )
        many_out = (attn.o_norm(many_out) * mx.sigmoid(many_gate)).reshape(
            many_n.shape[0], many_n.shape[1], -1
        )
        record(f"layer.{index}.linear.gated_norm", one_out, many_out)
        one_out = glm_language.linear(attn.o_proj, one_out)
        many_out = glm_language.linear(attn.o_proj, many_out)
        record(f"layer.{index}.linear.o_proj", one_out, many_out)
        one_branch = layer.self_attn(one_n, single_mask, None)
        many_branch = layer.self_attn(many_n, batch_mask, None)
    else:
        one_branch = layer.self_attn(one_n, single_mask, None, single_topk)
        many_branch = layer.self_attn(many_n, batch_mask, None, batch_topk)
    if isinstance(one_branch, tuple):
        one_branch = one_branch[0]
        many_branch = many_branch[0]
    record(f"layer.{index}.attention", one_branch, many_branch)
    one_a = hc_expand(one_branch, single, one_post, one_comb)
    many_a = hc_expand(many_branch, batched, many_post, many_comb)
    record(f"layer.{index}.attn_hc.expanded", one_a, many_a)
    one_c, one_post, one_comb = layer.ffn_hc(one_a)
    many_c, many_post, many_comb = layer.ffn_hc(many_a)
    record(f"layer.{index}.ffn_hc.collapsed", one_c, many_c)
    one_n = layer.post_attention_layernorm(one_c)
    many_n = layer.post_attention_layernorm(many_c)
    record(f"layer.{index}.ffn_norm", one_n, many_n)
    one_branch = layer.mlp(one_n)
    many_branch = layer.mlp(many_n)
    record(f"layer.{index}.mlp", one_branch, many_branch)
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def diagnose(
    model_path: str,
    prompt: str,
    length: int,
    batch_size: int,
    *,
    max_layers: int | None = None,
    disable_singleton_hc: bool = False,
    detail_first_layer: bool = False,
    force_singleton_qmm: bool = False,
    force_singleton_hc_mix: bool = False,
    detail_layer: int | None = None,
    force_rowwise_multilinear: bool = False,
    force_rowwise_sdpa: bool = False,
    force_rowwise_conv: bool = False,
) -> dict:
    if disable_singleton_hc:
        # B=1/T>1 normally takes exact_hc_normalized_norm while B>1 takes the
        # generic HC kernel.  Make both shapes take the same generic path to
        # test whether that dispatch split seeds the first divergence.
        from mlx_vlm.models.deepseek_v4 import hyper_connection

        hyper_connection.exact_hc_normalized_norm = lambda *_args: None
    if force_singleton_qmm:
        from mlx_vlm.models.glm5_next import language as glm_language

        original_linear = glm_language.linear
        original_tiled_linear = glm_language.tiled_linear

        def stable_linear(module, value):
            if value.ndim == 3 and value.shape[0] > 1:
                return mx.concatenate(
                    [
                        original_linear(module, value[row : row + 1])
                        for row in range(value.shape[0])
                    ],
                    axis=0,
                )
            return original_linear(module, value)

        glm_language.linear = stable_linear

        def stable_tiled_linear(module, value):
            if value.ndim == 3 and value.shape[0] > 1 and value.shape[1] <= 8:
                return mx.concatenate(
                    [
                        original_tiled_linear(module, value[row : row + 1])
                        for row in range(value.shape[0])
                    ],
                    axis=0,
                )
            return original_tiled_linear(module, value)

        glm_language.tiled_linear = stable_tiled_linear
    if force_singleton_hc_mix:
        from mlx_vlm.models import linear as linear_helpers
        from mlx_vlm.models.deepseek_v4 import hyper_connection

        original_mix = hyper_connection.HyperConnection._mix

        def stable_mix(self, value):
            if value.shape[0] > 1 and value.shape[1] <= 8:
                return linear_helpers.tokenwise(
                    lambda row: linear_helpers.tokenwise(
                        lambda token: token @ self.fn.T, row
                    ),
                    value,
                    axis=0,
                )
            return original_mix(self, value)

        hyper_connection.HyperConnection._mix = stable_mix
    if force_rowwise_multilinear:
        from mlx_vlm.models import mla

        original_multilinear = mla.QuantizedMultiLinear.__call__

        def stable_multilinear(self, value, transpose=True):
            if value.shape[0] > 1:
                return mx.concatenate(
                    [
                        original_multilinear(self, value[row : row + 1], transpose)
                        for row in range(value.shape[0])
                    ],
                    axis=0,
                )
            return original_multilinear(self, value, transpose)

        mla.QuantizedMultiLinear.__call__ = stable_multilinear
    if force_rowwise_sdpa:
        from mlx_vlm.models.glm5_next import language as glm_language

        original_sdpa = glm_language.scaled_dot_product_attention

        def stable_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
            if queries.shape[0] > 1 and cache is None:
                return mx.concatenate(
                    [
                        original_sdpa(
                            queries[row : row + 1],
                            keys[row : row + 1],
                            values[row : row + 1],
                            None,
                            scale,
                            None if mask is None else mask[row : row + 1],
                            sinks,
                        )
                        for row in range(queries.shape[0])
                    ],
                    axis=0,
                )
            return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

        glm_language.scaled_dot_product_attention = stable_sdpa
    if force_rowwise_conv:
        from mlx_vlm.models.glm5_next import language as glm_language

        original_conv = glm_language.ShortConv1d.__call__

        def stable_conv(self, value, mask=None, cache=None):
            if value.shape[0] > 1 and cache is None:
                return mx.concatenate(
                    [
                        original_conv(
                            self,
                            value[row : row + 1],
                            None if mask is None else mask[row : row + 1],
                            None,
                        )
                        for row in range(value.shape[0])
                    ],
                    axis=0,
                )
            return original_conv(self, value, mask, cache)

        glm_language.ShortConv1d.__call__ = stable_conv
    model, processor = load(model_path)
    language = getattr(model, "language_model", model)
    backbone = language.model
    single_tokens = _tokens(processor, prompt, length)
    batch_tokens = mx.repeat(single_tokens, batch_size, axis=0)

    single = backbone.embed_tokens(single_tokens)
    batched = backbone.embed_tokens(batch_tokens)
    single = mx.repeat(single[:, :, None], backbone.config.hc_mult, axis=2)
    batched = mx.repeat(batched[:, :, None], backbone.config.hc_mult, axis=2)
    single_mask = mx.ones(single_tokens.shape, dtype=mx.bool_)
    batch_mask = mx.ones(batch_tokens.shape, dtype=mx.bool_)
    single_topk = None
    batch_topk = None
    rows = []

    exact, maximum = _delta(single, batched)
    rows.append({"site": "embedding", "exact": exact, "max_abs": maximum})
    detail = (
        _first_layer_detail(backbone, single, batched, single_mask, batch_mask)
        if detail_first_layer
        else []
    )
    layers = backbone.layers[:max_layers]
    for index, layer in enumerate(layers):
        if detail_layer == index:
            detail.extend(
                _basic_layer_detail(
                    layer,
                    index,
                    single,
                    batched,
                    single_mask,
                    batch_mask,
                    single_topk,
                    batch_topk,
                )
            )
        single, single_topk = layer(single, single_mask, None, single_topk)
        batched, batch_topk = layer(batched, batch_mask, None, batch_topk)
        exact, maximum = _delta(single, batched)
        topk_exact = None
        if single_topk is not None and batch_topk is not None:
            mx.eval(single_topk, batch_topk)
            topk_exact = bool(mx.array_equal(single_topk, batch_topk[:1]).item())
        row = {
            "site": f"layer.{index}",
            "type": layer.block_type,
            "exact": exact,
            "max_abs": maximum,
            "topk_exact": topk_exact,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if len(layers) == len(backbone.layers):
        single = backbone.norm(single.mean(axis=2))
        batched = backbone.norm(batched.mean(axis=2))
        exact, maximum = _delta(single, batched)
        rows.append({"site": "norm", "exact": exact, "max_abs": maximum})

        head = (
            backbone.embed_tokens.as_linear
            if language.args.tie_word_embeddings
            else language.lm_head
        )
        single_logits = linear(head, single[:, -1:])
        batch_logits = linear(head, batched[:, -1:])
        exact, maximum = _delta(single_logits, batch_logits)
        single_token = int(mx.argmax(single_logits[0, -1]).item())
        batch_token = int(mx.argmax(batch_logits[0, -1]).item())
        rows.append(
            {
                "site": "logits",
                "exact": exact,
                "max_abs": maximum,
                "single_token": single_token,
                "batch_token": batch_token,
            }
        )
    return {
        "model": str(Path(model_path).resolve()),
        "prompt_tokens": length,
        "batch_size": batch_size,
        "max_layers": max_layers,
        "disable_singleton_hc": disable_singleton_hc,
        "detail_first_layer": detail_first_layer,
        "force_singleton_qmm": force_singleton_qmm,
        "force_singleton_hc_mix": force_singleton_hc_mix,
        "detail_layer": detail_layer,
        "force_rowwise_multilinear": force_rowwise_multilinear,
        "force_rowwise_sdpa": force_rowwise_sdpa,
        "force_rowwise_conv": force_rowwise_conv,
        "detail": detail,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--prompt",
        default="Explain why deterministic local inference matters.",
    )
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--disable-singleton-hc", action="store_true")
    parser.add_argument("--detail-first-layer", action="store_true")
    parser.add_argument("--force-singleton-qmm", action="store_true")
    parser.add_argument("--force-singleton-hc-mix", action="store_true")
    parser.add_argument("--detail-layer", type=int)
    parser.add_argument("--force-rowwise-multilinear", action="store_true")
    parser.add_argument("--force-rowwise-sdpa", action="store_true")
    parser.add_argument("--force-rowwise-conv", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(
        args.model,
        args.prompt,
        args.length,
        args.batch_size,
        max_layers=args.max_layers,
        disable_singleton_hc=args.disable_singleton_hc,
        detail_first_layer=args.detail_first_layer,
        force_singleton_qmm=args.force_singleton_qmm,
        force_singleton_hc_mix=args.force_singleton_hc_mix,
        detail_layer=args.detail_layer,
        force_rowwise_multilinear=args.force_rowwise_multilinear,
        force_rowwise_sdpa=args.force_rowwise_sdpa,
        force_rowwise_conv=args.force_rowwise_conv,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
