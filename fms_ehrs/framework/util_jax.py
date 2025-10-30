#!/usr/bin/env python3

"""
jax-based utilities
"""

import functools
import typing

import jax as jx
import jax.numpy as jnp
import numpy as np

# gpu --> OOM
jx.config.update("jax_default_device", jx.devices("cpu")[0])


@jx.jit
def attention_rollout(attentions: jnp.ndarray) -> jnp.ndarray:
    return jx.lax.associative_scan(
        lambda x, y: y @ x,
        0.5
        * (
            attentions
            + jnp.tile(
                jnp.eye(attentions.shape[-1]), reps=(*attentions.shape[:3], 1, 1)
            )
        ),
    )[-1]


@functools.partial(
    jx.jit, static_argnames=("window", "aggregation", "last_layer_only", "rollout")
)
def token_importance(
    attentions: jnp.ndarray,
    values: jnp.ndarray = None,
    aggregation: typing.Literal["mean", "mean_log"] = "mean",
    window: int = None,
    last_layer_only: bool = False,
    rollout: bool = False,
) -> jnp.ndarray:
    if last_layer_only:
        a = attentions[-1][jnp.newaxis]
        v = values[-1][jnp.newaxis] if values is not None else 1
    elif rollout:
        a = attention_rollout(attentions)[jnp.newaxis]
        v = values
    else:
        a = attentions
        v = values
    lookahead_arr = (
        (jnp.tri(a.shape[-1]) - jnp.tri(a.shape[-1], k=-window))
        if window is not None
        else 1
    )
    v = jnp.linalg.norm(v, axis=-1, ord=1, keepdims=True) if values is not None else 1
    if "log" in aggregation:
        return jnp.nanmean(jnp.log(jnp.sum(a * lookahead_arr * v, axis=3)), axis=(0, 2))
    return jnp.nanmean(jnp.sum(a * lookahead_arr * v, axis=3), axis=(0, 2))


if __name__ == "__main__":
    import time

    from fms_ehrs.framework.util import attention_rollout as attention_rollout_orig
    from fms_ehrs.framework.util import token_importance as token_importance_orig

    rng = np.random.default_rng(42)
    att_eg = np.tril(rng.uniform(size=(1000, 8, 16, 4, 32, 32)))

    t0 = time.time()
    for i in range(1000):
        rll = attention_rollout_orig(att_eg[i])
    t1 = time.time()
    print("rollout: {:.2f}".format((t1 - t0)))

    t2 = time.time()
    for i in range(1000):
        rll_jax = attention_rollout(att_eg[i])
    t3 = time.time()
    print("jax rollout: {:.2f}".format((t3 - t2)))

    @jx.jit
    def attention_rollout_jax_alt(attentions: jnp.ndarray) -> jnp.ndarray:
        I = jnp.tile(jnp.eye(attentions.shape[-1]), reps=(*attentions.shape[:3], 1, 1))
        ret = 0.5 * (attentions[0] + I)
        for i in range(1, attentions.shape[0]):
            ret = 0.5 * (attentions[i] + I) @ ret
        return ret

    t4 = time.time()
    for i in range(1000):
        rll_jax_alt = attention_rollout_jax_alt(att_eg[i])
    t5 = time.time()
    print("alt jax rollout: {:.2f}".format((t5 - t4)))

    assert np.allclose(rll, rll_jax, rtol=1e-3, atol=1e-1)
    assert np.allclose(rll, rll_jax_alt, rtol=1e-2, atol=1e-1)

    t6 = time.time()
    for i in range(1000):
        tk_imp = token_importance_orig(att_eg[i])
    t7 = time.time()
    print("h20: {:.2f}".format((t7 - t6)))

    t8 = time.time()
    for i in range(1000):
        tk_imp_jax = token_importance(att_eg[i])
    t9 = time.time()
    print("jax h20: {:.2f}".format((t9 - t8)))

    assert np.allclose(tk_imp, tk_imp_jax, rtol=1e-4, atol=1e-5)

    t10 = time.time()
    for i in range(1000):
        tk_imp_sh = token_importance_orig(att_eg[i], window=10)
    t11 = time.time()
    print("sh: {:.2f}".format((t11 - t10)))

    t12 = time.time()
    for i in range(1000):
        tk_imp_sh_jax = token_importance(att_eg[i], window=10)
    t13 = time.time()
    print("jax h20: {:.2f}".format((t13 - t12)))

    assert np.allclose(tk_imp_sh, tk_imp_sh_jax, rtol=1e-4, atol=1e-5)

    assert np.allclose(
        token_importance(att_eg[-1], window=10, aggregation="mean_log"),
        token_importance_orig(att_eg[-1], window=10, aggregation="mean_log"),
        rtol=1e-4,
        atol=1e-5,
    )
