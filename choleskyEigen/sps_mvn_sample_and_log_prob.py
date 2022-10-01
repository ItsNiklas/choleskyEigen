from functools import partial

import jax
import jax.numpy as jnp
from jax import abstract_arrays, core, xla
from jax.experimental.sparse import BCOO
from jaxlib import xla_client

import choleskyEigenLib

sps_mvn_sample_and_log_prob_p = core.Primitive("sps_mvn_sample_and_log_prob")


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def sps_mvn(seed, mean, inv_cov):
    # This is akward, but the gradient does not like the BCOO matrix.
    # It can be sparsified, but then the attributes .data and .indices below
    # become unavailable. This is the simplest solution I've come up with,
    # which stores the whole memory, and the method below cannot be included
    # in JIT-compilation.
    inv_cov = BCOO.fromdense(inv_cov)  # dense to sparse
    # do the following steps in c++
    return jax.jit(sps_mvn_sample_and_log_prob_prim)(
        mean,
        inv_cov.data,
        inv_cov.indices.T,
        jax.random.normal(seed, mean.shape),  # sample
        jnp.sum(
            jax.scipy.stats.norm.logpdf(jax.random.normal(seed, mean.shape))
        ),  # log-prob
        # Transpose fixes some other bug (?)
    )


@sps_mvn.defjvp
def sps_mvn_jvp(seed, primals, tangents):
    mean, inv_cov = primals
    mean_dot, inv_cov_dot = tangents

    sample, log_prob = sps_mvn(seed, mean, inv_cov)
    primals_out = (sample, log_prob)

    # do the following steps in c++... or not
    # Maybe XLA is enough?
    # Not sparse tough...
    @jax.jit
    def __f():
        tmp1 = sample - mean
        tmp2 = inv_cov @ tmp1
        dot1 = mean_dot @ tmp2
        dot2 = jnp.trace(jnp.linalg.solve(inv_cov, inv_cov_dot))
        dot3 = tmp1 @ inv_cov_dot @ tmp1
        return jnp.zeros_like(sample), dot1 + (dot2 - dot3) / 2

    tangents_out = __f()
    return primals_out, tangents_out


# See solverDense.py for extensive comments.
# New bindings are just copied and adjusted.


def register():
    # Register module
    sps_mvn_sample_and_log_prob_p.def_impl(sps_mvn_sample_and_log_prob_impl)
    # Undocumented support for tuple return!
    # TypeError: <class 'tuple'> otherwise.
    sps_mvn_sample_and_log_prob_p.multiple_results = True
    sps_mvn_sample_and_log_prob_p.def_abstract_eval(
        sps_mvn_sample_and_log_prob_abstract_eval
    )
    xla.backend_specific_translations["cpu"][
        sps_mvn_sample_and_log_prob_p
    ] = sps_mvn_sample_and_log_prob_xla_translation


# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def sps_mvn_sample_and_log_prob_impl(*args):
    raise NotImplementedError("Please JIT this function.")


# prim
def sps_mvn_sample_and_log_prob_prim(*args):
    return sps_mvn_sample_and_log_prob_p.bind(*args)


# abstract
def sps_mvn_sample_and_log_prob_abstract_eval(
    mean, inv_cov_data, inv_cov_idx, sample, log_prob
):
    # Missing asserts

    # Returning tuple of outputs
    return (
        abstract_arrays.ShapedArray(sample.shape, sample.dtype),
        abstract_arrays.ShapedArray(log_prob.shape, log_prob.dtype),
    )


def sps_mvn_sample_and_log_prob_xla_translation(
    c, mean, inv_cov_data, inv_cov_idx, sample, log_prob
):
    mean_shape = c.get_shape(mean)
    inv_cov_data_shape = c.get_shape(inv_cov_data)
    inv_cov_idx_shape = c.get_shape(inv_cov_idx)
    sample_shape = c.get_shape(sample)
    log_prob_shape = c.get_shape(log_prob)

    out_shape = xla_client.Shape.tuple_shape((sample_shape, log_prob_shape))

    op_name = b"sps_mvn_sample_and_log_prob"

    n = sample_shape.dimensions()[0]
    nnz = inv_cov_data_shape.dimensions()[0]

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(
            mean,
            inv_cov_data,
            inv_cov_idx,
            sample,
            log_prob,
            xla_client.ops.ConstantLiteral(c, n),
            xla_client.ops.ConstantLiteral(c, nnz),
        ),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(
            mean_shape,
            inv_cov_data_shape,
            inv_cov_idx_shape,
            sample_shape,
            log_prob_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
        ),
    )
