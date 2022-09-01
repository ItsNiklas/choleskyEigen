import jax.numpy as jnp
from jax import abstract_arrays, core, dtypes, experimental, xla
from jaxlib import xla_client

import choleskyEigenLib

sps_mvn_sample_and_log_prob_p = core.Primitive("sps_mvn_sample_and_log_prob")


# See solverDense.py for extensive comments.
# New bindings are just copied and adjusted.


def sps_mvn_sample_and_log_prob(mean, inv_cov, sample, log_prob):
    r = sps_mvn_sample_and_log_prob_prim(
        mean, inv_cov.data, inv_cov.indices, sample, log_prob
    )
    # Possibly turn into BCOO
    return r


def register():
    # Register module
    xla.backend_specific_translations["cpu"][
        sps_mvn_sample_and_log_prob_p
    ] = sps_mvn_sample_and_log_prob_xla_translation
    sps_mvn_sample_and_log_prob_p.def_impl(sps_mvn_sample_and_log_prob_impl)
    sps_mvn_sample_and_log_prob_p.def_abstract_eval(
        sps_mvn_sample_and_log_prob_abstract_eval
    )


# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def sps_mvn_sample_and_log_prob_impl(*args):
    raise NotImplementedError("Please JIT this function.")


# prim
def sps_mvn_sample_and_log_prob_prim(mean, inv_cov_data, inv_cov_idx, sample, log_prob):
    return sps_mvn_sample_and_log_prob_p.bind(
        mean, inv_cov_data, inv_cov_idx, sample, log_prob
    )


# abstract
def sps_mvn_sample_and_log_prob_abstract_eval(
    mean, inv_cov_data, inv_cov_idx, sample, log_prob
):
    # Missing asserts

    # Returning tuple of outputs
    dt = dtypes.canonicalize_dtype(sample.dtype)
    dtlog = dtypes.canonicalize_dtype(log_prob.dtype)
    a = abstract_arrays.ShapedArray(sample.shape, dt)
    b = abstract_arrays.ShapedArray(log_prob.shape, dtlog)
    return a, b


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
