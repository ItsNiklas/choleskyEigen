import jax.numpy as jnp
from jax import abstract_arrays, core, experimental, jit, lax, xla
from jaxlib import xla_client

import choleskyEigenLib

choleskySparse_p = core.Primitive("choleskySparse")


# See solverDense.py for extensive comments.
# New bindings are just copied and adjusted.

# Symbolic Factorization of the Sparse Cholesky Decomposition #
# by @GianmarcoCallegher                                      #


@jit
def symbolic_factorization(nnz_m):
    def parent(k, nnz_m):
        p = -1
        i = k + 1

        def cond_fun(carry):
            i, p = carry
            return (p == -1) & (i < nnz_m.shape[0])

        def body_fun(carry):
            i, p = carry
            p = jnp.where(nnz_m[(i, k)] == 1.0, i, p)
            return i + 1, p

        return lax.while_loop(cond_fun, body_fun, (i, p))[1]

    n = nnz_m.shape[0]

    def outer_body(k, nnz_m):
        nnz_m = nnz_m.at[(k, k)].set(1.0)
        parent_k = parent(k, nnz_m)

        def inner_body(i, nnz_m):
            return jnp.where(
                nnz_m[(i, k)] == 1.0, nnz_m.at[(i, parent_k)].set(1.0), nnz_m
            )

        return lax.fori_loop(k + 1, n, inner_body, nnz_m)

    return jnp.sum(lax.fori_loop(0, n, outer_body, nnz_m), dtype=jnp.int32)


###

# Not jit-able!
def choleskySparse(A_sp):
    nnz_m = jnp.array(A_sp.todense() != 0, dtype=jnp.float32)
    nnz_m = nnz_m - jnp.triu(nnz_m)
    nnz = symbolic_factorization(nnz_m)
    n = A_sp.shape[0]

    outer_idx, inner_idx, data = choleskySparse_prim(
        A_sp.data, A_sp.indices, jnp.repeat(n, n), jnp.repeat(nnz, nnz)
    )
    idx = jnp.hstack((inner_idx, outer_idx))

    return experimental.sparse.BCOO((data.flatten(), idx), shape=(n, n))


def register():
    # Register module
    xla.backend_specific_translations["cpu"][
        choleskySparse_p
    ] = choleskySparse_xla_translation
    choleskySparse_p.multiple_results = True
    choleskySparse_p.def_impl(choleskySparse_impl)
    choleskySparse_p.def_abstract_eval(choleskySparse_abstract_eval)


# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def choleskySparse_impl(*args):
    raise NotImplementedError("Please JIT this function.")


# prim
@jit
def choleskySparse_prim(*args):
    return choleskySparse_p.bind(*args)


# abstract
def choleskySparse_abstract_eval(A_sp_data, A_sp_idx, n, nnz_L):
    assert len(A_sp_data.shape) == 1
    assert len(A_sp_idx.shape) == 2
    assert A_sp_data.shape[0] == A_sp_idx.shape[0]
    # Returning tuple of outputs
    return (
        abstract_arrays.ShapedArray((nnz_L.shape[0], 1), A_sp_idx.dtype),
        abstract_arrays.ShapedArray((nnz_L.shape[0], 1), A_sp_idx.dtype),
        abstract_arrays.ShapedArray((nnz_L.shape[0], 1), A_sp_data.dtype),
    )


def choleskySparse_xla_translation(c, A_sp_data, A_sp_idx, n, nnz_L):
    A_sp_data_shape = c.get_shape(A_sp_data)
    A_sp_idx_shape = c.get_shape(A_sp_idx)

    dtype = A_sp_data_shape.element_type()
    dtype_idx = A_sp_idx_shape.element_type()

    A_sp_data_dims = A_sp_data_shape.dimensions()
    A_sp_data_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype), A_sp_data_dims, (0,)
    )

    A_sp_idx_dims = A_sp_idx_shape.dimensions()
    A_sp_idx_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype_idx), A_sp_idx_dims, (0, 1)
    )

    n_dims = c.get_shape(n).dimensions()

    out_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype), (n_dims[0], n_dims[0]), (0, 1)
    )

    idx_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype_idx), (c.get_shape(nnz_L).dimensions()[0], 1), (0, 1)
    )
    data_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype), (c.get_shape(nnz_L).dimensions()[0], 1), (0, 1)
    )

    out_shape = xla_client.Shape.tuple_shape((idx_shape, idx_shape, data_shape))

    op_name = b"choleskySparse"

    nnz = A_sp_data_dims[0]

    r = xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(
            A_sp_data,
            A_sp_idx,
            xla_client.ops.ConstantLiteral(c, nnz),
            n,
        ),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(
            A_sp_data_shape,
            A_sp_idx_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            c.get_shape(n),
        ),
    )
    return r
