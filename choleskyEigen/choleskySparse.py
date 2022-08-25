import choleskyEigenLib
import jax.numpy as jnp
from jax import abstract_arrays, core, experimental, xla
from jaxlib import xla_client

choleskySparse_p = core.Primitive("choleskySparse")
# See solverDense.py for extensive comments.
# New bindings are just copied and adjusted.


def choleskySparse(A_sp):
    r = choleskySparse_prim(A_sp.data, A_sp.indices, A_sp)
    # Possibly turn into BCOO
    return r


def register():
    # Register module
    xla.backend_specific_translations["cpu"][
        choleskySparse_p
    ] = choleskySparse_xla_translation
    choleskySparse_p.def_impl(choleskySparse_impl)
    choleskySparse_p.def_abstract_eval(choleskySparse_abstract_eval)


# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def choleskySparse_impl(A_sp_data, A_sp_idx, A_sp_shape):
    raise NotImplementedError("Please JIT this function.")


# prim
def choleskySparse_prim(A_sp_data, A_sp_idx, A_sp):
    return choleskySparse_p.bind(
        A_sp_data, A_sp_idx, A_sp.todense()
    )  # todense() cheat :(


# abstract
def choleskySparse_abstract_eval(A_sp_data, A_sp_idx, A):
    assert len(A_sp_data.shape) == 1
    assert len(A_sp_idx.shape) == 2
    assert A_sp_data.shape[0] == A_sp_idx.shape[0]
    return abstract_arrays.ShapedArray(A.shape, A_sp_data.dtype)
    # TODO Output shape?


def choleskySparse_xla_translation(c, A_sp_data, A_sp_idx, A):
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

    out_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype), c.get_shape(A).dimensions(), (0, 1)
    )
    # sh = xla_client.Shape.tuple_shape([xla_client.Shape.token_shape()])

    op_name = b"choleskySparse"

    n = c.get_shape(A).dimensions()[0]
    nnz = A_sp_data_dims[0]

    r = xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(
            A_sp_data,
            A_sp_idx,
            xla_client.ops.ConstantLiteral(c, nnz),
            xla_client.ops.ConstantLiteral(c, n),
        ),
        shape_with_layout=out_shape,
        # shape=sh,
        operand_shapes_with_layout=(
            A_sp_data_shape,
            A_sp_idx_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
        ),
    )
    # return xla_client.ops.GetTupleElement(r, 0)
    return r
