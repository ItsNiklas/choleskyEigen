import choleskyEigenLib
import jax.numpy as jnp
from jax import abstract_arrays, core, xla
from jaxlib import xla_client

solverSparse_p = core.Primitive("solverSparse")
# See solverDense.py for extensive comments.
# New bindings are just copied and adjusted.


def register():
    # Register module
    xla.backend_specific_translations["cpu"][
        solverSparse_p
    ] = solverSparse_xla_translation
    solverSparse_p.def_impl(solverSparse_impl)
    solverSparse_p.def_abstract_eval(solverSparse_abstract_eval)


# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def solverSparse_impl(A_sp, rhs):
    raise NotImplementedError("Please JIT this function.")


# prim
def solverSparse_prim(A_sp, rhs):
    return solverSparse_p.bind(A_sp.data, A_sp.indices, rhs)


# abstract
def solverSparse_abstract_eval(A_sp_datas, A_sp_idxs, rhss):
    assert len(A_sp_datas.shape) == 1
    assert len(rhss.shape) == 1
    assert len(A_sp_idxs.shape) == 2
    assert A_sp_datas.shape[0] == A_sp_idxs.shape[0]
    return abstract_arrays.ShapedArray((rhss.shape[0],), rhss.dtype)


def solverSparse_xla_translation(c, A_sp_data, A_sp_idx, rhs):
    A_sp_data_shape = c.get_shape(A_sp_data)
    A_sp_idx_shape = c.get_shape(A_sp_idx)
    rhs_shape = c.get_shape(rhs)

    dtype = A_sp_data_shape.element_type()
    dtype_idx = A_sp_idx_shape.element_type()
    assert dtype == rhs_shape.element_type()

    A_sp_data_dims = A_sp_data_shape.dimensions()
    A_sp_data_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype), A_sp_data_dims, (0,)
    )

    A_sp_idx_dims = A_sp_idx_shape.dimensions()
    A_sp_idx_shape = xla_client.Shape.array_shape(
        jnp.dtype(dtype_idx), A_sp_idx_dims, (0, 1)
    )

    rhs_dims = rhs_shape.dimensions()
    rhs_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), rhs_dims, (0,))

    out_dims = rhs_dims
    out_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), out_dims, (0,))

    op_name = b"solverSparse"

    nnz = A_sp_data_dims[0]
    n = rhs_dims[0]

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(
            A_sp_data,
            A_sp_idx,
            xla_client.ops.ConstantLiteral(c, nnz),
            rhs,
            xla_client.ops.ConstantLiteral(c, n),
        ),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(
            A_sp_data_shape,
            A_sp_idx_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
            rhs_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
        ),
    )
