import choleskyEigenLib
import jax.numpy as jnp
import numpy as np
from jax import abstract_arrays, core, xla
from jaxlib import xla_client

choleskyDense_p = core.Primitive("choleskyDense")
# See solverDense.py for extensive comments.
# New bindings are just copied and adjusted.


def register():
    # Register module
    xla.backend_specific_translations["cpu"][
        choleskyDense_p
    ] = choleskyDense_xla_translation
    choleskyDense_p.def_impl(choleskyDense_impl)
    choleskyDense_p.def_abstract_eval(choleskyDense_abstract_eval)


# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def choleskyDense_impl(A):
    return np.linalg.cholesky(A)


# prim
def choleskyDense_prim(A):
    return choleskyDense_p.bind(A)


# abstract
def choleskyDense_abstract_eval(As):
    return abstract_arrays.ShapedArray(As.shape, As.dtype)


def choleskyDense_xla_translation(c, A):
    A_shape = c.get_shape(A)
    dtype = A_shape.element_type()
    A_dims = A_shape.dimensions()
    A_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), A_dims, (0, 1))

    out_dims = A_dims
    out_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), out_dims, (0, 1))

    op_name = b"choleskyDense"

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands=(A, xla_client.ops.ConstantLiteral(c, A_dims[0])),
        shape_with_layout=out_shape,
        operand_shapes_with_layout=(
            A_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
        ),
    )
