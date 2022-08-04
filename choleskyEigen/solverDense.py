import choleskyEigenLib
import numpy as np

import jax.numpy as jnp
from jaxlib import xla_client
from jax import abstract_arrays, core, xla
from jax import jit

solverDense_p = core.Primitive("solverDense")

def register():
    # Register module
    xla.backend_specific_translations["cpu"][solverDense_p] = solverDense_xla_translation
    solverDense_p.def_impl(solverDense_impl)
    solverDense_p.def_abstract_eval(solverDense_abstract_eval)

# Register the XLA custom calls
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)

# impl
def solverDense_impl(A, rhs):
    return np.linalg.solve(A, rhs)

# prim
def solverDense_prim(A, rhs):
    return solverDense_p.bind(A, rhs)

# abstract
def solverDense_abstract_eval(As, rhss):
    assert len(As.shape) == 2
    assert len(rhss.shape) == 1
    assert As.shape[0] == rhss.shape[0]
    return abstract_arrays.ShapedArray((rhss.shape[0],), rhss.dtype)

def solverDense_xla_translation(c, A, rhs):
    A_shape = c.get_shape(A)
    rhs_shape = c.get_shape(rhs)

    dtype = A_shape.element_type()
    assert dtype == rhs_shape.element_type()

    A_dims = A_shape.dimensions()
    A_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), A_dims, (0,1))

    rhs_dims = rhs_shape.dimensions()
    rhs_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), rhs_dims, (0,))

    out_dims = rhs_dims
    out_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), out_dims, (0,))

    op_name = b"solverDense"

    return xla_client.ops.CustomCallWithLayout(
        c,
        op_name,
        operands = (A, rhs, xla_client.ops.ConstantLiteral(c, A_dims[0])),
        shape_with_layout = out_shape,
        operand_shapes_with_layout = (A_shape, rhs_shape, xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ())),
    )