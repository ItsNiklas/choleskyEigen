import jax.numpy as jnp
import numpy as np
from jax import abstract_arrays, core, xla
from jax.core import ShapedArray
from jaxlib import xla_client, xla_extension
from numpy.typing import ArrayLike

import choleskyEigenLib

# Creating our new JAX Primitive
solverDense_p = core.Primitive("solverDense")


def register():
    """Convinent function to register/def all semantics."""
    xla.backend_specific_translations["cpu"][
        solverDense_p
    ] = solverDense_xla_translation
    solverDense_p.def_impl(solverDense_impl)
    solverDense_p.def_abstract_eval(solverDense_abstract_eval)


# Load respective C++ function from the pybind registration as
# a xla custom call target.
for _name, _val in choleskyEigenLib.registrations().items():
    if _name in __name__:
        xla_client.register_cpu_custom_call_target(_name, _val)


# impl
def solverDense_impl(A: ArrayLike, rhs: ArrayLike) -> ArrayLike:
    # Concrete implementation of the primitive.
    # Does not need to be traceable.
    return np.linalg.solve(A, rhs)


# prim
# The actual JAX Primitive being called as a function.
def solverDense_prim(A: ArrayLike, rhs: ArrayLike) -> ArrayLike:
    """Solve using a cholesky decomposition.
    :param A: symmetric, positive-definite matrix.
    :param rhs: right-hand side vector of the same length.
    :return: The solution-vector x such that Ax=rhs.
    """
    return solverDense_p.bind(A, rhs)


# abstract
def solverDense_abstract_eval(As: ShapedArray, rhss: ShapedArray) -> ShapedArray:
    # Abstract evaluation of the primitive. And a few asserts for safety.
    # Will be invoked with abstractions.
    assert len(As.shape) == 2
    assert len(rhss.shape) == 1
    assert As.shape[0] == rhss.shape[0]
    # Returns new ShapedArray with output dimensions and dtype.
    return abstract_arrays.ShapedArray((rhss.shape[0],), rhss.dtype)


# XLA compilation rule
def solverDense_xla_translation(
    c: xla_extension.XlaBuilder, A: xla_extension.XlaOp, rhs: xla_extension.XlaOp
) -> xla_extension.XlaOp:
    # The compilation of the primitive to XLA.
    # Gets called with a XlaBuilder and XlaOps of each argument,
    # returns a combined XlaOp(eration).

    # Extract a few dimensions/element types
    A_shape = c.get_shape(A)
    rhs_shape = c.get_shape(rhs)

    dtype = A_shape.element_type()
    assert dtype == rhs_shape.element_type()

    A_dims = A_shape.dimensions()
    # creating an array_shape: dtype, length of axes, axes used.
    # e.g. f64[3,3]{1,0}
    A_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), A_dims, (0, 1))

    rhs_dims = rhs_shape.dimensions()
    rhs_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), rhs_dims, (0,))

    out_dims = rhs_dims
    out_shape = xla_client.Shape.array_shape(jnp.dtype(dtype), out_dims, (0,))

    op_name = b"solverDense"

    return xla_client.ops.CustomCallWithLayout(
        # XLA Builder Object
        c,
        # C++ function to be called.
        op_name,
        # Xla Op objects of operands (including a new ConstantLiteral)
        operands=(A, rhs, xla_client.ops.ConstantLiteral(c, A_dims[0])),
        # Output shape, forged from the others; type: C-level object
        shape_with_layout=out_shape,
        # Input Shapes (including a new int)
        operand_shapes_with_layout=(
            A_shape,
            rhs_shape,
            xla_client.Shape.array_shape(jnp.dtype(jnp.int64), (), ()),
        ),
    )
