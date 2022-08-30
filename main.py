import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
from jax import jit

# Enable JAX double-precision at startup.
# Our Eigen functions currently work with double-precision.
from jax.config import config
from jax.experimental import sparse

from choleskyEigen import choleskyDense, choleskySparse, solverDense, solverSparse

config.update("jax_enable_x64", True)


# Just a function to test the library.
def main():
    # Testing
    # register() is used to actually activate the JAX Primitive.
    solverDense.register()
    f = jit(solverDense.solverDense_prim)

    # Example matricies
    A = jnp.array([[1, 3, -2], [3, 5, 6], [2, 4, 3]], dtype=jnp.double)
    A = A @ A.T  # make sym, pos.-semidef.
    b = jnp.array([5, 7, 8], dtype=jnp.double)

    A = sps.rand(100, 100, density=0.5, dtype=jnp.double)
    A = A @ A.T
    A = A.todense()
    b = np.random.rand(100) * 10

    assert np.allclose(np.linalg.solve(A, b), f(A, b))

    choleskyDense.register()
    f = jit(choleskyDense.choleskyDense_prim)
    assert np.allclose(np.linalg.cholesky(A), f(A))

    solverSparse.register()
    A_sp = sparse.BCOO.fromdense(A)
    f = jit(solverSparse.solverSparse_prim)
    assert np.allclose(np.linalg.solve(A, b), f(A_sp, b))

    choleskySparse.register()
    f = jit(choleskySparse.choleskySparse)
    # Creating an n x 1 array containing n works.
    L = f(A_sp, jnp.repeat(A.shape[0], A.shape[0]))
    assert np.allclose(L @ L.T, A)


if __name__ == "__main__":
    main()
