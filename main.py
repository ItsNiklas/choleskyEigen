from choleskyEigen import solverDense
from choleskyEigen import choleskyDense
import numpy as np

import jax.numpy as jnp
from jaxlib import xla_client
from jax import abstract_arrays, core, xla
from jax import jit

from jax.config import config
config.update("jax_enable_x64", True) #Enable double-precision

def main():
    #print(dir(choleskyEigen))
    solverDense.register()

    # Testing
    f = jit(solverDense.solverDense_prim)

    A = jnp.array([[1,3,-2],[3,5,6],[2,4,3]], dtype=jnp.double)
    A = A @ A.T
    b = jnp.array([5,7,8], dtype=jnp.double)
    assert np.allclose(np.linalg.solve(A,b), f(A,b))

    choleskyDense.register()
    f = jit(choleskyDense.choleskyDense_prim)
    assert np.allclose(np.linalg.cholesky(A), f(A))

if __name__ == "__main__":
    main()