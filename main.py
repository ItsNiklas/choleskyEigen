import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import scipy.stats as spt
from jax import jit

# Enable JAX double-precision at startup.
# Our Eigen functions currently work with double-precision.
from jax.config import config
from jax.experimental import sparse

from choleskyEigen import (
    choleskyDense,
    choleskySparse,
    solverDense,
    solverSparse,
    sps_mvn_sample_and_log_prob,
)

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

    # MVN
    rng = np.random.default_rng(seed=2)

    sps_mvn_sample_and_log_prob.register()
    f = jit(sps_mvn_sample_and_log_prob.sps_mvn_sample_and_log_prob)

    n = 5
    cov = rng.normal(size=(n, n))
    cov = cov @ cov.T * np.eye(n)
    inv_cov = sparse.BCOO.fromdense(np.linalg.inv(cov))
    mean = rng.uniform(size=n)
    sample = rng.uniform(size=n)
    log_prob = np.sum(spt.norm.logpdf(sample))

    sample, log_prob = f(mean, inv_cov, sample, log_prob)

    print(sample, log_prob)


if __name__ == "__main__":
    main()
