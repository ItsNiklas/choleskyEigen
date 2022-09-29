import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps
import scipy.stats as spt

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
    f = jax.jit(solverDense.solverDense_prim)

    # Example matricies
    A = jnp.array([[1, 3, -2], [3, 5, 6], [2, 4, 3]], dtype=jnp.double)
    A = A @ A.T  # make sym, pos.-semidef.
    b = jnp.array([5, 7, 8], dtype=jnp.double)

    n = 50
    A = sps.rand(n, n, density=0.2, dtype=jnp.double)
    A = A @ A.T
    A = A.todense() + 10 * np.eye(n)
    b = np.random.rand(n) * 10

    assert np.allclose(np.linalg.solve(A, b), f(A, b))

    choleskyDense.register()
    f = jax.jit(choleskyDense.choleskyDense_prim)
    assert np.allclose(np.linalg.cholesky(A), f(A))

    solverSparse.register()
    A_sp = sparse.BCOO.fromdense(A)
    f = jax.jit(solverSparse.solverSparse_prim)
    assert np.allclose(np.linalg.solve(A, b), f(A_sp, b))

    choleskySparse.register()
    f = choleskySparse.choleskySparse
    # Creating an n x 1 array containing n works.
    L = f(A_sp).todense()
    assert np.allclose(L @ L.T, A)

    # MVN
    rng = np.random.default_rng(seed=2)

    sps_mvn_sample_and_log_prob.register()

    n = 10
    seed = jax.random.PRNGKey(1337)
    inv_cov = np.cov(rng.standard_normal((n, n)))
    mean = rng.standard_normal(n)

    sample_updated, log_prob_updated = sps_mvn_sample_and_log_prob.sps_mvn(
        seed, mean, inv_cov
    )
    """
    assert np.allclose(
        sample_updated, mean + np.linalg.solve(inv_cov.todense(), sample)
    )
    assert np.allclose(
        log_prob_updated,
        log_prob + np.sum(np.log(np.diag(np.linalg.cholesky(inv_cov.todense())))),
    )
    """

    # print(sample_updated, log_prob_updated)

    # g = jax.grad(lambda *args: sps_mvn_sample_and_log_prob.sps_mvn(*args)[1], argnums=(1,2))
    # sample_updated, log_prob_updated = g(seed, mean, inv_cov)
    # print(sample_updated, log_prob_updated)

    # def f0(sample, mean, inv_cov):
    #    cov = jnp.linalg.inv(inv_cov)
    #    return jax.scipy.stats.multivariate_normal.logpdf(sample, mean, cov)

    # sample, _ = sps_mvn_sample_and_log_prob.sps_mvn(seed, mean, inv_cov)
    # print(f0(sample, mean, inv_cov))


if __name__ == "__main__":
    main()
