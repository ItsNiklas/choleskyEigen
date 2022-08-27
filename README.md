# choleskyEigen

Trying to bind a few Cholesky-related functions to Jax.

- All Eigen-related code lies in `cholesky.cpp`.
- Individual Jax Primitives are defined in their individual files under `/choleskyEigen`.
- `main.py` exists for testing purposes.

### Installation
Make sure you have Eigen development headers installed.

Run `cmake . && make` to compile the python module. It can then be imported
natively using `import choleskyEigenLib`, although this only includes the
`registrations` dictionary.

The main module lies under `choleskyEigen`.

### Commit
Run `pre-commit run -a` before comitting your work.