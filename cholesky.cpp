#include <iostream>
#include <pybind11/pybind11.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

namespace py = pybind11;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;

//Exposable function wrapper for solveLLT
void solverDense(void *out, void **in) {
    /* Arguments:
     * A, symmetrical positive definite Matrix with dimensions n x n.
     * rhs, right-hand side vector to be solved for with length n.
     * n, dimension.
     * Returns:
     * Vector x that solves Ax=rhs, computed with a Cholesky decomposition
     */

    //Parse and cast pointers
    auto *A_ptr = reinterpret_cast<double *>(in[0]);
    auto *rhs_ptr = reinterpret_cast<double *>(in[1]);
    auto n = *reinterpret_cast<const std::int64_t *>(in[2]);
    auto *out_ptr = reinterpret_cast<double *>(out);

    //Map pointers to Eigen data-structures
    MatrixXd A = Map<const MatrixXd>(A_ptr, n, n);
    VectorXd rhs = Map<const VectorXd>(rhs_ptr, n);

    //Get LLT solver object of A, and solve for rhs
    //Map data into out pointer
    Map<VectorXd>(out_ptr, n) = A.llt().solve(rhs);
}

//Exposable function wrapper for matrixL
void choleskyDense(void *out, void **in) {
    /* Arguments:
     * A, symmetrical positive definite Matrix with dimensions n x n.
     * n, dimension.
     * Returns:
     * The Cholesky decomposition of A, as a lower triangular matrix.
     */

    //Parse and cast pointers
    auto *A_ptr = reinterpret_cast<double *>(in[0]);
    auto n = *reinterpret_cast<const std::int64_t *>(in[1]);
    auto *out_ptr = reinterpret_cast<double *>(out);

    //Map pointers to Eigen data-structures
    MatrixXd A = Map<const MatrixXd>(A_ptr, n, n);

    //Get LLT solver object of A, and request the lower triangular decomposition.
    //Map data into out pointer
    Map<MatrixXd>(out_ptr, n, n) = A.llt().matrixL();
}

//Exposable function wrapper for sparse solveLLT
void solverSparse(void *out, void **in) {
    /* Solve Ax=b, when the symmetrical positive definite Matrix A is stored sprse.
    * Arguments:
    * A_sp_data, Vector of length nnz that contains the non-zero values of A.
    * A_sp_idx, nnz x 2 matrix that contains the indicies of the respective data.
    * nnz, Number of non-zeroes.
    * rhs, right-hand side vector to be solved for with length n.
    * n, dimension of the dense representation of A.
    * Returns:
    * Vector x that solves Ax=rhs, computed with a Cholesky decomposition
    */

    //Parse and cast pointers
    auto *A_sp_data_ptr = reinterpret_cast<double *>(in[0]);
    auto *A_sp_idx_ptr = reinterpret_cast<int *>(in[1]);
    auto nnz = *reinterpret_cast<const std::int64_t *>(in[2]);
    auto *rhs_ptr = reinterpret_cast<double *>(in[3]);
    auto n = *reinterpret_cast<const std::int64_t *>(in[4]);
    auto *out_ptr = reinterpret_cast<double *>(out);

    //Map pointers to Eigen data-structures
    VectorXd A_sp_data = Map<const VectorXd>(A_sp_data_ptr, nnz);
    MatrixXi A_sp_idx = Map<const MatrixXi>(A_sp_idx_ptr, nnz, 2);
    VectorXd rhs = Map<const VectorXd>(rhs_ptr, n);

    //Create our SparseMatrix object
    std::vector <Eigen::Triplet<double>> tripletList(nnz);
    for (int i = 0; i < nnz; i++) {
        //Create tuples with values (index0, index1, data)
        tripletList.emplace_back(
                A_sp_idx(i, 0), A_sp_idx(i, 1), A_sp_data(i));
    }
    Eigen::SparseMatrix<double> A_sp(n, n);
    //Set matrix with created tuples
    A_sp.setFromTriplets(tripletList.begin(), tripletList.end());

    //Create sparse solver object for our matrix class
    static Eigen::SimplicialLLT <Eigen::SparseMatrix<double>> solver;

    //Calculate decomposition
    solver.analyzePattern(A_sp);
    solver.factorize(A_sp);

    //Solve for rhs
    //Map data into out pointer
    Map<VectorXd>(out_ptr, n) = solver.solve(rhs);
}

//Exposable function wrapper for sparse cholesky-matrix L
void choleskySparse(void *out, void **in) {
    /* Solve Ax=b, when the symmetrical positive definite Matrix A is stored sprse.
    * Arguments:
    * A_sp_data, Vector of length nnz that contains the non-zero values of A.
    * A_sp_idx, nnz x 2 matrix that contains the indicies of the respective data.
    * nnz, Number of non-zeroes.
    * n, dimension of the dense representation of A.
    * Returns:
    * The Cholesky decomposition of A, as a lower triangular matrix.
    */

    //Parse and cast pointers
    auto *A_sp_data_ptr = reinterpret_cast<double *>(in[0]);
    auto *A_sp_idx_ptr = reinterpret_cast<int *>(in[1]);
    auto nnz = *reinterpret_cast<const std::int64_t *>(in[2]);
    auto n = *reinterpret_cast<const std::int64_t *>(in[3]);
    auto *out_ptr = reinterpret_cast<double *>(out);

    //Map pointers to Eigen data-structures
    VectorXd A_sp_data = Map<const VectorXd>(A_sp_data_ptr, nnz);
    MatrixXi A_sp_idx = Map<const MatrixXi>(A_sp_idx_ptr, nnz, 2);

    //Create our SparseMatrix object
    std::vector<Eigen::Triplet<double>> tripletList(nnz);
    for (int i = 0; i < nnz; i++) {
        //Create tuples with values (index0, index1, data)
        tripletList.emplace_back(
                A_sp_idx(i, 0), A_sp_idx(i, 1), A_sp_data(i));
    }
    Eigen::SparseMatrix<double> A_sp(n, n);
    //Set matrix with created tuples
    A_sp.setFromTriplets(tripletList.begin(), tripletList.end());

    //Create sparse solver object for our matrix class
    //Advanced constructor, as we need to clarify NaturalOrdering instead of
    //an optimized one. Otherwise, L will be wrong.
    static Eigen::SimplicialLLT<Eigen::SparseMatrix<double>,
                                Eigen::Lower,
                                Eigen::NaturalOrdering<int>> solver;

    //Calculate decomposition
    solver.analyzePattern(A_sp);
    solver.factorize(A_sp);

    //Request the lower triangular decomposition.
    Eigen::SparseMatrix<double> L_sp = solver.matrixL();

    //Possible Vectors to reconstruct a BCOO matrix in Python
    //Map<Eigen::VectorXi>(L_sp.outerIndexPtr(), L_sp.outerSize()+1);
    //Map<Eigen::VectorXi>(L_sp.innerIndexPtr(), L_sp.nonZeros());
    //Map<Eigen::VectorXd>(L_sp.valuePtr(), L_sp.nonZeros());

    //Map data into out pointer (dense!)
    Map<MatrixXd>(out_ptr, n, n) = L_sp;
}


//Fill a pybind dictionary with function pointers that should be exposed.
//Functions are capsuled with a XLA_CUSTOM_CALL_TARGET tag.
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["solverDense"] =pybind11::capsule((void *) solverDense,
                                           "xla._CUSTOM_CALL_TARGET");
    dict["choleskyDense"] = pybind11::capsule((void *) choleskyDense,
                                              "xla._CUSTOM_CALL_TARGET");
    dict["solverSparse"] = pybind11::capsule((void *) solverSparse,
                                             "xla._CUSTOM_CALL_TARGET");
    dict["choleskySparse"] = pybind11::capsule((void *) choleskySparse,
                                               "xla._CUSTOM_CALL_TARGET");
    return dict;
}

// Expose function dictionary as the Python module 'choleskyEigenLib'
PYBIND11_MODULE(choleskyEigenLib, m
){
m.def("registrations", &Registrations);
}
