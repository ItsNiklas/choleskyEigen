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
void solverDense(void *out, void **in){
    //A, rhs, n
    //Parse Inputs
    auto* A_ptr = reinterpret_cast<double *>(in[0]);
    auto* rhs_ptr = reinterpret_cast<double *>(in[1]);
    auto n = *reinterpret_cast<const std::int64_t*>(in[2]);
    auto* out_ptr = reinterpret_cast<double *>(out);

    MatrixXd A = Map<const MatrixXd>(A_ptr, n, n);
    VectorXd rhs = Map<const VectorXd>(rhs_ptr, n);

    //Cholesky: A has to be symmetrical positive definite!
    Map<VectorXd>(out_ptr, n, 1) = A.llt().solve(rhs);
}

//Exposable function wrapper for matrixL
void choleskyDense(void *out, void **in){
    //A, n
    //Parse Inputs
    auto* A_ptr = reinterpret_cast<double *>(in[0]);
    auto n = *reinterpret_cast<const std::int64_t*>(in[1]);
    auto* out_ptr = reinterpret_cast<double *>(out);

    MatrixXd A = Map<const MatrixXd>(A_ptr, n, n);

    //Cholesky: A has to be symmetrical positive definite!
    Map<MatrixXd>(out_ptr, n, n) = A.llt().matrixL();
}

//Exposable function wrapper for sparse solveLLT
void solverSparse(void *out, void **in){
    //A_sp_data, A_sp_idx, nnz, rhs, n
    //Parse Inputs
    auto* A_sp_data_ptr = reinterpret_cast<double *>(in[0]);
    auto* A_sp_idx_ptr = reinterpret_cast<int *>(in[1]);
    auto nnz = *reinterpret_cast<const std::int64_t*>(in[2]); //number of non-zeros
    auto* rhs_ptr = reinterpret_cast<double *>(in[3]);
    auto n = *reinterpret_cast<const std::int64_t*>(in[4]);

    VectorXd A_sp_data = Map<const VectorXd>(A_sp_data_ptr, nnz);
    MatrixXi A_sp_idx = Map<const MatrixXi>(A_sp_idx_ptr, nnz, 2);
    VectorXd rhs = Map<const VectorXd>(rhs_ptr, n);

    auto* out_ptr = reinterpret_cast<double *>(out);

    //Create SparseMatrix
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(nnz);
    for (int i = 0; i < nnz; i++) {
        tripletList.emplace_back(A_sp_idx(i,0), A_sp_idx(i,1), A_sp_data(i));
    }
    Eigen::SparseMatrix<double> A_sp(n,n);
    A_sp.setFromTriplets(tripletList.begin(), tripletList.end());
    //

    //Solve
    static Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

    solver.analyzePattern(A_sp);
    solver.factorize(A_sp);
    Map<VectorXd>(out_ptr, n) = solver.solve(rhs);
}

//Exposable function wrapper for sparse cholesky-matrix L
void choleskySparse(void *out, void **in){
    //A_sp_data, A_sp_idx, nnz, n
    //Parse Inputs
    auto* A_sp_data_ptr = reinterpret_cast<double *>(in[0]);
    auto* A_sp_idx_ptr = reinterpret_cast<double *>(in[1]);
    auto nnz = *reinterpret_cast<const std::int64_t*>(in[2]); //number of non-zeros
    auto n = *reinterpret_cast<const std::int64_t*>(in[3]);

    VectorXd A_sp_data = Map<const VectorXd>(A_sp_data_ptr, nnz);
    MatrixXd A_sp_idx = Map<const MatrixXd>(A_sp_idx_ptr, 2, nnz);

    auto* out_ptr = reinterpret_cast<double *>(out);

    //Create SparseMatrix
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(nnz);
    for (int i = 0; i < nnz; i++) {
        tripletList.emplace_back(A_sp_idx(0,i), A_sp_idx(1,i), A_sp_data(i));
    }
    Eigen::SparseMatrix<double> A_sp(n,n);
    A_sp.setFromTriplets(tripletList.begin(), tripletList.end());
    //

    //Solve
    static Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

    solver.analyzePattern(A_sp);
    solver.factorize(A_sp);
    Map<MatrixXd>(out_ptr, n,n) = solver.matrixL();
}


//Fill pybind dictionary
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["solverDense"] = pybind11::capsule((void*) solverDense, "xla._CUSTOM_CALL_TARGET");
    dict["choleskyDense"] = pybind11::capsule((void*) choleskyDense, "xla._CUSTOM_CALL_TARGET");
    dict["solverSparse"] = pybind11::capsule((void*) solverSparse, "xla._CUSTOM_CALL_TARGET");
    dict["choleskySparse"] = pybind11::capsule((void*) choleskySparse, "xla._CUSTOM_CALL_TARGET");
    return dict;
}

// Expose functions in Registrations dictionary
PYBIND11_MODULE(choleskyEigenLib, m) { m.def("registrations", &Registrations); }