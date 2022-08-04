#include <iostream>
#include <pybind11/pybind11.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

namespace py = pybind11;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;

Eigen::VectorXd solveSpLLT(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& rhs){
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> C(A);
    return C.solve(rhs);
}

//Exposable function wrapper for solveLLT
void solverDense(void *out, void **in){
    //A, rhs, rhs.shape
    //Parse Inputs
    auto* A_ptr = reinterpret_cast<double *>(in[0]);
    auto* rhs_ptr = reinterpret_cast<double *>(in[1]);
    auto* out_ptr = reinterpret_cast<double *>(out);
    auto n = *reinterpret_cast<const std::int64_t*>(in[2]);

    MatrixXd A = Map<const MatrixXd>(A_ptr, n, n);
    VectorXd rhs = Map<const VectorXd>(rhs_ptr, n, 1);

    VectorXd res = A.llt().solve(rhs); //Cholesky: A has to be symmetrical positive definite!
    Map<VectorXd>(out_ptr, n, 1) = res;
}

//Exposable function wrapper for matrixL
void choleskyDense(void *out, void **in){
    //A, n
    //Parse Inputs
    auto* A_ptr = reinterpret_cast<double *>(in[0]);
    auto* out_ptr = reinterpret_cast<double *>(out);
    auto n = *reinterpret_cast<const std::int64_t*>(in[1]);

    MatrixXd A = Map<const MatrixXd>(A_ptr, n, n);

    MatrixXd res = A.llt().matrixL(); //Cholesky: A has to be symmetrical positive definite!
    Map<MatrixXd>(out_ptr, n, n) = res;
}


//Fill pybind dictionary
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["solverDense"] = pybind11::capsule((void*)solverDense, "xla._CUSTOM_CALL_TARGET");
    dict["choleskyDense"] = pybind11::capsule((void*) choleskyDense, "xla._CUSTOM_CALL_TARGET");
    return dict;
}

// Expose functions in Registrations dictionary
PYBIND11_MODULE(choleskyEigenLib, m) { m.def("registrations", &Registrations); }